import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TABLE_FILES = [
    ("silver_studies", PROJECT_ROOT / "data/tables/silver_studies.parquet"),
    ("silver_conditions", PROJECT_ROOT / "data/tables/silver_conditions.parquet"),
    ("silver_interventions", PROJECT_ROOT / "data/tables/silver_interventions.parquet"),
    ("gold_trials_summary", PROJECT_ROOT / "data/tables/gold_trials_summary.parquet"),
]

OUTPUT_FILE = PROJECT_ROOT / "data/enriched_catalog.json"


def get_gemini_client():
    """Initialize Google Gemini client lazily from environment variables."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY was not found. Please set it in your root .env file.")
    client = genai.Client(api_key=gemini_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return client, model


def create_catalog_schema(columns_data: list) -> dict:
    column_properties = {
        col["name"]: {
            "type": "STRING",
            "description": f"Plain English explanation of the column {col['name']}."
        }
        for col in columns_data
    }

    return {
        "type": "OBJECT",
        "properties": {
            "table_description": {
                "type": "STRING",
                "description": "Two-sentence plain English description of what the table represents."
            },
            "columns": {
                "type": "OBJECT",
                "properties": column_properties,
                "required": list(column_properties.keys())
            }
        },
        "required": ["table_description", "columns"]
    }


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (408, 429, 500, 502, 503, 504):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, json.JSONDecodeError)):
        return True
    return False


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=2, max=15),
    retry=retry_if_exception(is_retryable_error),
    reraise=True
)
def call_gemini_with_retry(client, model: str, prompt: str, schema: dict) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a Lead Healthcare Data Architect. "
                "Generate accurate, professional and concise documentation "
                "for healthcare clinical-trial data warehouse tables."
            ),
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )
    )
    content = response.text
    if not content:
        raise ValueError("Gemini returned an empty response.")
    return json.loads(content)


def generate_table_documentation(client, model: str, table_name: str, columns_data: list) -> dict:
    schema = create_catalog_schema(columns_data)
    prompt = f"""Generate professional data catalog documentation for the following healthcare data warehouse table.

TABLE NAME:
{table_name}

TABLE METADATA:
{json.dumps(columns_data, indent=2)}

REQUIREMENTS:
1. Provide a two-sentence description of the table in "table_description".
2. In "columns", explain every column name with a concise, one-sentence plain English description.
3. Use healthcare and clinical-trial terminology wherever appropriate.
4. Output strictly valid JSON matching the requested structure.
"""
    return call_gemini_with_retry(client, model, prompt, schema)


def load_existing_catalog() -> dict:
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {item["table_name"]: item for item in data}
        except Exception:
            return {}
    return {}


def save_catalog(catalog: list):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    temp_file.replace(OUTPUT_FILE)


def process_single_table(table_name: str, file_path: Path, existing_catalog: dict, client, model: str, force_regenerate: bool, refresh_counts_only: bool) -> dict:
    import duckdb

    if not file_path.exists():
        print(f"[WARNING] Table file not found: {file_path}")
        return None

    con = duckdb.connect()
    try:
        escaped_path = str(file_path).replace("'", "''")
        schema_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')").fetchall()
        row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{escaped_path}')").fetchone()[0]
    finally:
        con.close()

    columns_data = [
        {"name": col[0], "type": col[1], "nullable": col[2]}
        for col in schema_info
    ]

    cached = existing_catalog.get(table_name)
    cached_cols = {c["name"]: c.get("description", "") for c in cached.get("columns", [])} if cached else {}
    
    # Check if schema has changed
    schema_unchanged = cached and set(cached_cols.keys()) == {c["name"] for c in columns_data}

    # Use cache if schema unchanged and regeneration not forced
    if (refresh_counts_only or (schema_unchanged and not force_regenerate)) and cached:
        print(f"[INFO] '{table_name}': Reusing cached documentation (row count: {row_count}).")
        table_desc = cached.get("table_description", f"Data table for {table_name}.")
        col_descriptions = cached_cols
    else:
        table_desc = None
        col_descriptions = {}
        if client and not refresh_counts_only:
            try:
                print(f"[INFO] Requesting Gemini documentation for '{table_name}'...")
                ai_data = generate_table_documentation(client, model, table_name, columns_data)
                table_desc = ai_data.get("table_description")
                col_descriptions = ai_data.get("columns", {})
            except Exception as exc:
                print(f"[WARNING] Gemini call failed for '{table_name}' ({exc}). Falling back to cached descriptions if available.")

        if not table_desc and cached:
            table_desc = cached.get("table_description", "")
            col_descriptions = cached_cols

        if not table_desc:
            table_desc = f"Cleaned and curated clinical trials data table for {table_name}."

    enriched_columns = [
        {
            "name": col["name"],
            "type": col["type"],
            "description": col_descriptions.get(col["name"], f"Stores the {col['name']} attribute for the trial record.")
        }
        for col in columns_data
    ]

    return {
        "table_name": table_name,
        "row_count": row_count,
        "table_description": table_desc,
        "columns": enriched_columns
    }


def extract_and_enrich(refresh_counts_only: bool = False, force_regenerate: bool = False, parallel: bool = True):
    existing_catalog = load_existing_catalog()

    client, model = None, None
    if not refresh_counts_only:
        try:
            client, model = get_gemini_client()
            print(f"[INFO] Using Google Gemini (model: {model})")
        except Exception as e:
            print(f"[WARNING] Could not initialize Gemini client ({e}). Falling back to cached descriptions if available.")

    results_dict = {}
    start_time = time.time()

    # Process tables in parallel for maximum speed
    if parallel and len(TABLE_FILES) > 1:
        print(f"[INFO] Processing {len(TABLE_FILES)} tables in parallel...")
        with ThreadPoolExecutor(max_workers=len(TABLE_FILES)) as executor:
            future_to_table = {
                executor.submit(
                    process_single_table,
                    table_name,
                    file_path,
                    existing_catalog,
                    client,
                    model,
                    force_regenerate,
                    refresh_counts_only
                ): table_name
                for table_name, file_path in TABLE_FILES
            }
            for future in as_completed(future_to_table):
                t_name = future_to_table[future]
                try:
                    res = future.result()
                    if res:
                        results_dict[t_name] = res
                        print(f"[SUCCESS] Table '{t_name}' processed.")
                except Exception as exc:
                    print(f"[ERROR] Error processing '{t_name}': {exc}")
    else:
        for table_name, file_path in TABLE_FILES:
            res = process_single_table(
                table_name,
                file_path,
                existing_catalog,
                client,
                model,
                force_regenerate,
                refresh_counts_only
            )
            if res:
                results_dict[table_name] = res

    # Ensure all tables are represented: use newly processed result, or preserve existing cached entry
    ordered_catalog = []
    for table_name, _ in TABLE_FILES:
        if table_name in results_dict:
            ordered_catalog.append(results_dict[table_name])
        elif table_name in existing_catalog:
            ordered_catalog.append(existing_catalog[table_name])

    save_catalog(ordered_catalog)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("CATALOG GENERATION SUMMARY")
    print("=" * 60)
    print(f"Tables processed: {len(ordered_catalog)}/{len(TABLE_FILES)}")
    print(f"Elapsed time:     {elapsed:.2f} seconds")
    print(f"Output file:      {OUTPUT_FILE}")


if __name__ == "__main__":
    counts_only = "--refresh-counts-only" in sys.argv
    force = "--force" in sys.argv
    no_parallel = "--no-parallel" in sys.argv

    try:
        extract_and_enrich(
            refresh_counts_only=counts_only,
            force_regenerate=force,
            parallel=not no_parallel
        )
        print("\n[SUCCESS] Pipeline execution completed successfully.")
    except Exception as exc:
        print("\n[PIPELINE FAILED]")
        print(f"Error type: {type(exc).__name__}")
        print(f"Reason:     {exc}")
        sys.exit(1)
