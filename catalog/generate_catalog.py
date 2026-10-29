import os
import json
import time
import duckdb
from google import genai
from google.genai.errors import ServerError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

TABLE_FILES = [
    ("silver_studies", "data/tables/silver_studies.parquet"),
    ("silver_conditions", "data/tables/silver_conditions.parquet"),
    ("silver_interventions", "data/tables/silver_interventions.parquet"),
    ("gold_trials_summary", "data/tables/gold_trials_summary.parquet")
]

@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((ServerError, ClientError)),
    reraise=True
)
def call_gemini_with_retry(prompt: str) -> dict:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

def extract_and_enrich():
    os.makedirs("data", exist_ok=True)
    con = duckdb.connect()
    catalog = []

    for name, path in TABLE_FILES:
        if not os.path.exists(path):
            print(f"[WARNING] Skipping {name}, file not found.")
            continue

        schema_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]

        columns_data = [
            {"name": col[0], "type": col[1], "nullable": col[2]}
            for col in schema_info
        ]

        print(f"[INFO] Generating AI documentation for table '{name}' ({row_count} rows)...")

        prompt = f"""
You are a Lead Healthcare Data Architect.
Provide production data catalog documentation for this table:
Table Name: {name}
Columns: {json.dumps(columns_data)}

Return your answer strictly as a valid JSON object matching this schema:
{{
  "table_description": "2-sentence plain English summary of what this table represents",
  "columns": {{
    "column_name": "1-sentence plain English description of what this column stores"
  }}
}}
"""
        ai_data = call_gemini_with_retry(prompt)

        catalog.append({
            "table_name": name,
            "row_count": row_count,
            "table_description": ai_data.get("table_description", ""),
            "columns": [
                {
                    "name": col["name"],
                    "type": col["type"],
                    "description": ai_data.get("columns", {}).get(col["name"], "No description available")
                }
                for col in columns_data
            ]
        })

        # Short pause between tables to respect free-tier burst limits
        time.sleep(3)

    with open("data/enriched_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    print("[SUCCESS] Metadata catalog written to data/enriched_catalog.json")

if __name__ == "__main__":
    extract_and_enrich()