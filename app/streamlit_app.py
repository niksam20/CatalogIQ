import os
import json
from pathlib import Path
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Ensure fast local loading of cached sentence-transformer weights
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Must be the first Streamlit command
st.set_page_config(page_title="CatalogIQ Explorer", page_icon="🔬", layout="wide")

gemini_api_key = os.getenv("GEMINI_API_KEY")

st.title("🔬 CatalogIQ: Autonomous Clinical Data Catalog")
st.markdown("Query clinical trial schemas, column definitions, and table statistics using natural language powered by Google Gemini.")

if not gemini_api_key:
    st.error(
        "Missing **GEMINI_API_KEY**. Please add your Gemini API key to the `.env` file in the project root to enable the AI assistant.",
        icon=":material/error:"
    )
    st.stop()

client = genai.Client(api_key=gemini_api_key)
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (408, 429, 500, 502, 503, 504):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return False


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=2, max=12),
    retry=retry_if_exception(is_retryable_error),
    reraise=True
)
def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text


catalog_path = PROJECT_ROOT / "data/enriched_catalog.json"
catalog_mtime = catalog_path.stat().st_mtime if catalog_path.exists() else 0.0


@st.cache_resource
def setup_vector_store(mtime_key: float):
    chroma_dir = str(PROJECT_ROOT / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            local_files_only=True
        )
    except Exception:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    collection = chroma_client.get_or_create_collection(
        name="catalog_metadata",
        embedding_function=embed_fn
    )

    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        docs, metadatas, ids = [], [], []
        for idx, table in enumerate(catalog):
            doc = (
                f"Table: {table['table_name']}\n"
                f"Purpose: {table['table_description']}\n"
                f"Total Records: {table['row_count']}\n"
                "Columns:\n"
            )
            for col in table["columns"]:
                doc += f" - {col['name']} ({col['type']}): {col['description']}\n"

            docs.append(doc)
            metadatas.append({"table_name": table["table_name"]})
            ids.append(f"table_{table['table_name']}")

        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

    return collection


collection = setup_vector_store(catalog_mtime)

# AI Assistant Search Bar
query = st.text_input(
    "Ask a question about the lakehouse schemas or columns:",
    placeholder="e.g. Which table contains trial phases, and what is its primary key?"
)

if query:
    total_docs = collection.count()
    if total_docs == 0:
        st.warning("No metadata catalog documents indexed. Please generate the catalog first.", icon=":material/warning:")
    else:
        with st.spinner("Retrieving schema context and generating answer with Gemini..."):
            n_results = min(4, total_docs)
            results = collection.query(query_texts=[query], n_results=n_results)
            retrieved_docs = results.get("documents", [[]])[0]
            context = "\n---\n".join(retrieved_docs) if retrieved_docs else "No context found."

            prompt = f"""You are CatalogIQ, an AI Data Catalog Assistant.
Answer the user's question strictly using the documented Lakehouse metadata below.
If the information is not present in the context, say: "This information is not documented in the current catalog."

Context:
{context}

Question:
{query}
"""
            try:
                answer = ask_gemini(prompt)
                st.subheader("Assistant response")
                st.info(answer, icon=":material/psychology:")

                with st.expander("Inspected catalog schema context"):
                    st.text(context)
            except Exception as exc:
                st.error(f"Failed to query Gemini assistant ({type(exc).__name__}): {exc}", icon=":material/error:")

st.divider()
st.subheader("Browse enriched data catalog")

if catalog_path.exists():
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    for tbl in catalog_data:
        with st.expander(f"📁 {tbl['table_name']} ({tbl['row_count']} rows)"):
            st.markdown(f"**Description:** {tbl['table_description']}")
            cols_display = [
                {
                    "Column": c.get("name"),
                    "Type": c.get("type"),
                    "Description": c.get("description", "No description available")
                }
                for c in tbl.get("columns", [])
            ]
            st.dataframe(cols_display, width="stretch", hide_index=True)
else:
    st.warning(
        "Catalog metadata file `data/enriched_catalog.json` not found. Please run `python catalog/generate_catalog.py` to generate documentation.",
        icon=":material/warning:"
    )
