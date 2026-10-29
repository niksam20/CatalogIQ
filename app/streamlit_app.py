import os
import json
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai.errors import ServerError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

st.set_page_config(page_title="CatalogIQ Explorer", page_icon="🔬", layout="wide")
st.title("🔬 CatalogIQ: Autonomous Clinical Data Catalog")
st.markdown("Query clinical trial schemas, column definitions, and table statistics using natural language.")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type((ServerError, ClientError)),
    reraise=True
)
def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    return response.text

@st.cache_resource
def setup_vector_store():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="catalog_metadata",
        embedding_function=embed_fn
    )

    if os.path.exists("data/enriched_catalog.json"):
        with open("data/enriched_catalog.json", "r") as f:
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
            ids.append(f"doc_{idx}")

        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

    return collection

collection = setup_vector_store()

query = st.text_input(
    "Ask a question about the Lakehouse schemas or columns:", 
    placeholder="e.g. Which table contains trial phases, and what is its primary key?"
)

if query:
    with st.spinner("Retrieving schema context and generating answer..."):
        results = collection.query(query_texts=[query], n_results=2)
        context = "\n---\n".join(results["documents"][0]) if results["documents"] else "No context found."

        prompt = f"""
You are CatalogIQ, an AI Data Catalog Assistant.
Answer the user's question strictly using the documented Lakehouse metadata below.
If the information is not present in the context, say: "This information is not documented in the current catalog."

Context:
{context}

Question:
{query}
"""
        answer = ask_gemini(prompt)

    st.markdown("### Assistant Answer")
    st.info(answer)

    with st.expander("Inspected Catalog Schema Context"):
        st.text(context)