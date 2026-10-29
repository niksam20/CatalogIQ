# CatalogIQ

CatalogIQ is an AI-assisted data catalog for clinical-trial data. It collects
clinical-trial records from ClinicalTrials.gov, stores raw batches in an AWS S3
bronze zone, transforms them into DuckDB/Parquet silver and gold tables,
generates human-readable metadata with Google Gemini, and exposes that metadata
through a natural-language Streamlit explorer.

## What it does

- **Ingests clinical trials** from the ClinicalTrials.gov API, currently
  searching for oncology studies.
- **Stores raw API responses** as JSON batches in S3 under
  `bronze/studies/year=<YYYY>/month=<MM>/`.
- **Transforms nested study JSON** with DuckDB into normalized tables:
  - `silver_studies`
  - `silver_conditions`
  - `silver_interventions`
  - `gold_trials_summary`
- **Checks data quality** by verifying that `silver_studies.nct_id` is unique.
- **Documents the tables with Gemini**, including table descriptions, row
  counts, column types, and plain-English column definitions.
- **Indexes catalog metadata in ChromaDB** with
  `all-MiniLM-L6-v2` embeddings.
- **Provides a Streamlit UI** where users can ask questions about schemas and
  browse the enriched catalog.

## Architecture

```text
ClinicalTrials.gov API
          |
          v
  ingestion/extract_clinical_trials.py
          |
          v
       AWS S3
    (bronze JSON batches)
          |
          v
  transformations/pipeline.py
          |
          +--> data/tables/silver_studies.parquet
          +--> data/tables/silver_conditions.parquet
          +--> data/tables/silver_interventions.parquet
          +--> data/tables/gold_trials_summary.parquet
          |
          v
  catalog/generate_catalog.py
          |
          v
  data/enriched_catalog.json
          |
          v
       ChromaDB
          |
          v
      Streamlit app
```

## Requirements

- Python 3.10 or newer
- AWS account and an S3 bucket for bronze data
- Google Gemini API key
- Network access to ClinicalTrials.gov, AWS S3, and the Gemini API
- The dependencies listed in [`requirements.txt`](requirements.txt)

The project also includes a local `venv/` directory in some working
environments, but virtual environments are ignored by Git and should be created
locally rather than committed.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/niksam20/CatalogIQ.git
cd CatalogIQ

python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root. It is intentionally ignored by Git.

```dotenv
GEMINI_API_KEY=your_gemini_api_key

AWS_S3_BUCKET=your-catalogiq-bucket
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-east-1
```

The ingestion and transformation scripts use the AWS credentials above to
access S3. In production, prefer the standard AWS credential provider chain
(for example an IAM role or AWS profile) rather than putting long-lived
credentials in `.env`.

The Gemini API is used in two places:

1. `catalog/generate_catalog.py` creates the enriched catalog descriptions.
2. `app/streamlit_app.py` answers natural-language catalog questions.

Gemini requests may incur usage charges and are subject to API rate limits. Both
scripts use retries with exponential backoff.

## Running the data pipeline

Run the stages in this order from the repository root.

### 1. Ingest clinical trials into S3

The default entry point searches ClinicalTrials.gov for oncology studies and
targets 300 records:

```bash
python ingestion/extract_clinical_trials.py
```

To use the ingestion function with another condition or record limit:

```bash
python -c "from ingestion.extract_clinical_trials import ingest_trials; ingest_trials(condition='Diabetes', max_records=300)"
```

Each paginated API response is uploaded as a separate JSON object. The
`max_records` value is a target; the final total can be slightly above it
because the API returns pages of studies.

### 2. Download bronze data and build Parquet tables

This step downloads JSON objects from the S3 `bronze/studies/` prefix into
`data/bronze/`, then uses DuckDB to create the silver and gold tables:

```bash
python transformations/pipeline.py
```

The transformation step performs the following quality checks:

- filters out studies without an NCT identifier;
- removes duplicate records with `SELECT DISTINCT`;
- verifies that the number of rows in `silver_studies` equals the number of
  distinct `nct_id` values;
- stops with an error if duplicate primary keys are detected.

### 3. Generate the AI-enriched catalog

After the Parquet files exist, generate table and column documentation:

```bash
python catalog/generate_catalog.py
```

This writes `data/enriched_catalog.json`. The script reads the schema and row
count from each of the four Parquet files and sends the schema to Gemini for
plain-English descriptions.

## Running the Streamlit explorer

The app reads `data/enriched_catalog.json`, creates or updates a persistent
ChromaDB collection in `chroma_db/`, and uses Gemini to answer questions using
the most relevant catalog documents.

Start it from the repository root:

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown by Streamlit, usually
<http://localhost:8501>.

Example questions:

- `Which table contains trial phases?`
- `What is the primary key of the studies table?`
- `Which columns describe interventions?`
- `How many rows are in the gold trials summary?`

The app also includes a **Browse Enriched Data Catalog** section with expandable
details for each documented table.

### Running with an alternate port

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## Data model

| Layer | Table/file | Description |
| --- | --- | --- |
| Bronze | `data/bronze/*.json` | Raw, paginated ClinicalTrials.gov API responses |
| Silver | `silver_studies.parquet` | One row per study, including NCT ID, title, status, phase, sponsor, and start date |
| Silver | `silver_conditions.parquet` | Study-to-condition relationships |
| Silver | `silver_interventions.parquet` | Study interventions and intervention types |
| Gold | `gold_trials_summary.parquet` | Trial counts and distinct sponsor counts grouped by condition, phase, and status |
| Catalog | `data/enriched_catalog.json` | Gemini-generated table and column documentation |
| Vector index | `chroma_db/` | Persistent embeddings used for semantic catalog retrieval |

## Project layout

```text
.
├── app/
│   └── streamlit_app.py              # Streamlit catalog explorer
├── catalog/
│   └── generate_catalog.py           # Gemini metadata generation
├── data/
│   ├── bronze/                       # Downloaded raw JSON batches
│   ├── tables/                       # Silver and gold Parquet outputs
│   └── enriched_catalog.json         # Generated catalog metadata
├── ingestion/
│   └── extract_clinical_trials.py    # ClinicalTrials.gov → S3 ingestion
├── transformations/
│   └── pipeline.py                   # S3 bronze → DuckDB/Parquet
├── tests/
│   └── test_pipeline.py              # Data artifact and quality tests
├── chroma_db/                        # Persistent local vector store
├── requirements.txt
└── README.md
```

## Running tests

With the environment activated:

```bash
pytest
```

The tests expect the generated Parquet tables and enriched catalog to be
present. If starting from a clean checkout, run the pipeline stages first, or
provide equivalent generated artifacts under `data/tables/` and
`data/enriched_catalog.json`.

## Generated files and reproducibility

The following are local or generated artifacts and are ignored by Git:

- `.env`
- `data/*.json`
- `chroma_db/`
- Python virtual environments and caches

The bronze JSON files under `data/bronze/` and Parquet files under
`data/tables/` may be present in a working copy as sample outputs, but they are
not required source files. Recreate them by running ingestion and
transformation as described above.

To rebuild the local semantic index after changing catalog metadata, remove the
local `chroma_db/` directory and start the Streamlit app again. ChromaDB will
recreate its collection from `data/enriched_catalog.json`.

## Troubleshooting

### `GEMINI_API_KEY environment variable is missing`

Create `.env` in the repository root and add a valid `GEMINI_API_KEY`. Make
sure the command is run from the project root so `python-dotenv` can load it.

### S3 access or bucket errors

Check `AWS_S3_BUCKET`, the AWS credentials, and `AWS_DEFAULT_REGION`. The IAM
identity needs permission to list objects under `bronze/studies/`, upload
objects during ingestion, and download objects during transformation.

### The Streamlit app shows no catalog

Run the catalog-generation step and confirm that
`data/enriched_catalog.json` exists. The app cannot build the catalog view
without that file.

### Embedding model download is slow

The first app launch downloads the `all-MiniLM-L6-v2` sentence-transformer
model. Later launches reuse the local model cache and the persistent ChromaDB
index.

## Data source and limitations

Clinical-trial records are sourced from the public
[ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api). The catalog
descriptions are AI-generated and should be reviewed before being treated as
authoritative data documentation. This project is an engineering/data-catalog
demonstration, not a medical decision-support system.