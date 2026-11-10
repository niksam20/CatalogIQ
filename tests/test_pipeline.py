import os
import json
import duckdb
import pytest


EXPECTED_TABLES = [
    "data/tables/silver_studies.parquet",
    "data/tables/silver_conditions.parquet",
    "data/tables/silver_interventions.parquet",
    "data/tables/gold_trials_summary.parquet"
]


def test_parquet_tables_exist():
    for table_path in EXPECTED_TABLES:
        assert os.path.exists(table_path), f"Missing expected table: {table_path}"


def test_all_parquet_tables_have_data():
    con = duckdb.connect()
    try:
        for table_path in EXPECTED_TABLES:
            count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{table_path}')").fetchone()[0]
            assert count > 0, f"Table {table_path} is unexpectedly empty."
    finally:
        con.close()


def test_silver_studies_schema_and_uniqueness():
    con = duckdb.connect()
    try:
        columns = [
            col[0]
            for col in con.execute("DESCRIBE SELECT * FROM read_parquet('data/tables/silver_studies.parquet')").fetchall()
        ]
        assert "nct_id" in columns
        assert "title" in columns
        assert "overall_status" in columns
        assert "phase" in columns
        assert "lead_sponsor" in columns
        assert "start_date" in columns

        total_count = con.execute("SELECT COUNT(*) FROM read_parquet('data/tables/silver_studies.parquet')").fetchone()[0]
        distinct_count = con.execute("SELECT COUNT(DISTINCT nct_id) FROM read_parquet('data/tables/silver_studies.parquet')").fetchone()[0]
        null_ids = con.execute("SELECT COUNT(*) FROM read_parquet('data/tables/silver_studies.parquet') WHERE nct_id IS NULL").fetchone()[0]

        assert total_count == distinct_count, "Found duplicate nct_id in silver_studies"
        assert null_ids == 0, "Found NULL nct_id in silver_studies"
    finally:
        con.close()


def test_referential_integrity():
    con = duckdb.connect()
    try:
        orphan_conditions = con.execute("""
            SELECT COUNT(*)
            FROM read_parquet('data/tables/silver_conditions.parquet') c
            LEFT JOIN read_parquet('data/tables/silver_studies.parquet') s ON c.nct_id = s.nct_id
            WHERE s.nct_id IS NULL
        """).fetchone()[0]
        assert orphan_conditions == 0, f"Found {orphan_conditions} orphan records in silver_conditions."

        orphan_interventions = con.execute("""
            SELECT COUNT(*)
            FROM read_parquet('data/tables/silver_interventions.parquet') i
            LEFT JOIN read_parquet('data/tables/silver_studies.parquet') s ON i.nct_id = s.nct_id
            WHERE s.nct_id IS NULL
        """).fetchone()[0]
        assert orphan_interventions == 0, f"Found {orphan_interventions} orphan records in silver_interventions."
    finally:
        con.close()


def test_gold_trials_summary_aggregation():
    con = duckdb.connect()
    try:
        invalid_rows = con.execute("""
            SELECT COUNT(*)
            FROM read_parquet('data/tables/gold_trials_summary.parquet')
            WHERE condition_name IS NULL
               OR phase IS NULL
               OR trial_count <= 0
               OR distinct_sponsors <= 0
        """).fetchone()[0]
        assert invalid_rows == 0, "gold_trials_summary contains rows with null keys or non-positive counts."
    finally:
        con.close()


def test_enriched_catalog_structure_and_counts():
    catalog_path = "data/enriched_catalog.json"
    assert os.path.exists(catalog_path), "Enriched catalog JSON was not generated"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert isinstance(catalog, list)
    assert len(catalog) == 4, f"Expected 4 tables in catalog, got {len(catalog)}"

    con = duckdb.connect()
    try:
        for table in catalog:
            table_name = table["table_name"]
            assert "table_description" in table and len(table["table_description"]) > 0
            assert "columns" in table and len(table["columns"]) > 0

            # Verify recorded row count matches actual parquet file
            actual_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('data/tables/{table_name}.parquet')").fetchone()[0]
            assert table["row_count"] == actual_count, (
                f"Row count mismatch for {table_name}: catalog has {table['row_count']}, parquet has {actual_count}"
            )

            for col in table["columns"]:
                assert "name" in col
                assert "type" in col
                assert "description" in col and len(col["description"]) > 0
    finally:
        con.close()


def test_clean_module_imports():
    # Ensure modules can be imported without throwing top-level exceptions
    import ingestion.extract_clinical_trials
    import transformations.pipeline
    import catalog.generate_catalog
    assert hasattr(transformations.pipeline, "run_pipeline")
    assert hasattr(catalog.generate_catalog, "extract_and_enrich")
    assert hasattr(ingestion.extract_clinical_trials, "ingest_trials")
