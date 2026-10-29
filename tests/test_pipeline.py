import os
import pytest
import duckdb

def test_parquet_tables_exist():
    expected_tables = [
        "data/tables/silver_studies.parquet",
        "data/tables/silver_conditions.parquet",
        "data/tables/silver_interventions.parquet",
        "data/tables/gold_trials_summary.parquet"
    ]
    for table_path in expected_tables:
        assert os.path.exists(table_path), f"Missing expected table: {table_path}"

def test_silver_studies_schema_and_uniqueness():
    con = duckdb.connect()
    columns = [col[0] for col in con.execute("DESCRIBE SELECT * FROM read_parquet('data/tables/silver_studies.parquet')").fetchall()]
    assert "nct_id" in columns
    assert "overall_status" in columns

    total_count = con.execute("SELECT COUNT(*) FROM read_parquet('data/tables/silver_studies.parquet')").fetchone()[0]
    distinct_count = con.execute("SELECT COUNT(DISTINCT nct_id) FROM read_parquet('data/tables/silver_studies.parquet')").fetchone()[0]
    assert total_count == distinct_count, "Found duplicate nct_id in silver_studies"

def test_gold_summary_has_rows():
    con = duckdb.connect()
    count = con.execute("SELECT COUNT(*) FROM read_parquet('data/tables/gold_trials_summary.parquet')").fetchone()[0]
    assert count > 0, "gold_trials_summary table is empty"

def test_enriched_catalog_exists():
    assert os.path.exists("data/enriched_catalog.json"), "Enriched catalog JSON was not generated"