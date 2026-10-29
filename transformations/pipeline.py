import os
import boto3
import duckdb
from dotenv import load_dotenv

load_dotenv()
S3_BUCKET = os.getenv("AWS_S3_BUCKET")

def download_s3_bronze_locally():
    os.makedirs("data/bronze", exist_ok=True)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="bronze/studies/")
    downloaded = 0
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".json"):
            local_filename = os.path.join("data/bronze", os.path.basename(key))
            s3.download_file(S3_BUCKET, key, local_filename)
            downloaded += 1
            
    print(f"[INFO] Downloaded {downloaded} raw JSON files from S3.")

def run_pipeline():
    download_s3_bronze_locally()
    os.makedirs("data/tables", exist_ok=True)
    
    con = duckdb.connect()

    # 1. Base unnested studies view to simplify downstream parsing
    con.execute("""
        CREATE OR REPLACE VIEW raw_exploded AS
        SELECT unnest(studies) AS study
        FROM read_json_auto('data/bronze/*.json');
    """)

    # 2. SILVER: Cleaned Studies Dimension
    con.execute("""
        CREATE OR REPLACE TABLE silver_studies AS
        SELECT DISTINCT
            study.protocolSection.identificationModule.nctId AS nct_id,
            study.protocolSection.identificationModule.briefTitle AS title,
            study.protocolSection.statusModule.overallStatus AS overall_status,
            study.protocolSection.designModule.phases[1] AS phase,
            study.protocolSection.sponsorCollaboratorsModule.leadSponsor.name AS lead_sponsor,
            study.protocolSection.statusModule.startDateStruct.date AS start_date
        FROM raw_exploded
        WHERE study.protocolSection.identificationModule.nctId IS NOT NULL;
    """)

    # 3. SILVER: Conditions Fact Table
    con.execute("""
        CREATE OR REPLACE TABLE silver_conditions AS
        SELECT DISTINCT
            study.protocolSection.identificationModule.nctId AS nct_id,
            unnest(study.protocolSection.conditionsModule.conditions) AS condition_name
        FROM raw_exploded
        WHERE study.protocolSection.identificationModule.nctId IS NOT NULL
          AND study.protocolSection.conditionsModule.conditions IS NOT NULL;
    """)

    # 4. SILVER: Interventions Fact Table
    con.execute("""
        CREATE OR REPLACE TABLE silver_interventions AS
        WITH unnested_interventions AS (
            SELECT
                study.protocolSection.identificationModule.nctId AS nct_id,
                unnest(study.protocolSection.armsInterventionsModule.interventions) AS item
            FROM raw_exploded
            WHERE study.protocolSection.identificationModule.nctId IS NOT NULL
              AND study.protocolSection.armsInterventionsModule.interventions IS NOT NULL
        )
        SELECT DISTINCT
            nct_id,
            item.type AS intervention_type,
            item.name AS intervention_name
        FROM unnested_interventions
        WHERE item.name IS NOT NULL;
    """)

    # 5. DATA QUALITY CHECKS
    total_studies = con.execute("SELECT COUNT(*) FROM silver_studies").fetchone()[0]
    distinct_ids = con.execute("SELECT COUNT(DISTINCT nct_id) FROM silver_studies").fetchone()[0]
    null_status = con.execute("SELECT COUNT(*) FROM silver_studies WHERE overall_status IS NULL").fetchone()[0]

    print("--- [DATA QUALITY REPORT] ---")
    print(f"Total Studies: {total_studies} | Unique IDs: {distinct_ids} | Missing Statuses: {null_status}")
    if total_studies != distinct_ids:
        raise ValueError("Data Quality Alert: Duplicate nct_id detected in silver_studies!")
    print("Quality checks passed: Primary key uniqueness confirmed.")

    # 6. GOLD: Analytical Mart
    con.execute("""
        CREATE OR REPLACE TABLE gold_trials_summary AS
        SELECT 
            c.condition_name,
            COALESCE(s.phase, 'NOT_SPECIFIED') AS phase,
            s.overall_status,
            COUNT(s.nct_id) AS trial_count,
            COUNT(DISTINCT s.lead_sponsor) AS distinct_sponsors
        FROM silver_studies s
        JOIN silver_conditions c ON s.nct_id = c.nct_id
        GROUP BY 1, 2, 3;
    """)

    # 7. Save as Parquet Files
    for tbl in ["silver_studies", "silver_conditions", "silver_interventions", "gold_trials_summary"]:
        target_path = f"data/tables/{tbl}.parquet"
        con.execute(f"COPY {tbl} TO '{target_path}' (FORMAT PARQUET)")
        row_count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"[SUCCESS] Table '{tbl}' exported to {target_path} ({row_count} rows).")

if __name__ == "__main__":
    run_pipeline()