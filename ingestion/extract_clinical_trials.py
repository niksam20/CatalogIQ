import os
import json
import logging
from datetime import datetime
import requests
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://clinicaltrials.gov/api/v2/studies"
S3_BUCKET = os.getenv("AWS_S3_BUCKET")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.HTTPError))
)
def fetch_studies_page(condition: str, page_token: str = None) -> dict:
    params = {"format": "json", "query.cond": condition, "pageSize": 100}
    if page_token:
        params["pageToken"] = page_token

    res = requests.get(API_URL, params=params, timeout=20)
    res.raise_for_status()
    return res.json()

def upload_raw_to_s3(payload: dict, batch_id: int):
    now = datetime.utcnow()
    partition_path = f"bronze/studies/year={now.year}/month={now.strftime('%m')}"
    file_name = f"trials_batch_{now.strftime('%Y%m%d_%H%M%S')}_{batch_id}.json"
    s3_key = f"{partition_path}/{file_name}"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(payload),
        ContentType="application/json"
    )
    logger.info(f"Successfully uploaded: s3://{S3_BUCKET}/{s3_key}")

def execute_ingestion(target_records: int = 300, condition: str = "Oncology"):
    records_ingested = 0
    token = None
    batch = 1

    while records_ingested < target_records:
        data = fetch_studies_page(condition=condition, page_token=token)
        studies = data.get("studies", [])
        if not studies:
            break

        upload_raw_to_s3(data, batch)
        records_ingested += len(studies)
        batch += 1
        token = data.get("nextPageToken")
        if not token:
            break

    logger.info(f"Ingestion complete. Total ingested records: {records_ingested}")

if __name__ == "__main__":
    execute_ingestion(target_records=300, condition="Oncology")