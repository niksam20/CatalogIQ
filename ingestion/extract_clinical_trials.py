import os
import json
import logging
from datetime import datetime, timezone
import requests
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("IngestionService")

API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
S3_BUCKET = os.getenv("AWS_S3_BUCKET")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.HTTPError)),
    reraise=True
)
def fetch_api_page(condition: str, page_token: str = None) -> dict:
    params = {
        "format": "json",
        "query.cond": condition,
        "pageSize": 100
    }
    if page_token:
        params["pageToken"] = page_token

    response = requests.get(API_BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def upload_to_s3_bronze(payload: dict, batch_index: int) -> str:
    now = datetime.now(timezone.utc)
    partition = f"bronze/studies/year={now.year}/month={now.strftime('%m')}"
    filename = f"trials_batch_{now.strftime('%Y%m%d_%H%M%S')}_{batch_index}.json"
    s3_key = f"{partition}/{filename}"

    body = json.dumps(payload, ensure_ascii=False)
    
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=body.encode("utf-8"),
        ContentType="application/json"
    )
    logger.info(f"Wrote batch {batch_index} to s3://{S3_BUCKET}/{s3_key}")
    return s3_key

def ingest_trials(condition: str = "Oncology", max_records: int = 300):
    logger.info(f"Starting ingestion for condition '{condition}', target={max_records} records.")
    total_fetched = 0
    page_token = None
    batch_idx = 1

    while total_fetched < max_records:
        data = fetch_api_page(condition=condition, page_token=page_token)
        studies = data.get("studies", [])
        
        if not studies:
            logger.info("No further studies returned by endpoint.")
            break

        upload_to_s3_bronze(data, batch_idx)
        
        total_fetched += len(studies)
        logger.info(f"Ingested {total_fetched}/{max_records} cumulative studies.")
        batch_idx += 1

        page_token = data.get("nextPageToken")
        if not page_token:
            logger.info("Reached end of result pagination.")
            break

    logger.info(f"Ingestion complete. Total ingested: {total_fetched}")

if __name__ == "__main__":
    ingest_trials(condition="Oncology", max_records=300)