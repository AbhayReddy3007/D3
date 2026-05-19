"""
batch_innovator_patterns.py
───────────────────────────
Parallel batch processing + BigQuery storage
"""

import os
import sys
import csv
import asyncio
import argparse
import random
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv(override=True)

# ✅ CONFIG
MAX_CONCURRENT_DRUGS = 3
TIMEOUT_PER_DRUG = 300
MAX_RETRIES = 3
LOG_FILE = "batch_errors.log"

# ✅ BIGQUERY CONFIG
BQ_LOCATION = "asia-south1"
PROJECT_ID = "cognito-prod-394707"
DATASET_ID = "cognito_prod_datamart"
TABLE_ID = "filing_pattern_table"
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

OUTPUT_DIR = Path("innovator_patterns")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def log_error(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────
# IMPORT LOCAL MODULES
# ─────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_parent = _here.parent
_pkg = _here.name

for _p in [str(_here), str(_parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib

# ✅ GCS
_gcs = importlib.import_module(f"{_pkg}.gcs_lister")
get_gcs_client = _gcs.get_gcs_client
GCS_BUCKET_NAME = _gcs.GCS_BUCKET_NAME
GCS_PATENTS_PREFIX = _gcs.GCS_PATENTS_PREFIX

# ✅ ANALYSIS
_ifp = importlib.import_module(f"{_pkg}.test_innovator_filing_patterns")
analyze_innovator_filing_patterns = _ifp.analyze_innovator_filing_patterns

# ─────────────────────────────────────────────
# ✅ BIGQUERY SETUP
# ─────────────────────────────────────────────
def get_bq_client():
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        return bigquery.Client(credentials=creds, project=PROJECT_ID)
    return bigquery.Client(project=PROJECT_ID)


def create_dataset_if_not_exists(client):
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = BQ_LOCATION

    try:
        client.get_dataset(dataset_ref)
        print("[BQ] Dataset exists")
    except Exception:
        client.create_dataset(dataset)
        print("[BQ] Dataset created")


def create_table_if_not_exists(client):
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    schema = [
        bigquery.SchemaField("drug", "STRING"),
        bigquery.SchemaField("company", "STRING"),
        bigquery.SchemaField("characterization", "STRING"),
        bigquery.SchemaField("confidence", "STRING"),
        bigquery.SchemaField("rationale", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]

    table = bigquery.Table(table_ref, schema=schema)

    try:
        client.get_table(table_ref)
        print("[BQ] Table exists")
    except Exception:
        client.create_table(table)
        print("[BQ] Table created")


def insert_into_bq(client, drug_name, result):
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    rows = []
    for inv in result.get("innovators", []):
        rows.append({
            "drug": drug_name,
            "company": inv.get("company"),
            "characterization": inv.get("characterization"),
            "confidence": str(inv.get("confidence")),
            "rationale": inv.get("rationale"),
            "created_at": datetime.utcnow().isoformat()
        })

    if rows:
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            print(f"[BQ ERROR] {errors}")
        else:
            print(f"[BQ] Inserted {len(rows)} rows for {drug_name}")

# ─────────────────────────────────────────────
# GCS LIST
# ─────────────────────────────────────────────
def list_all_gcs_drugs():
    client = get_gcs_client()
    prefix = GCS_PATENTS_PREFIX.rstrip("/") + "/"

    blobs = list(client.list_blobs(GCS_BUCKET_NAME, prefix=prefix))

    drugs = sorted({b.name.split("/")[1] for b in blobs if "/" in b.name})
    print(f"[GCS] Found {len(drugs)} drugs")
    return drugs

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
def save_drug_csv(drug, result):
    file = OUTPUT_DIR / f"{drug}.csv"
    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Drug", "Company", "Characterization", "Confidence"])

        for inv in result.get("innovators", []):
            writer.writerow([
                drug,
                inv.get("company"),
                inv.get("characterization"),
                inv.get("confidence"),
            ])
    return file

# ─────────────────────────────────────────────
# PROCESS DRUG
# ─────────────────────────────────────────────
async def process_drug(i, total, drug, semaphore, bq_client, results, failed):

    print(f"\n[{i}/{total}] {drug}")

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                result = await asyncio.wait_for(
                    analyze_innovator_filing_patterns(drug),
                    timeout=TIMEOUT_PER_DRUG
                )

            if result:
                results[drug] = result

                save_drug_csv(drug, result)
                insert_into_bq(bq_client, drug, result)

                print(f"✅ Done: {drug}")
                return

            raise Exception("Empty result")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + random.random()
                print(f"[Retry] {drug} in {wait:.2f}s")
                await asyncio.sleep(wait)
            else:
                print(f"[FAILED] {drug}: {e}")
                log_error(f"{drug}: {e}")
                failed.append((drug, str(e)))

# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────
async def run_all(drugs):

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ✅ INIT BIGQUERY
    bq_client = get_bq_client()
    create_dataset_if_not_exists(bq_client)
    create_table_if_not_exists(bq_client)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DRUGS)

    results = {}
    failed = []

    tasks = [
        process_drug(i + 1, len(drugs), drug, semaphore, bq_client, results, failed)
        for i, drug in enumerate(drugs)
    ]

    await asyncio.gather(*tasks)

    print("\n======== SUMMARY ========")
    print(f"Success: {len(results)}")
    print(f"Failed : {len(failed)}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--drug")

    args = parser.parse_args()

    if args.drug:
        drugs = [args.drug]
    else:
        drugs = list_all_gcs_drugs()

    if args.limit:
        drugs = drugs[:args.limit]

    start = datetime.now()

    asyncio.run(run_all(drugs))

    print(f"\nTime taken: {(datetime.now() - start).total_seconds():.2f}s")
