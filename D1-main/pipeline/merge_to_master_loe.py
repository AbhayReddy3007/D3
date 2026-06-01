#!/usr/bin/env python3
"""
merge_to_master_loe.py
──────────────────────
Merges loe_table + forecasted_loe → Master_LOE in BigQuery.

- IP_Dimension_1_Score     → LOWEST  per Drug_Name
- Avg_Years_to_Entry_US__EP → HIGHEST per Drug_Name
- Appends (never truncates)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not needed on Cloud Run

from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID  = os.getenv("BQ_UPLOAD_PROJECT",  os.getenv("PROJECT_ID",  "cognito-prod-394707"))
DATASET_ID  = os.getenv("BQ_UPLOAD_DATASET",  os.getenv("BQ_DATASET_ID", "cognito_prod_datamart"))
BQ_LOCATION = os.getenv("BQ_UPLOAD_LOCATION", "asia-south1")

LOE_TABLE        = f"{PROJECT_ID}.{DATASET_ID}.loe_table"
FORECASTED_TABLE = f"{PROJECT_ID}.{DATASET_ID}.forecasted_loe"
MASTER_TABLE     = f"{PROJECT_ID}.{DATASET_ID}.Master_LOE"

MASTER_COLUMNS = [
    "Drug_Name", "Patent_Number", "Jurisdiction", "Tag", "Blocking_Category",
    "Reason", "Step_1_Claim_Category", "Step_2_Matched_Elements",
    "S2_Active_Ingredient__Form", "S2_Formulation_Details",
    "S2_Route_of_Administration", "S2_Device_Description",
    "S2_Combination_TechProcess", "Step_3_Technical_Barrier",
    "Step_3_Confidence", "Step_3_Evidence_Type", "Step_3_Evidence_Summary",
    "Step_4_Blocking_Indicator", "Step_4_Confidence",
    "Step_4_Regulatory_Failure_if_Removed", "Step_4_Bridging_Studies_Required",
    "Step_4_Formulation_Consistent_Across_Phases", "Step_4_Reason",
    "Step_5_Novel__Difficult", "Step_5_Novelty_Signal", "Step_5_FirstinClass",
    "Step_5_Prior_Failed_Attempts", "Step_5_Complex_Implementation",
    "Step_5_Confidence", "Step_5_Reason", "Filing_Date", "Grant_Date",
    "PTE_months", "Pediatric_Exclusivity", "Phase", "Launch_Date",
    "Approval_Date", "Approval_Date_Source", "Est_Approval_Year",
    "Exclusivity_Year", "Controlling_Patent_Expiry_Year", "Years_to_Entry",
    "Avg_Years_to_Entry", "Score", "Avg_Years_to_Entry_US__EP",
    "IP_Dimension_1_Score", "Source_File", "Type", "No_Of_Forecasted_Patents",
    "Rationale", "Report_Timestamp", "created_at", "updated_at",
]

_LOE_COL_MAP = {
    "Drug_Name": "Drug_Name", "Patent_Number": "Patent_Number",
    "Jurisdiction": "Jurisdiction", "Tag": "Tag",
    "Blocking_Category": "Blocking_Category", "Reason": "Reason",
    "Step_1_Claim_Category": "Step_1_Claim_Category",
    "Step_2_Matched_Elements": "Step_2_Matched_Elements",
    "S2_Active_Ingredient_Form": "S2_Active_Ingredient__Form",
    "S2_Active_Ingredient__Form": "S2_Active_Ingredient__Form",
    "S2_Formulation_Details": "S2_Formulation_Details",
    "S2_Route_of_Administration": "S2_Route_of_Administration",
    "S2_Device_Description": "S2_Device_Description",
    "S2_Combination_TechProcess": "S2_Combination_TechProcess",
    "S2_Combination_Tech_Process": "S2_Combination_TechProcess",
    "Step_3_Technical_Barrier": "Step_3_Technical_Barrier",
    "Step_3_Confidence": "Step_3_Confidence",
    "Step_3_Evidence_Type": "Step_3_Evidence_Type",
    "Step_3_Evidence_Summary": "Step_3_Evidence_Summary",
    "Step_4_Blocking_Indicator": "Step_4_Blocking_Indicator",
    "Step_4_Confidence": "Step_4_Confidence",
    "Step_4_Regulatory_Failure_if_Removed": "Step_4_Regulatory_Failure_if_Removed",
    "Step_4_Bridging_Studies_Required": "Step_4_Bridging_Studies_Required",
    "Step_4_Formulation_Consistent_Across_Phases": "Step_4_Formulation_Consistent_Across_Phases",
    "Step_4_Reason": "Step_4_Reason",
    "Step_5_Novel_Difficult": "Step_5_Novel__Difficult",
    "Step_5_Novel__Difficult": "Step_5_Novel__Difficult",
    "Step_5_Novelty_Signal": "Step_5_Novelty_Signal",
    "Step_5_FirstinClass": "Step_5_FirstinClass",
    "Step_5_First_in_Class": "Step_5_FirstinClass",
    "Step_5_Prior_Failed_Attempts": "Step_5_Prior_Failed_Attempts",
    "Step_5_Complex_Implementation": "Step_5_Complex_Implementation",
    "Step_5_Confidence": "Step_5_Confidence",
    "Step_5_Reason": "Step_5_Reason",
    "Filing_Date": "Filing_Date", "Grant_Date": "Grant_Date",
    "PTE_months": "PTE_months", "PTE_months_": "PTE_months",
    "Pediatric_Exclusivity": "Pediatric_Exclusivity", "Phase": "Phase",
    "Launch_Date": "Launch_Date", "Approval_Date": "Approval_Date",
    "Approval_Date_Source": "Approval_Date_Source",
    "Est_Approval_Year": "Est_Approval_Year",
    "Exclusivity_Year": "Exclusivity_Year",
    "Controlling_Patent_Expiry_Year": "Controlling_Patent_Expiry_Year",
    "Years_to_Entry": "Years_to_Entry",
    "Avg_Years_to_Entry": "Avg_Years_to_Entry", "Score": "Score",
    "Avg_Years_to_Entry_US__EP": "Avg_Years_to_Entry_US__EP",
    "Avg_Years_to_Entry_US_EP_": "Avg_Years_to_Entry_US__EP",
    "IP_Dimension_1_Score": "IP_Dimension_1_Score",
    "Source_File": "Source_File",
}

_FORECAST_COL_MAP = {
    "drug_name": "Drug_Name", "patent_number": "Patent_Number",
    "jurisdiction": "Jurisdiction", "tag": "Tag",
    "blocking_category": "Blocking_Category", "reason": "Reason",
    "step1_claim_category": "Step_1_Claim_Category",
    "step_1_claim_category": "Step_1_Claim_Category",
    "filing_date": "Filing_Date", "filing_date_lower": "Filing_Date",
    "grant_date": "Grant_Date", "pte_months": "PTE_months",
    "pediatric_exclusivity": "Pediatric_Exclusivity",
    "phase": "Phase", "phase_in_jurisdiction": "Phase",
    "launch_date": "Launch_Date", "approval_date": "Approval_Date",
    "approval_date_source": "Approval_Date_Source",
    "est_approval_year": "Est_Approval_Year",
    "exclusivity_year": "Exclusivity_Year",
    "controlling_patent_expiry_year": "Controlling_Patent_Expiry_Year",
    "years_to_entry": "Years_to_Entry",
    "avg_years_to_entry": "Avg_Years_to_Entry", "score": "Score",
    "avg_years_to_entry_us_ep": "Avg_Years_to_Entry_US__EP",
    "ip_dimension_1_score": "IP_Dimension_1_Score",
    "source_file": "Source_File", "type": "Type",
    "no_of_forecasted_patents": "No_Of_Forecasted_Patents",
    "rationale": "Rationale", "scored_at": "Report_Timestamp",
}


def _get_bq_client():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        creds = service_account.Credentials.from_service_account_file(creds_path)
        return bigquery.Client(project=PROJECT_ID, credentials=creds, location=BQ_LOCATION)
    return bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)


def _read_table(client, table_id):
    try:
        df = client.query(f"SELECT * FROM `{table_id}`").to_dataframe()
        print(f"  [{table_id.split('.')[-1]}] {len(df)} rows, {len(df.columns)} cols")
        return df
    except Exception as e:
        print(f"  [{table_id.split('.')[-1]}] Failed: {e}")
        return pd.DataFrame()


def _map_and_align(df, col_map):
    rename = {c: col_map[c] for c in df.columns if c in col_map}
    df = df.rename(columns=rename)
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[[c for c in MASTER_COLUMNS if c in df.columns]]


def _compute_drug_aggregates(df):
    if df.empty:
        return df
    df = df.copy()
    df["_score"] = pd.to_numeric(df["IP_Dimension_1_Score"], errors="coerce")
    df["_yte"]   = pd.to_numeric(df["Avg_Years_to_Entry_US__EP"], errors="coerce")
    df["IP_Dimension_1_Score"] = (
        df.groupby("Drug_Name")["_score"].transform("min")
        .apply(lambda x: str(x) if pd.notna(x) else None)
    )
    df["Avg_Years_to_Entry_US__EP"] = (
        df.groupby("Drug_Name")["_yte"].transform("max")
        .apply(lambda x: str(x) if pd.notna(x) else None)
    )
    return df.drop(columns=["_score", "_yte"])


def _deduplicate_in_bq(client):
    """Replace Master_LOE with a fully deduplicated copy, keeping the latest updated_at per unique row."""
    dedup_sql = f"""
        CREATE OR REPLACE TABLE `{MASTER_TABLE}` AS
        SELECT * EXCEPT(rn)
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY {", ".join(
                        f"`{c}`" for c in MASTER_COLUMNS
                        if c not in ("created_at", "updated_at", "Report_Timestamp")
                    )}
                    ORDER BY updated_at DESC
                ) AS rn
            FROM `{MASTER_TABLE}`
        )
        WHERE rn = 1
    """
    print("  Deduplicating in BigQuery...")
    client.query(dedup_sql).result()


def merge_and_upload(dry_run=False):
    client = _get_bq_client()
    now = datetime.now(timezone.utc)

    print("\n[1/4] Reading source tables...")
    loe_df      = _read_table(client, LOE_TABLE)
    forecast_df = _read_table(client, FORECASTED_TABLE)
    if loe_df.empty and forecast_df.empty:
        print("[ERROR] Both tables empty.")
        sys.exit(1)

    print("\n[2/4] Mapping columns...")
    if not loe_df.empty:
        loe_df = _map_and_align(loe_df, _LOE_COL_MAP)
        loe_df["Type"] = "Existing"
    if not forecast_df.empty:
        forecast_df = _map_and_align(forecast_df, _FORECAST_COL_MAP)
        if "Type" not in forecast_df.columns or forecast_df["Type"].isna().all():
            forecast_df["Type"] = "Forecasted"

    print("\n[3/4] Merging...")
    master = pd.concat([loe_df, forecast_df], ignore_index=True)

    ts_cols = {"Report_Timestamp", "created_at", "updated_at"}
    for col in master.columns:
        if col not in ts_cols:
            master[col] = master[col].astype(str).replace({"nan": None, "None": None, "": None})

    master["Report_Timestamp"] = pd.to_datetime(master["Report_Timestamp"], errors="coerce")
    master["created_at"] = pd.to_datetime(master.get("created_at"), errors="coerce").fillna(now)
    master["updated_at"] = now
    master = _compute_drug_aggregates(master)

    existing   = len(master[master["Type"] == "Existing"])
    forecasted = len(master[master["Type"] == "Forecasted"])
    print(f"  Total: {len(master)} | Existing: {existing} | Forecasted: {forecasted}")

    if dry_run:
        print("\n[DRY RUN]", MASTER_TABLE)
        print(master.head(3).to_string())
        return

    print(f"\n[4/5] Appending to {MASTER_TABLE}...")
    job = client.load_table_from_dataframe(
        master, MASTER_TABLE,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        ),
    )
    job.result()
    print(f"  Appended {len(master)} rows.")

    print(f"\n[5/5] Deduplicating {MASTER_TABLE}...")
    _deduplicate_in_bq(client)
    t = client.get_table(MASTER_TABLE)
    print(f"  [DONE] {t.num_rows} rows remaining after dedup.")


def main():
    parser = argparse.ArgumentParser(description="Merge → Master_LOE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    print("=" * 60)
    print(f"  MERGE → Master_LOE  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    merge_and_upload(dry_run=args.dry_run)
    print(f"\n[DONE] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
