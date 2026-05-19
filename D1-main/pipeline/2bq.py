"""
upload_to_bq.py
────────────────
Standalone script to upload patent analysis results directly to BigQuery.

Reads patent data from the pipeline's results cache (results_cache/*.json),
builds the same row structure as the main pipeline, and uploads to BQ.
No Excel files involved.

Usage:
    # Upload a specific drug
    python upload_to_bq.py --drug Bgm0504

    # Upload ALL drugs in the results cache
    python upload_to_bq.py --all

    # Dry run — preview rows without uploading
    python upload_to_bq.py --drug Bgm0504 --dry-run

    # Override BQ destination
    python upload_to_bq.py --drug Bgm0504 --project my-project --dataset my_ds --table my_tbl
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ─────────────────────────────────────────────
# Default config (matches excel_exporter.py)
# ─────────────────────────────────────────────

RESULTS_CACHE_DIR = Path(
    os.getenv("RESULTS_CACHE_DIR", Path(__file__).parent / "results_cache")
)

DEFAULT_PROJECT  = os.getenv("BQ_UPLOAD_PROJECT",  "cognito-prod-394707")
DEFAULT_DATASET  = os.getenv("BQ_UPLOAD_DATASET",  "cognito_prod_datamart")
DEFAULT_TABLE    = os.getenv("BQ_UPLOAD_TABLE",    "loe_table")
DEFAULT_LOCATION = os.getenv("BQ_UPLOAD_LOCATION", "asia-south1")


# ─────────────────────────────────────────────
# Row builder (same logic as excel_exporter.py)
# ─────────────────────────────────────────────

def _format_approval_date(raw: Optional[str]) -> Optional[str]:
    """Normalises approval dates to DD-MMM-YYYY."""
    if not raw or str(raw).lower() in ("none", "null", "n/a", ""):
        return None
    raw = str(raw).strip()
    formats = [
        "%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d %B %Y", "%B %d, %Y", "%B %d %Y", "%d-%b-%Y", "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%d-%b-%Y")
        except ValueError:
            continue
    return raw


def build_rows(drug_name: str, patents: List[Dict]) -> List[Dict]:
    """
    Builds flat row dicts from patent dicts — identical to
    excel_exporter._build_rows() so BQ schema matches exactly.
    """
    rows = []
    for p in patents:
        jurisdiction = (p.get("jurisdiction") or "").upper()
        approval_date = _format_approval_date(
            p.get("approval_date_us") if jurisdiction == "US"
            else p.get("approval_date_eu") if jurisdiction in ("EP", "EU")
            else p.get(f"approval_date_{jurisdiction.lower()}")
        )
        approval_source = (
            p.get("approval_date_us_source") if jurisdiction == "US"
            else p.get("approval_date_eu_source") if jurisdiction in ("EP", "EU")
            else p.get(f"approval_date_{jurisdiction.lower()}_source")
        )
        phase = p.get("phase_at_filing")

        rows.append({
            "Drug Name":                      drug_name,
            "Patent Number":                  p.get("patent_number", ""),
            "Jurisdiction":                   p.get("jurisdiction", ""),
            "Tag":                            p.get("tag", ""),
            "Blocking Category":              p.get("blocking_category") or "N/A",
            "Reason":                         p.get("reason") or "N/A",
            "Step 1 Claim Category":          p.get("claim_category") or "N/A",
            "Step 2 Matched Elements":        (
                ", ".join(k for k, v in (p.get("step2_elements_present") or {}).items() if v)
                or ("N/A" if p.get("tag") == "BLOCKING" else "None matched")
            ),
            "S2: Active Ingredient & Form":   (
                str((p.get("step2_elements_present") or {}).get("active_ingredient_and_form", "N/A"))
                if p.get("step2_elements_present") is not None else "N/A"
            ),
            "S2: Formulation Details":        (
                str((p.get("step2_elements_present") or {}).get("formulation_details", "N/A"))
                if p.get("step2_elements_present") is not None else "N/A"
            ),
            "S2: Route of Administration":    (
                str((p.get("step2_elements_present") or {}).get("route_of_administration", "N/A"))
                if p.get("step2_elements_present") is not None else "N/A"
            ),
            "S2: Device Description":         (
                str((p.get("step2_elements_present") or {}).get("device_description", "N/A"))
                if p.get("step2_elements_present") is not None else "N/A"
            ),
            "S2: Combination Tech/Process":   (
                str((p.get("step2_elements_present") or {}).get("combination_tech_process", "N/A"))
                if p.get("step2_elements_present") is not None else "N/A"
            ),
            "Step 3 Technical Barrier":       (
                "Yes" if p.get("step3_is_technical_barrier") is True
                else "No" if p.get("step3_is_technical_barrier") is False
                else "N/A"
            ),
            "Step 3 Confidence":              p.get("step3_confidence") or "N/A",
            "Step 3 Evidence Type":           p.get("step3_evidence_type") or "N/A",
            "Step 3 Evidence Summary":        p.get("step3_evidence_summary") or "N/A",
            "Step 4 Blocking Indicator":      (
                "Yes" if p.get("step4_is_blocking_indicator") is True
                else "No" if p.get("step4_is_blocking_indicator") is False
                else "N/A"
            ),
            "Step 4 Confidence":              p.get("step4_confidence") or "N/A",
            "Step 4 Regulatory Failure if Removed": (
                "Yes" if p.get("step4_regulatory_failure_if_removed") is True
                else "No" if p.get("step4_regulatory_failure_if_removed") is False
                else "N/A"
            ),
            "Step 4 Bridging Studies Required": (
                "Yes" if p.get("step4_bridging_studies_required") is True
                else "No" if p.get("step4_bridging_studies_required") is False
                else "N/A"
            ),
            "Step 4 Formulation Consistent Across Phases": (
                "Yes" if p.get("step4_formulation_consistent_across_phases") is True
                else "No" if p.get("step4_formulation_consistent_across_phases") is False
                else "N/A"
            ),
            "Step 4 Reason":                  p.get("step4_reason") or "N/A",
            "Step 5 Novel & Difficult":       (
                "Yes" if p.get("step5_is_novel_and_difficult") is True
                else "No" if p.get("step5_is_novel_and_difficult") is False
                else "N/A"
            ),
            "Step 5 Novelty Signal":          p.get("step5_novelty_signal") or "N/A",
            "Step 5 First-in-Class":          (
                "Yes" if p.get("step5_first_in_class") is True
                else "No" if p.get("step5_first_in_class") is False
                else "N/A"
            ),
            "Step 5 Prior Failed Attempts":   (
                "Yes" if p.get("step5_prior_failed_attempts") is True
                else "No" if p.get("step5_prior_failed_attempts") is False
                else "N/A"
            ),
            "Step 5 Complex Implementation":  (
                "Yes" if p.get("step5_complex_implementation") is True
                else "No" if p.get("step5_complex_implementation") is False
                else "N/A"
            ),
            "Step 5 Confidence":              p.get("step5_confidence") or "N/A",
            "Step 5 Reason":                  p.get("step5_reason") or "N/A",
            "Filing Date":                    p.get("filing_date") or "Unknown",
            "Grant Date":                     p.get("grant_date") or "Not yet granted",
            "PTE (months)":                   p.get("pte") if p.get("pte") is not None else "N/A",
            "Pediatric Exclusivity":          "Yes" if p.get("pediatric_exclusivity") else "No",
            "Phase":                          phase if phase else "Info N/A",
            "Launch Date":                    "",
            "Approval Date":                  approval_date or "N/A",
            "Approval Date Source":           approval_source or "N/A",
            "Est. Approval Year":             p.get("estimated_approval_year") or "N/A",
            "Exclusivity Year":               p.get("exclusivity_year") or "N/A",
            "Controlling Patent Expiry Year": p.get("controlling_patent_expiry_year") or "N/A",
            "Years to Entry":                 p.get("years_to_entry") if p.get("years_to_entry") is not None else "N/A",
            "Avg Years to Entry":             p.get("avg_years_to_entry") if p.get("avg_years_to_entry") is not None else "N/A",
            "Score":                          p.get("score") if p.get("score") is not None else "N/A",
            "Avg Years to Entry (US & EP)":   p.get("avg_years_to_entry_us_ep") if p.get("avg_years_to_entry_us_ep") is not None else "N/A",
            "IP Dimension 1 Score":           p.get("ip_dimension_1_score") if p.get("ip_dimension_1_score") is not None else "N/A",
            "Source File":                    p.get("source_file", ""),
        })
    return rows


# ─────────────────────────────────────────────
# BQ helpers
# ─────────────────────────────────────────────

def sanitise_bq_columns(df: pd.DataFrame) -> pd.DataFrame:
    """BQ column names must match [A-Za-z_][A-Za-z0-9_]*."""
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", col)
        safe = re.sub(r"_+", "_", safe).strip("_")
        if safe and safe[0].isdigit():
            safe = f"col_{safe}"
        new_cols[col] = safe or "unnamed"
    df.rename(columns=new_cols, inplace=True)
    return df


def table_exists(project_id: str, dataset_id: str, table_id: str) -> bool:
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=project_id)
        client.get_table(f"{project_id}.{dataset_id}.{table_id}")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Load from results cache
# ─────────────────────────────────────────────

def load_drug_from_cache(drug_name: str) -> Optional[dict]:
    """Load a single drug's results from the JSON cache."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    path = RESULTS_CACHE_DIR / f"{safe}.json"

    if not path.exists():
        print(f"[ERROR] Cache file not found: {path}")
        print(f"[ERROR] Run the pipeline first: get_dimension_i_patent_data(\"{drug_name}\")")
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        drug = payload.get("drug", drug_name)
        patents = payload.get("patents", [])
        analysis_date = payload.get("analysis_date", "")
        print(f"[CACHE] Loaded '{drug}' — {len(patents)} patent(s), analysed: {analysis_date}")
        return payload
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Failed to read cache: {e}")
        return None


def load_all_from_cache() -> List[dict]:
    """Load all drug results from the JSON cache."""
    if not RESULTS_CACHE_DIR.exists():
        print(f"[ERROR] Results cache directory not found: {RESULTS_CACHE_DIR}")
        return []

    cache_files = sorted(RESULTS_CACHE_DIR.glob("*.json"))
    if not cache_files:
        print(f"[ERROR] No cached results found in: {RESULTS_CACHE_DIR}")
        return []

    results = []
    for path in cache_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            drug = payload.get("drug", path.stem)
            patents = payload.get("patents", [])
            print(f"[CACHE] {drug}: {len(patents)} patent(s)")
            results.append(payload)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Skipping {path.name}: {e}")

    return results


# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────

def upload_to_bigquery(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str,
    dry_run: bool = False,
) -> bool:
    """Upload a DataFrame to BigQuery."""
    if df.empty:
        print("[SKIP] Empty DataFrame — nothing to upload")
        return False

    df = sanitise_bq_columns(df)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    print(f"\n{'─'*50}")
    print(f"[BQ] Target:   {table_ref}")
    print(f"[BQ] Location: {location}")
    print(f"[BQ] Rows:     {len(df)}")
    print(f"[BQ] Columns:  {len(df.columns)}")
    print(f"{'─'*50}")

    print(f"\n[BQ] Column names:")
    for i, col in enumerate(df.columns):
        print(f"      {i+1:2d}. {col}")

    if "Drug_Name" in df.columns:
        drugs = df["Drug_Name"].unique()
        print(f"\n[BQ] Drug(s): {list(drugs)}")

    print(f"\n[BQ] First 3 rows preview:")
    preview_cols = [c for c in df.columns if c in (
        "Drug_Name", "Patent_Number", "Jurisdiction", "Tag",
        "Phase", "Filing_Date", "Years_to_Entry", "IP_Dimension_1_Score",
    )]
    if preview_cols:
        print(df[preview_cols].head(3).to_string(index=False))
    else:
        print(df.head(3).to_string(index=False, max_colwidth=30))

    if dry_run:
        print(f"\n[DRY RUN] Would upload {len(df)} row(s) to {table_ref}")
        print(f"[DRY RUN] No data was sent to BigQuery")
        return True

    # ── Actual upload ──
    try:
        import pandas_gbq
    except ImportError:
        print("[ERROR] pandas-gbq not installed. Run: pip install pandas-gbq")
        return False

    try:
        exists = table_exists(project_id, dataset_id, table_id)
        action = "Appending to existing" if exists else "Creating new"
        print(f"\n[BQ] {action} table...")

        pandas_gbq.to_gbq(
            df,
            destination_table=f"{dataset_id}.{table_id}",
            project_id=project_id,
            location=location,
            if_exists="append",
            progress_bar=True,
        )

        print(f"[BQ] ✓ Upload complete — {len(df)} row(s) → {table_ref}")
        return True

    except Exception as e:
        print(f"[BQ] ✗ Upload FAILED: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload patent analysis results directly to BigQuery (from results cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_to_bq.py --drug Bgm0504
  python upload_to_bq.py --drug Bgm0504 --dry-run
  python upload_to_bq.py --all
  python upload_to_bq.py --all --dry-run
  python upload_to_bq.py --drug Bgm0504 --project my-proj --dataset my_ds --table my_tbl
        """,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--drug", type=str, help="Upload a specific drug from results cache")
    source.add_argument("--all", action="store_true", help="Upload ALL drugs from results cache")

    parser.add_argument("--dry-run", action="store_true",
                        help="Preview rows without uploading")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help=f"Results cache dir. Default: {RESULTS_CACHE_DIR}")

    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT,
                        help=f"BQ project. Default: {DEFAULT_PROJECT}")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"BQ dataset. Default: {DEFAULT_DATASET}")
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE,
                        help=f"BQ table. Default: {DEFAULT_TABLE}")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION,
                        help=f"BQ location. Default: {DEFAULT_LOCATION}")

    args = parser.parse_args()

    global RESULTS_CACHE_DIR
    if args.cache_dir:
        RESULTS_CACHE_DIR = Path(args.cache_dir)

    print(f"\n{'='*60}")
    print(f"  LOE → BigQuery (direct upload)")
    print(f"  Cache dir: {RESULTS_CACHE_DIR}")
    print(f"  Target:    {args.project}.{args.dataset}.{args.table}")
    if args.dry_run:
        print(f"  Mode:      DRY RUN")
    print(f"{'='*60}\n")

    # ── Load patent data from cache ──
    if args.drug:
        payload = load_drug_from_cache(args.drug)
        if not payload:
            sys.exit(1)
        payloads = [payload]
    else:
        payloads = load_all_from_cache()
        if not payloads:
            sys.exit(1)

    # ── Build rows ──
    all_rows = []
    for payload in payloads:
        drug_name = payload.get("drug", "unknown")
        patents = payload.get("patents", [])
        if not patents:
            print(f"[SKIP] '{drug_name}' — no patents in cache")
            continue
        rows = build_rows(drug_name, patents)
        all_rows.extend(rows)
        print(f"[ROWS] '{drug_name}' → {len(rows)} row(s) built")

    if not all_rows:
        print(f"\n[ERROR] No rows to upload")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"\n[TOTAL] {len(df)} row(s) from {len(payloads)} drug(s)")

    # ── Upload ──
    success = upload_to_bigquery(
        df=df,
        project_id=args.project,
        dataset_id=args.dataset,
        table_id=args.table,
        location=args.location,
        dry_run=args.dry_run,
    )

    if success:
        print(f"\n{'='*60}")
        print(f"  ✓ Done")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"  ✗ Upload failed")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
