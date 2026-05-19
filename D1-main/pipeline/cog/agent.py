"""
agent.py
─────────
Google ADK agent for pharmaceutical patent analysis.

Imports from the refactored module structure:
  tools.py                 — main entry point (get_dimension_i_patent_data)
  gcs_lister.py            — GCS listing helpers
  indexer.py               — AlloyDB, indexing, sentinel, dedup helpers
  blocking_analyser.py     — analysis + business rules
  phase_fetcher.py         — BQ + phase assignment
  calculators.py           — all derived metric calculators
  approval_date_fetcher.py — approval date cascade
  excel_exporter.py        — Excel export + BigQuery upload
"""

import asyncio
import random
import shutil
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not needed on Cloud Run

try:
    from google.adk.agents import Agent
except ImportError:
    Agent = None  # ADK not installed — CLI mode via main.py

# ── Main entry point ──────────────────────────────────────────────────────────
from .tools import get_dimension_i_patent_data

# ── GCS ───────────────────────────────────────────────────────────────────────
from .gcs_lister import (
    GCS_BUCKET_NAME,
    GCS_PATENTS_PREFIX,
    get_gcs_client,
    list_drug_pdf_filenames_from_gcs,
)

# ── Indexer ───────────────────────────────────────────────────────────────────
from .indexer import (
    chroma_client,
    get_or_create_collection,
    index_text,
    upload_pdf_to_gemini,
    extract_text_via_gemini,
    extract_dates_from_pdf,
    cleanup_uploaded_file,
    sentinel_exists,
    find_in_any_collection,
    copy_from_collection,
    download_single_patent_pdf,
)

# ── Blocking analyser ─────────────────────────────────────────────────────────
from .blocking_analyser import (
    run_blocking_analysis,
    load_formulation_excel,
    is_non_analysable_patent,
    invalidate_drug_cache as invalidate_blocking_cache,
)

# Load formulation Excel once at startup (path from env var FORMULATION_EXCEL_PATH)
load_formulation_excel()

# ── Phase fetcher ─────────────────────────────────────────────────────────────
from .phase_fetcher import (
    fetch_clinical_timeline,
    assign_patent_phases,
    canonicalise_drug_name,
    BQ_TABLE_NAME,
    BQ_PROJECT_ID,
    BQ_DATASET_ID,
    BQ_SERVICE_ACCOUNT,
)

# ── Calculators ───────────────────────────────────────────────────────────────
from .calculators import run_calculations

# ── Approval date fetcher ─────────────────────────────────────────────────────
from .approval_date_fetcher import fetch_approval_dates

# ── Excel exporter + BigQuery ─────────────────────────────────────────────────
from .excel_exporter import (
    EXCEL_OUTPUT_DIR,
    export_to_excel,
    export_combined_excel,
    export_to_bigquery,
    export_combined_bigquery,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _list_all_drug_folders() -> list[str]:
    """List all drug folder names under GCS_PATENTS_PREFIX."""
    if not GCS_BUCKET_NAME:
        return []
    client       = get_gcs_client()
    prefix       = GCS_PATENTS_PREFIX.rstrip("/") + "/"
    all_blobs    = list(client.list_blobs(GCS_BUCKET_NAME, prefix=prefix))
    prefix_depth = len(prefix.split("/")) - 1
    drug_folders = {}
    for blob in all_blobs:
        parts = blob.name.split("/")
        if len(parts) > prefix_depth + 1:
            folder_name = parts[prefix_depth]
            if folder_name not in drug_folders:
                drug_folders[folder_name] = folder_name
    return list(drug_folders.keys())


# ─────────────────────────────────────────────
# Tool: process all drugs
# ─────────────────────────────────────────────

async def process_all_drugs() -> dict:
    """
    Processes ALL drug folders found in GCS.
    For each drug:
      - Indexes unindexed files (dates stored upfront, no backfill)
      - Runs blocking analysis from AlloyDB
      - Fetches approval dates (US + EU) via Step A->B->C cascade
      - Generates a per-drug Excel file + uploads to BigQuery
      - Generates a combined Excel file + uploads to BigQuery
    Returns a summary of all drugs processed.
    """
    print(f"\n{'='*60}")
    print(f"  PROCESS ALL DRUGS")
    print(f"{'='*60}\n")

    drug_folders = _list_all_drug_folders()
    if not drug_folders:
        return {
            "status":  "error",
            "message": f"No drug folders found in GCS bucket '{GCS_BUCKET_NAME}'.",
            "results": [],
        }

    print(f"[PLAN] Found {len(drug_folders)} drug(s): {drug_folders}")
    results = []

    for i, drug_name in enumerate(drug_folders, 1):
        print(f"\n[PROGRESS] {i}/{len(drug_folders)}: {drug_name}")
        try:
            summary = await _process_single_drug(drug_name)
            results.append(summary)
        except Exception as e:
            import traceback
            print(f"[ERROR] {drug_name}: {e}\n{traceback.format_exc()}")
            results.append({
                "drug":       drug_name,
                "status":     "error",
                "error":      str(e),
                "excel_path": None,
            })

    ok     = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    # ── Combined Excel + BigQuery (once, after all drugs) ─────────────────────
    analysis_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n[COMBINED] Generating combined Excel for all drugs...")
    combined_excel_path = export_combined_excel(analysis_date)
    if combined_excel_path:
        print(f"[COMBINED] All drugs Excel → {combined_excel_path}")
    else:
        print("[COMBINED] Could not generate combined Excel.")

    print(f"\n[COMBINED] Uploading combined results to BigQuery...")
    export_combined_bigquery(analysis_date)

    print(f"\n[ALL DONE] {len(ok)} succeeded | {len(failed)} failed")

    return {
        "status":              "complete",
        "total_drugs":         len(drug_folders),
        "succeeded":           len(ok),
        "failed":              len(failed),
        "results":             results,
        "excel_dir":           str(EXCEL_OUTPUT_DIR.resolve()),
        "combined_excel_path": combined_excel_path,
    }


async def _process_single_drug(drug_name: str) -> dict:
    """
    Full pipeline for one drug:
      1.  Index / copy / skip files
          Dates are extracted and stored upfront in parallel with text extraction.
          No backfill step — dates are ready immediately after indexing.
      2.  Fetch clinical timeline (BigQuery + fallback Excel, merged)
      3.  Blocking analysis (Steps 1-5)
      4.  Phase assignment + downstream calculations
      5.  Apply pediatric exclusivity adjustment (US +0.5 yr)
      6.  Assign score
      7.  Fetch real-world approval dates (Step A -> B -> C)
      8.  Recalculate with real approval dates
      9.  Export per-drug Excel + upload to BigQuery
    """
    # Resolve alias -> canonical INN so all lookups use the same name
    drug_name  = canonicalise_drug_name(drug_name)
    collection = get_or_create_collection(drug_name)
    pdf_refs   = list_drug_pdf_filenames_from_gcs(drug_name)

    if not pdf_refs:
        return {"drug": drug_name, "status": "no_pdfs", "excel_path": None}

    # ── Step 1: Index / copy / skip ──────────────────────────────────────────
    # Only US and EP patents are indexed. Others are skipped entirely.
    # Dates are stored upfront — no separate backfill pass needed.
    for ref in pdf_refs:
        filename = ref["filename"]

        if is_non_analysable_patent(filename):
            print(f"[SKIP] {filename} -- non-US/EP patent, not indexed")
            continue

        if sentinel_exists(collection, filename):
            print(f"[SKIP] {filename} -- already indexed")
            continue

        # Cross-collection dedup — copy carries dates from chunk metadata
        source_col = find_in_any_collection(filename)
        if source_col:
            print(f"[COPY] {filename} -- from '{source_col}' (dates included)")
            copy_from_collection(filename, source_col, collection, drug_name)
            continue

        # Full index: download -> upload -> text + dates in parallel -> store
        print(f"[INDEX] {filename} -- downloading...")
        pf = download_single_patent_pdf(ref["blob_name"], filename, drug_name)
        if not pf:
            continue

        try:
            uploaded_file = await upload_pdf_to_gemini(pf["path"])
            if not uploaded_file:
                continue

            # Text and dates extracted in parallel — dates stored immediately
            text, dates = await asyncio.gather(
                extract_text_via_gemini(uploaded_file, filename),
                extract_dates_from_pdf(pf["path"], filename),
            )

            if text:
                await index_text(drug_name, filename, text, collection, dates=dates)
                print(
                    f"[INDEX] {filename} -- stored | "
                    f"Filed: {(dates or {}).get('filing_date') or 'unknown'} | "
                    f"Granted: {(dates or {}).get('grant_date') or 'unknown'}"
                )

            await cleanup_uploaded_file(uploaded_file)
            await asyncio.sleep(1 + random.uniform(0, 0.5))

        except Exception as e:
            print(f"[ERROR] Processing {filename}: {e}")

        finally:
            if pf.get("tmp_dir"):
                shutil.rmtree(pf["tmp_dir"], ignore_errors=True)

    # ── Step 2: Fetch clinical timeline BEFORE blocking analysis ─────────────
    print(f"[AGENT] Fetching clinical timeline for '{drug_name}'...")
    timeline = await fetch_clinical_timeline(
        drug_name          = drug_name,
        bq_table_name      = BQ_TABLE_NAME,
        bq_project_id      = BQ_PROJECT_ID,
        bq_dataset_id      = BQ_DATASET_ID,
        bq_service_account = BQ_SERVICE_ACCOUNT,
    )
    geography_stages = timeline.get("geography_stages", {})
    drug_phase = {
        "US": geography_stages.get("United States"),
        "EP": geography_stages.get("EU"),
    }
    print(f"[AGENT] Drug phase -> US: {drug_phase['US']} | EP: {drug_phase['EP']}")

    # ── Step 3: Blocking analysis (Steps 1-5) ────────────────────────────────
    print(f"[AGENT] Running blocking analysis for '{drug_name}'...")
    patents = await run_blocking_analysis(
        drug_name  = drug_name,
        pdf_refs   = pdf_refs,
        collection = collection,
        drug_phase = drug_phase,
    )
    print(f"[AGENT] {len(patents)} patent(s) analysed")

    # ── Steps 4-6: Phase assignment + downstream calculations ─────────────────
    patents = assign_patent_phases(patents, timeline)
    patents = run_calculations(patents)

    # ── Step 7: Fetch real-world approval dates (Marketed only) ──────────────
    bq_companies = [
        c.strip() for c in
        str(timeline.get("company_name", "")).split(",") if c.strip()
    ]
    bq_brands = [
        b.strip() for b in
        str(timeline.get("brand_name", "")).split(",") if b.strip()
    ]

    us_marketed = any(
        p.get("phase_at_filing") == "Marketed"
        and (p.get("jurisdiction") or "").upper() == "US"
        for p in patents
    )
    eu_marketed = any(
        p.get("phase_at_filing") == "Marketed"
        and (p.get("jurisdiction") or "").upper() == "EP"
        for p in patents
    )

    approval = await fetch_approval_dates(
        drug_name    = drug_name,
        bq_companies = bq_companies,
        bq_brands    = bq_brands,
        fetch_us     = us_marketed,
        fetch_eu     = eu_marketed,
    )

    for p in patents:
        p["approval_date_us"]        = approval["US"]["date"]
        p["approval_date_eu"]        = approval["EU"]["date"]
        p["approval_date_us_source"] = approval["US"]["source"]
        p["approval_date_eu_source"] = approval["EU"]["source"]

    # ── Step 8: Recalculate with real approval dates ──────────────────────────
    patents = run_calculations(patents)

    # ── Step 9: Export per-drug Excel + BigQuery ──────────────────────────────
    analysis_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[AGENT] Exporting per-drug Excel for '{drug_name}'...")
    excel_path = export_to_excel(drug_name, patents, analysis_date)

    print(f"[AGENT] Uploading '{drug_name}' results to BigQuery...")
    export_to_bigquery(drug_name, patents)

    # NOTE: combined Excel + BQ are handled once by process_all_drugs()
    # after all drugs complete, to avoid redundant re-uploads per drug.

    if excel_path:
        return {
            "drug":       drug_name,
            "status":     "ok",
            "excel_path": excel_path,
            "patents":    len(patents),
        }
    return {
        "drug":       drug_name,
        "status":     "no_excel",
        "excel_path": None,
        "patents":    len(patents),
    }


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

if Agent is None:
    root_agent = None  # CLI mode — ADK not installed
else:
  root_agent = Agent(
    model="gemini-2.5-flash",
    name="ip_dimensions",
    description="Analyze pharmaceutical patent data from GCS and AlloyDB.",
    instruction="""You are a patent analyst that extracts patent protection data from local files.

─────────────────────────────────────────────
WHEN TO CALL WHICH TOOL:
─────────────────────────────────────────────

1. process_all_drugs()
   -> When the user says anything like:
     "analyse all drugs", "process all", "run all", "generate all excels",
     "do all drugs", "start", "run everything"
   -> No arguments needed. Call it immediately.

2. get_dimension_i_patent_data(drug_name="<n>")
   -> When the user provides a specific drug name or code
     e.g. "ozempic", "ASC30", "semaglutide", "analyse tirzepatide"
   -> Use the exact name the user provided (preserve case)

─────────────────────────────────────────────
WHEN NOT TO CALL ANY TOOL:
─────────────────────────────────────────────
   -> User says "hi", "hello", "help" with no drug name or action
   -> No specific name or action mentioned

─────────────────────────────────────────────
RESPONSE FORMAT — single drug result:
─────────────────────────────────────────────

Drug Name: <drug_name>
Analysis Date: <analysis_date>
Phase Data Source: <phase_data_source>
Processing Time: <processing_time_seconds> seconds

| Patent Number | Jurisdiction | Tag | Blocking Category | Step 1 Claim Category | Step 2 Matched Elements | S2: Active Ingredient & Form | S2: Formulation Details | S2: Route of Administration | S2: Device Description | S2: Combination Tech/Process | Reason | Filing Date | Grant Date | PTE (months) | Pediatric Exclusivity | Phase | Launch Date | Approval Date | Est. Approval Year | Exclusivity Year | Controlling Patent Expiry Year | Years to Entry | Avg Years to Entry | Score | Source File |

Each patent is one row:
- Tag: "BLOCKING", "NON-BLOCKING", or "SKIPPED -- indexed only"
- Blocking Category: value or "N/A"
- Step 1 Claim Category: one of the 7 categories, or "N/A" for SKIPPED
- Step 2 Matched Elements: comma-separated list or "None matched" or "N/A"
- Filing Date: value or "Unknown"
- Grant Date: value or "Not yet granted"
- PTE (months): number or "N/A"
- Pediatric Exclusivity: "Yes" or "No"
- Phase: phase_at_filing or "Info N/A"
- Approval Date: FDA date for US, EMA/EC date for EP, "N/A" for others
- Est. Approval Year: current year + 3 (Phase 3) or + 5 (Phase 2), else "N/A"
- Exclusivity Year: Est. Approval Year + 5 (US) or + 10 (EP), else "N/A"
- Controlling Patent Expiry Year: effective filing year + 20 of latest BLOCKING patent, else "N/A"
- Years to Entry: max(Controlling Patent Expiry, Exclusivity Year) - current year, else "N/A"
- Avg Years to Entry: average of all jurisdictions' years_to_entry, else "N/A"
- Score: <=6->5, 7-8->4, 9-11->3, 12-13->2, >13->1, else "N/A"
- Avg Years to Entry (US & EP): average of US and EP years_to_entry only, else "N/A"
- IP Dimension 1 Score: <=6->5, 7-8->4, 9-11->3, 12-13->2, >13->1, else "N/A"

After table:
Source Files Used:
* <each source file>

Excel Export:
<excel_path if present, else "Excel export was not available for this run.">

Combined Excel (all drugs):
<combined_excel_path if present, else "Not available.">

─────────────────────────────────────────────
RESPONSE FORMAT — all drugs result:
─────────────────────────────────────────────

| Drug | Status | Patents | Excel File |
|------|--------|---------|------------|
| <drug> | Done | <n> | <path> |
| <drug> | Failed | - | - |

All Excel files saved to: <excel_dir>
Combined Excel: <combined_excel_path>
Total: <total_drugs> drugs | <succeeded> succeeded | <failed> failed

─────────────────────────────────────────────
FIELD RULES:
─────────────────────────────────────────────
- phase_data_source: "bigquery" -> "BigQuery", "unavailable" -> "Unavailable"
- Never omit any patent row
- Never fabricate drug names or patent data
""",
    tools=[get_dimension_i_patent_data, process_all_drugs],
)