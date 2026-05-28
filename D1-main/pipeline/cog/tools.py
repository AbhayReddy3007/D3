"""
tools.py
─────────
Main orchestration entry point for the patent analysis pipeline.

Imports and delegates to the following modules:

  gcs_lister.py            — Step 1:  List PDFs from GCS
  indexer.py               — Steps 2–3: Index + deduplicate + date backfill
  blocking_analyser.py     — Steps 4–5: Blocking analysis + business rules
  phase_fetcher.py         — Steps 6–7: Clinical phase (BigQuery + fallback Excel)
  calculators.py           — Steps 8–12: Approval year, exclusivity, expiry,
                                          years to entry, pediatric, PTE, score
  approval_date_fetcher.py — Step 13:   Real-world approval dates (FDA/EMA/Gemini/news)
  excel_exporter.py        — Steps 14–15: Per-drug + combined Excel export

Results are cached as JSON files (results_cache/ and analysis_cache/).
When re-running, only NEW drugs or drugs with NEW patents get re-analysed.

Usage:
    import asyncio
    from tools import get_dimension_i_patent_data

    # Analyse all jurisdictions
    result = asyncio.run(get_dimension_i_patent_data("Semaglutide"))

    # Analyse only US and EU patents
    result = asyncio.run(get_dimension_i_patent_data("Semaglutide", jurisdictions=["US", "EP"]))
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Module imports ───────────────────────────────────────────────────────────

from .gcs_lister import list_drug_pdf_filenames_from_gcs, GCS_BUCKET_NAME

from .indexer import (
    get_or_create_collection,
    run_indexing,
    chroma_client,
)

from .blocking_analyser import run_blocking_analysis, load_formulation_excel

from .blocking_analyser import (
    invalidate_drug_cache as invalidate_blocking_cache,
)

from .phase_fetcher import (
    fetch_clinical_timeline,
    assign_patent_phases,
)

from .calculators import run_calculations

from .approval_date_fetcher import fetch_approval_dates

from .excel_exporter import export_to_excel, export_combined_excel

from . import gcs_cache

# ── Environment config ───────────────────────────────────────────────────────

BQ_TABLE_NAME      = os.getenv("BQ_TABLE_NAME")
BQ_PROJECT_ID      = os.getenv("BQ_PROJECT_ID")
BQ_DATASET_ID      = os.getenv("BQ_DATASET_ID")
BQ_SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# ── Results cache (GCS-backed JSON) ──────────────────────────────────────────
#
# Structure in GCS:
#   gs://{bucket}/pipeline_cache/results_cache/{drug_name}.json
#     {
#       "drug": "...",
#       "analysis_date": "...",
#       "source_files": ["file1.pdf", "file2.pdf"],
#       "patents": [ {patent_dict}, ... ]
#     }

_RESULTS_CACHE_SUBFOLDER = "results_cache"


def _results_cache_filename(drug_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    return f"{safe}.json"


def _store_results(drug_name: str, patents: list, analysis_date: str, source_files: list):
    """Stores the full analysis results for a drug as a JSON file in GCS."""
    try:
        payload = {
            "drug":          drug_name,
            "analysis_date": analysis_date,
            "source_files":  sorted(source_files),
            "patents":       patents,
        }
        uri = gcs_cache.write_json(
            _RESULTS_CACHE_SUBFOLDER,
            _results_cache_filename(drug_name),
            payload,
        )
        print(f"[RESULTS CACHE] Stored {len(patents)} patent(s) for '{drug_name}' → {uri}")
    except Exception as e:
        print(f"[RESULTS CACHE] Failed to store results: {e}")


def _load_cached_results(
    drug_name: str,
    current_file_count: int,
    current_files: Optional[List[str]] = None,
) -> Optional[list]:
    """
    Loads cached results for a drug from GCS if they exist and are not stale.

    Staleness detection:
      - If the EXACT SAME set of files → full cache hit (return all results)
      - If files were ADDED → returns None so the pipeline can do incremental
        analysis (per-patent cache in blocking_analyser handles the rest)
      - If files were REMOVED → returns None (full re-analysis needed)

    Returns:
        List of patent dicts if fully cached, None otherwise.
    """
    payload = gcs_cache.read_json(
        _RESULTS_CACHE_SUBFOLDER,
        _results_cache_filename(drug_name),
    )
    if payload is None:
        return None

    try:
        cached_files = set(payload.get("source_files", []))
        cached_date  = payload.get("analysis_date", "")
        patents      = payload.get("patents", [])

        if not patents:
            return None

        # Per-file staleness check
        if current_files:
            current_files_set = set(current_files)

            if current_files_set == cached_files:
                print(f"[RESULTS CACHE] Full cache hit for '{drug_name}' "
                      f"({len(patents)} patent(s), analysed: {cached_date})")
                return patents

            elif current_files_set > cached_files:
                new_files = current_files_set - cached_files
                print(f"[RESULTS CACHE] {len(new_files)} new file(s) for '{drug_name}': "
                      f"{sorted(new_files)}")
                print(f"[RESULTS CACHE] Per-patent cache will handle incremental analysis")
                return None

            else:
                removed = cached_files - current_files_set
                print(f"[RESULTS CACHE] {len(removed)} file(s) removed for '{drug_name}' "
                      f"→ full re-analysis")
                return None
        else:
            if len(cached_files) != current_file_count:
                print(f"[RESULTS CACHE] Cache stale for '{drug_name}': "
                      f"cached {len(cached_files)} vs current {current_file_count}")
                return None

            print(f"[RESULTS CACHE] Loaded {len(patents)} cached patent(s) for '{drug_name}' "
                  f"(analysed: {cached_date})")
            return patents

    except Exception as e:
        print(f"[RESULTS CACHE] Failed to load cache for '{drug_name}': {e}")

    return None


def _filter_by_jurisdictions(patents: list, jurisdictions: Optional[List[str]]) -> list:
    """Filters patents to only the specified jurisdictions."""
    if not jurisdictions:
        return patents  # no filter = all jurisdictions

    jur_set = {j.upper() for j in jurisdictions}
    filtered = [p for p in patents if (p.get("jurisdiction") or "").upper() in jur_set]
    print(f"[FILTER] {len(patents)} patents → {len(filtered)} after jurisdiction filter {list(jur_set)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def get_dimension_i_patent_data(
    drug_name:                 str,
    has_pediatric_exclusivity: bool = True,
    reindex:                   bool = False,
    jurisdictions:             Optional[List[str]] = None,
    bq_table_name:             Optional[str] = None,
    bq_project_id:             Optional[str] = None,
    bq_dataset_id:             Optional[str] = None,
    bq_service_account:        Optional[str] = None,
) -> dict:
    """
    Full RAG pipeline for patent analysis of a single drug.

    Args:
        drug_name:                 Drug name (must match GCS folder)
        has_pediatric_exclusivity: Default True
        reindex:                   Force re-indexing
        jurisdictions:             Filter results to these jurisdictions (e.g. ["US", "EP"]).
                                   None = all jurisdictions.
        bq_*:                      Override env vars

    Returns dict with patents list, scores, dates, etc.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[PIPELINE] Starting for: {drug_name}")
    if jurisdictions:
        print(f"[PIPELINE] Jurisdiction filter: {jurisdictions}")
    print(f"{'='*60}")

    _bq_table   = bq_table_name      or BQ_TABLE_NAME
    _bq_project = bq_project_id      or BQ_PROJECT_ID
    _bq_dataset = bq_dataset_id      or BQ_DATASET_ID
    _bq_sa      = bq_service_account or BQ_SERVICE_ACCOUNT

    # ── Step 1: List PDFs from GCS ────────────────────────────────────────────
    print(f"\n[STEP 1] Listing PDFs from GCS...")
    pdf_refs = list_drug_pdf_filenames_from_gcs(drug_name)
    print(f"[STEP 1] {len(pdf_refs)} PDF(s) found")

    if not pdf_refs:
        return {
            "drug_name":     drug_name,
            "error":         f"No PDFs found for '{drug_name}' in GCS bucket '{GCS_BUCKET_NAME}'.",
            "patents":       [],
            "source_files":  [],
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        }

    # ── Check results cache — skip analysis if already done ───────────────────
    if not reindex:
        current_files = [ref["filename"] for ref in pdf_refs]
        cached = _load_cached_results(drug_name, len(pdf_refs), current_files=current_files)
        if cached:
            # Apply jurisdiction filter if requested
            filtered = _filter_by_jurisdictions(cached, jurisdictions)

            # Recalculate scores for the filtered subset
            filtered = run_calculations(filtered)

            analysis_date = datetime.now().strftime("%Y-%m-%d")

            # Generate Excel even from cache
            excel_path          = export_to_excel(drug_name, filtered, analysis_date)
            combined_excel_path = export_combined_excel(analysis_date)

            elapsed = time.time() - t0
            print(f"\n[PIPELINE] Using cached results — {len(filtered)} patent(s) in {elapsed:.1f}s")

            return {
                "drug_name":               drug_name,
                "analysis_date":           analysis_date,
                "patents":                 filtered,
                "source_files":            [ref["filename"] for ref in pdf_refs],
                "processing_time_seconds": round(elapsed, 1),
                "clinical_timeline":       {},
                "phase_data_source":       "cached",
                "excel_path":              excel_path,
                "combined_excel_path":     combined_excel_path,
                "from_cache":              True,
            }

    # ── Steps 2–3: Index + cross-collection dedup + date backfill ─────────────
    print(f"\n[STEPS 2–3] Indexing and date backfill...")
    collection = get_or_create_collection(drug_name)

    if reindex:
        print("[PIPELINE] Reindex requested — deleting existing collection...")
        chroma_client.delete_collection(name=collection.name)
        collection = get_or_create_collection(drug_name)

    await run_indexing(drug_name, pdf_refs, collection, reindex=reindex)

    # ── Steps 4–5 (pre): Fetch clinical timeline BEFORE blocking analysis ──────
    # Needed so Step 3 knows whether to use FDA/EMA reviews (Marketed)
    # or peer-reviewed journals (Clinical).
    print(f"\n[STEPS 4–5 PRE] Fetching clinical timeline for Step 3 phase routing...")
    timeline = await fetch_clinical_timeline(
        drug_name          = drug_name,
        bq_table_name      = _bq_table,
        bq_project_id      = _bq_project,
        bq_dataset_id      = _bq_dataset,
        bq_service_account = _bq_sa,
    )

    # Convert to {jurisdiction: phase} for blocking analyser
    geography_stages = timeline.get("geography_stages", {})
    drug_phase = {
        "US": geography_stages.get("United States"),
        "EP": geography_stages.get("EU"),
    }
    # Add all other geographies found
    _GEO_TO_JUR = {
        "United States": "US", "EU": "EP", "Japan": "JP", "China": "CN",
        "India": "IN", "South Korea": "KR", "Australia": "AU", "Canada": "CA",
        "Brazil": "BR", "Mexico": "MX", "Russia": "RU",
    }
    for geo_name, phase in geography_stages.items():
        jur = _GEO_TO_JUR.get(geo_name)
        if jur and jur not in drug_phase:
            drug_phase[jur] = phase

    phase_summary = " | ".join(f"{k}: {v}" for k, v in sorted(drug_phase.items()) if v)
    print(f"[STEPS 4–5 PRE] Drug phase → {phase_summary}")

    # ── Steps 4–5: Blocking analysis + business rules ─────────────────────────
    print(f"\n[STEPS 4–5] Running blocking analysis...")
    patents = await run_blocking_analysis(
        drug_name, pdf_refs, collection, drug_phase=drug_phase,
        force_reanalyse=reindex,
    )
    print(f"[STEPS 4–5] {len(patents)} patent(s) analysed")

    # ── Steps 6–7: Assign phase per patent (timeline already fetched above) ───
    print(f"\n[STEPS 6–7] Assigning clinical phase to patents...")
    patents = assign_patent_phases(patents, timeline)

    # ── Steps 8–12: Derived calculations (first pass) ─────────────────────────
    print(f"\n[STEPS 8–12] Running first-pass calculations...")
    patents = run_calculations(patents)

    # ── Step 13: Fetch real-world approval dates (Marketed only) ──────────────
    print(f"\n[STEP 13] Fetching real-world approval dates...")
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
    print(f"[STEP 13] US Marketed: {us_marketed} | EU Marketed: {eu_marketed}")

    approval = await fetch_approval_dates(
        drug_name    = drug_name,
        bq_companies = bq_companies,
        bq_brands    = bq_brands,
        fetch_us     = us_marketed,
        fetch_eu     = eu_marketed,
    )

    # Attach approval dates to all patents
    for p in patents:
        p["approval_date_us"]        = approval["US"]["date"]
        p["approval_date_eu"]        = approval["EU"]["date"]
        p["approval_date_us_source"] = approval["US"]["source"]
        p["approval_date_eu_source"] = approval["EU"]["source"]

    # ── Step 13b: Recalculate with real approval dates ─────────────────────────
    print(f"\n[STEP 13b] Recalculating with real approval dates...")
    patents = run_calculations(patents)

    analysis_date = datetime.now().strftime("%Y-%m-%d")

    # ── Store ALL results in cache DB (before jurisdiction filtering) ──────────
    _store_results(drug_name, patents, analysis_date, [ref["filename"] for ref in pdf_refs])

    # ── Apply jurisdiction filter if requested ────────────────────────────────
    if jurisdictions:
        patents = _filter_by_jurisdictions(patents, jurisdictions)
        # Recalculate scores for the filtered subset
        patents = run_calculations(patents)

    # ── Step 14: Export per-drug Excel ────────────────────────────────────────
    print(f"\n[STEP 14] Exporting per-drug Excel...")
    excel_path = export_to_excel(drug_name, patents, analysis_date)

    # ── Step 15: Regenerate combined Excel ────────────────────────────────────
    print(f"\n[STEP 15] Regenerating combined Excel...")
    combined_excel_path = export_combined_excel(analysis_date)

    elapsed = time.time() - t0
    print(f"\n[PIPELINE] Done in {elapsed:.1f}s — {len(patents)} patent(s)")
    print(f"{'='*60}\n")

    return {
        "drug_name":               drug_name,
        "analysis_date":           analysis_date,
        "patents":                 patents,
        "source_files":            [ref["filename"] for ref in pdf_refs],
        "processing_time_seconds": round(elapsed, 1),
        "clinical_timeline":       timeline,
        "phase_data_source":       timeline.get("source", "unavailable"),
        "excel_path":              excel_path,
        "combined_excel_path":     combined_excel_path,
        "from_cache":              False,
    }
