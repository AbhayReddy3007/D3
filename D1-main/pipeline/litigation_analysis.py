#!/usr/bin/env python3
"""
litigation_analysis.py
──────────────────────
Finds all patent litigations and court cases for one or more drugs.

Supports three input modes for drug selection:
  1. Explicit list via --drugs
  2. Built-in GLP-1 query via --use-default-glp1-query
  3. Custom BigQuery SQL via --drug-query

Usage examples:
    python litigation_analysis.py --drugs semaglutide tirzepatide --max-workers 2

    python litigation_analysis.py --use-default-glp1-query --max-workers 4

    python litigation_analysis.py \
        --drug-query "SELECT DISTINCT cleaned_generic_name FROM project.dataset.table" \
        --max-workers 4
"""

import argparse
import asyncio
import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
from google.genai import types
def _safe_parse_json(text: str):
    """Parse JSON string, handling common Gemini output quirks without json_repair.

    Handles: trailing commas, single quotes, unquoted keys, truncated output.
    Returns parsed object (dict/list) or None on failure.
    """
    if not text or not text.strip():
        return None

    # Attempt 1: straight parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas before } or ]
    import re as _re
    cleaned = _re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: single quotes → double quotes (careful with apostrophes)
    try:
        return json.loads(cleaned.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    # Attempt 4: truncated JSON — try closing brackets
    for suffix in ["]", "}", "]}", "]}']:
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            continue

    print(f"[WARN] Could not parse JSON (first 200 chars): {text[:200]}")
    return None

from constants import (
    BQ_PROJECT_ID,
    BQ_DATASET,
    BQ_TABLE,
    GEMINI_BRAND_AND_INNOVATOR_MODEL,
    GEMINI_LITIGATION_SEARCH_MODEL,
    LITIGATION_SEARCH_TEMPERATURE,
    BRAND_LOOKUP_TEMPERATURE,
    PATENTS_PER_LLM_CALL,
    PATENT_GROUP_THREAD_BATCH_SIZE,
)
from prompts import BRAND_AND_INNOVATOR_PROMPT, LITIGATION_SEARCH_PROMPT

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

RETRIABLE_ERRORS_ALL        = (ResourceExhausted, InternalServerError, ServiceUnavailable)
RETRIABLE_ERRORS_RATE_LIMIT = (ResourceExhausted,)

# ── Default BigQuery query for GLP-1 drugs ─────────────────────────────────────
LITIGATION_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.litigation_analysis_table"

# Explicit BQ schema — avoids autodetect mis-typing nullable string columns
from google.cloud.bigquery import SchemaField
LITIGATION_TABLE_SCHEMA = [
    SchemaField("drug_name",         "STRING",    mode="REQUIRED"),
    SchemaField("innovator",         "STRING",    mode="NULLABLE"),
    SchemaField("brand_names",       "STRING",    mode="NULLABLE"),   # JSON array as string
    SchemaField("patent_number",     "STRING",    mode="NULLABLE"),
    SchemaField("case_type",         "STRING",    mode="NULLABLE"),
    SchemaField("challenger",        "STRING",    mode="NULLABLE"),
    SchemaField("case_number",       "STRING",    mode="NULLABLE"),
    SchemaField("court",             "STRING",    mode="NULLABLE"),
    SchemaField("status",            "STRING",    mode="NULLABLE"),
    SchemaField("filing_date",       "STRING",    mode="NULLABLE"),
    SchemaField("outcome",           "STRING",    mode="NULLABLE"),
    SchemaField("summary",           "STRING",    mode="NULLABLE"),
    SchemaField("analysis_date",     "DATE",      mode="NULLABLE"),
    SchemaField("search_time_seconds","FLOAT64",  mode="NULLABLE"),
    SchemaField("total_cases",       "INT64",     mode="NULLABLE"),
    SchemaField("unique_challengers","INT64",     mode="NULLABLE"),
    SchemaField("loaded_at",         "TIMESTAMP", mode="REQUIRED"),
]

DEFAULT_DRUG_QUERY = """
SELECT DISTINCT cleaned_generic_name
FROM `cognito-prod-394707.cognito_prod_datamart.vw_drug_details_full`
WHERE
(
    UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
    OR UPPER(cleaned_Target) LIKE '%GLP-1%'
    OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
    OR (
        data_source = 'IPD'
        AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
    )
)
AND Mechanism_of_Action IS NOT NULL
AND LOWER(Mechanism_of_Action) NOT LIKE '%antagonist%'
"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS — single drug
# ══════════════════════════════════════════════════════════════════════════════

async def list_all_litigations(drug_name: str) -> dict:
    """
    Find all patent litigations and court cases for a drug.

    Searches for:
    - ANDA Paragraph IV patent challenges (US)
    - Inter Partes Review (IPR) at PTAB
    - European Patent Office (EPO) oppositions
    - India patent litigation (Delhi High Court)
    - Compounding pharmacy lawsuits

    Returns:
        dict: {
            "drug_name": str,
            "brand_names": list,
            "innovator": str,
            "litigations": list of litigation dicts,
            "litigations_by_type": dict grouped by case type,
            "summary": str,
            "analysis_date": str,
            "search_time_seconds": float,
            "statistics": {
                "total_cases": int,
                "by_type": dict,
                "unique_challengers": int
            }
        }
    """
    t0 = time.time()
    print(f"[Patent Litigation] Finding litigation/court cases for {drug_name}...")

    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    brand_lookup_config = types.GenerateContentConfig(
        tools=tools,
        temperature=BRAND_LOOKUP_TEMPERATURE,
    )
    litigation_search_config = types.GenerateContentConfig(
        tools=tools,
        temperature=LITIGATION_SEARCH_TEMPERATURE,
    )

    # ── Step 1: Brand names and innovator ─────────────────────────────────────
    brand_names, innovator = await _get_brand_and_innovator(drug_name, brand_lookup_config)

    # ── Step 2: Patents from BigQuery ─────────────────────────────────────────
    print(f"[Patent Litigation] Step 2: Retrieving patents from BigQuery...")
    patents = _get_patent_numbers_from_bigquery(drug_name)
    print(f"[Patent Litigation] Found {len(patents)} patents for {drug_name}")

    # ── Step 3: Litigation search per patent group ────────────────────────────
    print(f"[Patent Litigation] Step 3: Searching litigation for grouped patents...")
    all_litigations = await _patent_level_litigation_search(
        patents=patents,
        drug_name=drug_name,
        innovator=innovator,
        config=litigation_search_config,
    )
    print(f"[Patent Litigation] Found {len(all_litigations)} unique cases across {len(patents)} patents")

    # ── Step 4: Consolidate and organize results ──────────────────────────────
    print(f"[Patent Litigation] Step 4: Consolidating results...")
    all_litigations, cases_by_type = _consolidate_litigation_results(all_litigations)

    # ── Step 5: Deduplicate cases ─────────────────────────────────────────────
    print(f"[Patent Litigation] Step 5: Deduplicating cases...")
    all_litigations = _deduplicate_litigations(all_litigations)
    cases_by_type   = _group_by_type(all_litigations)

    # ── Step 6: Generate summary ──────────────────────────────────────────────
    print(f"[Patent Litigation] Step 6: Generating summary...")
    summary = _generate_summary(all_litigations, cases_by_type)

    litigation_data = {
        "drug_name":           drug_name,
        "brand_names":         brand_names,
        "innovator":           innovator,
        "litigations":         all_litigations,
        "litigations_by_type": cases_by_type,
        "summary":             summary,
        "analysis_date":       datetime.now().strftime("%Y-%m-%d"),
        "search_time_seconds": round(time.time() - t0, 1),
        "statistics": {
            "total_cases":        len(all_litigations),
            "by_type":            {k: len(v) for k, v in cases_by_type.items()},
            "unique_challengers": len({
                lit.get("challenger", "")
                for lit in all_litigations
                if lit.get("challenger")
            }),
        },
    }

    print(f"[Patent Litigation] Done — {len(all_litigations)} litigation(s) in {time.time()-t0:.1f}s")
    return litigation_data


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def _gemini_generate_with_retry(
    model: str,
    contents,
    config,
    max_retries: int = 3,
    base_delay: float = 2.0,
    rate_limit_only: bool = False,
):
    """Async Gemini call with exponential backoff retry."""
    retriable_errors = RETRIABLE_ERRORS_RATE_LIMIT if rate_limit_only else RETRIABLE_ERRORS_ALL
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await gemini_client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except retriable_errors as e:
            last_exception = e
            if attempt == max_retries:
                print(f"[Gemini Retry] All {max_retries} retries exhausted: {type(e).__name__}")
                raise
            delay = base_delay * (2 ** attempt) * (1 + random.uniform(0, 0.25))
            print(f"[Gemini Retry] Attempt {attempt+1}/{max_retries} failed: {type(e).__name__}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    raise last_exception


def _gemini_generate_with_retry_sync(
    model: str,
    contents,
    config,
    max_retries: int = 3,
    base_delay: float = 2.0,
    rate_limit_only: bool = False,
):
    """Sync Gemini call with exponential backoff retry (used inside asyncio.to_thread)."""
    retriable_errors = RETRIABLE_ERRORS_RATE_LIMIT if rate_limit_only else RETRIABLE_ERRORS_ALL
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return gemini_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except retriable_errors as e:
            last_exception = e
            if attempt == max_retries:
                print(f"[Gemini Retry] All {max_retries} retries exhausted: {type(e).__name__}")
                raise
            delay = base_delay * (2 ** attempt) * (1 + random.uniform(0, 0.25))
            print(f"[Gemini Retry] Attempt {attempt+1}/{max_retries} failed: {type(e).__name__}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise last_exception


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

async def _get_brand_and_innovator(
    drug_name: str,
    config: types.GenerateContentConfig,
) -> tuple:
    """Step 1 — Retrieve brand names and innovator for a drug via Gemini + Google Search."""
    print(f"[Patent Litigation] Step 1: Finding brand names and innovator...")

    brand_names = []
    innovator   = ""
    try:
        brand_resp = await _gemini_generate_with_retry(
            model=GEMINI_BRAND_AND_INNOVATOR_MODEL,
            contents=BRAND_AND_INNOVATOR_PROMPT.format(drug_name=drug_name),
            config=config,
        )
        brand_json = _extract_json_from_response(brand_resp.text.strip())
        if brand_json:
            brand_data = _safe_parse_json(brand_json)
            if brand_data is None:
                brand_data = {}
            if isinstance(brand_data, str):
                brand_data = json.loads(brand_data)
            brand_names = brand_data.get("brand_names", []) or []
            innovator   = brand_data.get("innovator", "") or ""
            print(
                f"[Patent Litigation] Brands: {', '.join(brand_names) if brand_names else 'none'} | "
                f"Innovator: {innovator}"
            )
    except Exception as e:
        print(f"[Patent Litigation] Brand lookup error: {e}")

    return brand_names, innovator


async def _patent_level_litigation_search(
    patents: list,
    drug_name: str,
    innovator: str,
    config: types.GenerateContentConfig,
) -> list:
    """Step 3 — Run grouped litigation search with semaphore-controlled concurrency."""
    if not patents:
        return []

    total_patents = len(patents)
    patent_groups = [
        patents[i:i + PATENTS_PER_LLM_CALL]
        for i in range(0, len(patents), PATENTS_PER_LLM_CALL)
    ]
    semaphore = asyncio.Semaphore(PATENT_GROUP_THREAD_BATCH_SIZE)

    async def _run_patent_group(patent_group: list) -> tuple:
        async with semaphore:
            group_litigations = await asyncio.to_thread(
                _search_litigation_for_patent_group_sync,
                patent_group,
                drug_name,
                innovator,
                config,
            )
            return len(patent_group), group_litigations

    total_groups            = len(patent_groups)
    progress_interval       = 10 if total_patents >= 50 else 5
    next_progress_milestone = min(progress_interval, total_patents)

    print(
        f"[Patent Litigation] Concurrency: {PATENT_GROUP_THREAD_BATCH_SIZE} group(s) at a time, "
        f"{PATENTS_PER_LLM_CALL} patents per call | "
        f"Progress every {progress_interval} patents for {total_patents} total"
    )

    tasks          = [asyncio.create_task(_run_patent_group(g)) for g in patent_groups]
    all_litigations  = []
    completed_patents = 0
    completed_groups  = 0

    for task in asyncio.as_completed(tasks):
        group_patent_count, group_litigations = await task
        completed_patents += group_patent_count
        completed_groups  += 1
        all_litigations.extend(group_litigations)

        if completed_patents >= next_progress_milestone or completed_patents == total_patents:
            pct = (completed_patents / total_patents) * 100
            print(
                f"[Patent Litigation] Progress: {completed_patents}/{total_patents} patents "
                f"({pct:.1f}%) | {completed_groups}/{total_groups} groups"
            )
            while next_progress_milestone <= completed_patents:
                next_progress_milestone += progress_interval

    return all_litigations


def _search_litigation_for_patent_group_sync(
    patent_numbers: list,
    drug_name: str,
    innovator: str,
    config: types.GenerateContentConfig,
) -> list:
    """Search all litigation categories for a group of patents in one LLM call (sync, runs in thread)."""
    if not patent_numbers:
        return []

    patent_numbers_text = "\n".join(f"- {p}" for p in patent_numbers)
    prompt = LITIGATION_SEARCH_PROMPT.format(
        drug_name=drug_name,
        innovator=innovator,
        patent_numbers=patent_numbers_text,
    )

    try:
        resp = _gemini_generate_with_retry_sync(
            model=GEMINI_LITIGATION_SEARCH_MODEL,
            contents=prompt,
            config=config,
        )
        if not resp.text:
            return []
        json_str = _extract_json_from_response(resp.text.strip())
        if json_str:
            data = _safe_parse_json(json_str)
            if data is None:
                data = []
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, list):
                requested = {str(p) for p in patent_numbers}
                return [
                    item for item in data
                    if isinstance(item, dict)
                    and str(item.get("patent_number", "")) in requested
                ]
    except Exception as e:
        print(f"[Patent Litigation] Search error for group {patent_numbers}: {e}")

    return []


def _consolidate_litigation_results(patent_results: list) -> tuple:
    """Step 4 — Group flat litigation list by case_type."""
    cases_by_type: dict = {}
    for lit in patent_results:
        case_type = lit.get("case_type", "Other")
        cases_by_type.setdefault(case_type, []).append(lit)
    return patent_results, cases_by_type


def _deduplicate_litigations(litigations: list) -> list:
    """Step 5 — Remove duplicate cases based on (patent_number, case_type, challenger)."""
    seen:   set  = set()
    unique: list = []
    for lit in litigations:
        key = (
            str(lit.get("patent_number", "")).strip(),
            str(lit.get("case_type",     "")).strip(),
            str(lit.get("challenger",    "")).strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(lit)
    return unique


def _group_by_type(litigations: list) -> dict:
    """Re-group a deduplicated list by case_type."""
    groups: dict = {}
    for lit in litigations:
        groups.setdefault(lit.get("case_type", "Other"), []).append(lit)
    return groups


def _generate_summary(all_litigations: list, cases_by_type: dict) -> str:
    """Step 6 — Build a plain-text summary."""
    challengers = {
        lit.get("challenger", "")
        for lit in all_litigations
        if lit.get("challenger")
    }
    summary = (
        f"Found {len(all_litigations)} litigation case(s) across "
        f"{len(cases_by_type)} category/categories. "
        f"Key challengers: {', '.join(list(challengers)[:5]) or 'none identified'}."
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json_from_response(text: str) -> str:
    """Extract the first JSON object or array from a Gemini response string."""
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return ""
    elif obj_start == -1:
        start, opening, closing = arr_start, "[", "]"
    elif arr_start == -1:
        start, opening, closing = obj_start, "{", "}"
    else:
        start = min(obj_start, arr_start)
        opening, closing = ("{", "}") if start == obj_start else ("[", "]")

    depth       = 0
    in_string   = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text[start:]


def _get_patent_numbers_from_bigquery(molecule_name: str) -> list:
    """Query BigQuery for distinct patent numbers for a molecule."""
    bq_sa = os.getenv("BQ_SERVICE_ACCOUNT")
    if not bq_sa:
        print("⚠️  BQ_SERVICE_ACCOUNT not set — skipping BigQuery.")
        return []
    bq_sa_path = Path(bq_sa)
    if not bq_sa_path.exists():
        print(f"⚠️  Key file not found: {bq_sa_path}")
        return []
    try:
        from google.cloud import bigquery as bq
        client = bq.Client.from_service_account_json(str(bq_sa_path))
    except Exception as e:
        print(f"⚠️  BigQuery auth failed: {e}")
        return []

    # Parameterised query — avoids SQL injection from molecule_name
    sql = f"""
        SELECT DISTINCT patent_number
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE LOWER(molecule_name) = LOWER(@molecule)
        ORDER BY patent_number
    """
    from google.cloud import bigquery as bq
    job_config = bq.QueryJobConfig(
        query_parameters=[bq.ScalarQueryParameter("molecule", "STRING", molecule_name.strip())]
    )
    print("Executing BigQuery query for distinct patent numbers...")
    try:
        rows = [row.patent_number for row in client.query(sql, job_config=job_config).result()]
        print(f"✓ Retrieved {len(rows)} distinct patent numbers from BigQuery")
        return rows
    except Exception as e:
        print(f"⚠️  BigQuery query error: {e}")
        return []


def _get_bigquery_client():
    """Return a BigQuery client, preferring service-account credentials."""
    from google.cloud import bigquery as bq
    sa_path = os.getenv("BQ_SERVICE_ACCOUNT")
    if sa_path and Path(sa_path).exists():
        return bq.Client.from_service_account_json(sa_path)
    print("Warning: BQ_SERVICE_ACCOUNT not set or file missing — using default credentials.")
    return bq.Client()


def _fetch_drugs_from_query(query: str) -> list:
    """Execute a BigQuery SQL and return the first column as a deduplicated drug list."""
    if not query or not query.strip():
        return []
    client = _get_bigquery_client()
    print("Fetching drug list from BigQuery query...")
    rows = list(client.query(query).result())

    drugs: list = []
    seen:  set  = set()
    for row in rows:
        value = row[0] if len(row) > 0 else None
        if value is None:
            continue
        name = str(value).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        drugs.append(name)

    print(f"Fetched {len(drugs)} unique drug(s) from BigQuery.")
    return drugs


def _push_to_bigquery(litigation_data: dict) -> int:
    """
    Flatten litigation_data and append rows to litigation_analysis_table.

    - Creates the table with an explicit schema if it does not exist.
    - Appends rows (WRITE_APPEND) — never truncates existing data.
    - One row per individual litigation case.  If the drug had zero cases,
      one summary-only row is written so the drug still appears in the table.

    Returns the number of rows written.
    """
    from google.cloud import bigquery as bq

    client   = _get_bigquery_client()
    now      = datetime.now(timezone.utc)
    drug     = litigation_data.get("drug_name", "")
    innovator = litigation_data.get("innovator", "")
    brand_names_str = json.dumps(litigation_data.get("brand_names", []))
    summary         = litigation_data.get("summary", "")
    analysis_date   = litigation_data.get("analysis_date")
    search_secs     = litigation_data.get("search_time_seconds")
    stats           = litigation_data.get("statistics", {})
    total_cases     = stats.get("total_cases", 0)
    unique_challengers = stats.get("unique_challengers", 0)

    litigations = litigation_data.get("litigations", [])

    # Build one row per litigation case; fall back to one summary row
    rows = []
    if litigations:
        for lit in litigations:
            rows.append({
                "drug_name":          drug,
                "innovator":          innovator,
                "brand_names":        brand_names_str,
                "patent_number":      str(lit.get("patent_number", "") or ""),
                "case_type":          str(lit.get("case_type",     "") or ""),
                "challenger":         str(lit.get("challenger",    "") or ""),
                "case_number":        str(lit.get("case_number",   "") or ""),
                "court":              str(lit.get("court",         "") or ""),
                "status":             str(lit.get("status",        "") or ""),
                "filing_date":        str(lit.get("filing_date",   "") or ""),
                "outcome":            str(lit.get("outcome",       "") or ""),
                "summary":            summary,
                "analysis_date":      analysis_date,
                "search_time_seconds": search_secs,
                "total_cases":        total_cases,
                "unique_challengers": unique_challengers,
                "loaded_at":          now.isoformat(),
            })
    else:
        rows.append({
            "drug_name":           drug,
            "innovator":           innovator,
            "brand_names":         brand_names_str,
            "patent_number":       None,
            "case_type":           None,
            "challenger":          None,
            "case_number":         None,
            "court":               None,
            "status":              None,
            "filing_date":         None,
            "outcome":             None,
            "summary":             summary,
            "analysis_date":       analysis_date,
            "search_time_seconds": search_secs,
            "total_cases":         0,
            "unique_challengers":  0,
            "loaded_at":           now.isoformat(),
        })

    df = pd.DataFrame(rows)
    df["analysis_date"] = pd.to_datetime(df["analysis_date"], errors="coerce").dt.date
    df["loaded_at"]     = pd.to_datetime(df["loaded_at"],     errors="coerce", utc=True)

    # Create table if missing (WRITE_APPEND won't auto-create it)
    try:
        client.get_table(LITIGATION_TABLE)
    except Exception:
        table = bq.Table(LITIGATION_TABLE, schema=LITIGATION_TABLE_SCHEMA)
        client.create_table(table)
        print(f"  [BQ] Created table {LITIGATION_TABLE}")

    job = client.load_table_from_dataframe(
        df,
        LITIGATION_TABLE,
        job_config=bq.LoadJobConfig(
            schema=LITIGATION_TABLE_SCHEMA,
            write_disposition=bq.WriteDisposition.WRITE_APPEND,
        ),
    )
    job.result()
    print(f"  [BQ] ✅ {len(rows)} row(s) appended to {LITIGATION_TABLE} for '{drug}'")
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _run_litigation_for_drug(drug_name: str, run_timestamp: str, output_dir: str) -> dict:
    """Worker function executed in a separate process for one drug."""
    output_path = Path(output_dir) / drug_name.strip()
    output_path.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(list_all_litigations(drug_name))

    litigations = result.get("litigations", []) if isinstance(result, dict) else []

    json_file = output_path / f"{drug_name.strip()}_litigation_analysis_{run_timestamp}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    excel_file = output_path / f"{drug_name.strip()}_litigation_analysis_{run_timestamp}.xlsx"
    pd.DataFrame(litigations).to_excel(excel_file, index=False)

    # Push to BigQuery
    rows_written = _push_to_bigquery(result)

    return {
        "drug_name":         drug_name,
        "json_file":         str(json_file),
        "excel_file":        str(excel_file),
        "total_litigations": len(litigations),
        "bq_rows_written":   rows_written,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run litigation analysis for multiple drugs in parallel."
    )
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=None,
        help="Explicit drug names. Example: --drugs semaglutide tirzepatide",
    )
    parser.add_argument(
        "--drug-query",
        default=None,
        help="Custom BigQuery SQL to fetch drugs from the first selected column.",
    )
    parser.add_argument(
        "--use-default-glp1-query",
        action="store_true",
        help="Fetch drugs using the built-in GLP-1 BigQuery query.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(cpu_count, number of drugs)).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Build drug list from all provided sources
    drugs: list = [d.strip() for d in (args.drugs or []) if d and d.strip()]

    query_drugs: list = []
    if args.use_default_glp1_query:
        query_drugs = _fetch_drugs_from_query(DEFAULT_DRUG_QUERY)
    elif args.drug_query:
        query_drugs = _fetch_drugs_from_query(args.drug_query)

    if query_drugs:
        existing_keys = {d.lower() for d in drugs}
        for drug in query_drugs:
            if drug.lower() not in existing_keys:
                drugs.append(drug)
                existing_keys.add(drug.lower())

    if not drugs:
        raise ValueError(
            "No drugs specified. Use --drugs, --use-default-glp1-query, or --drug-query."
        )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    worker_count  = max(1, min(args.max_workers or len(drugs), len(drugs), os.cpu_count() or 1))

    print("=" * 80)
    print("Parallel Litigation Analysis")
    print(f"Drugs         : {', '.join(drugs)}")
    print(f"Output folder : {output_dir}")
    print(f"Timestamp     : {run_timestamp}")
    print(f"Workers       : {worker_count}")
    print("=" * 80)

    successes: list = []
    failures:  list = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_drug = {
            executor.submit(_run_litigation_for_drug, drug, run_timestamp, str(output_dir)): drug
            for drug in drugs
        }
        total     = len(future_to_drug)
        completed = 0

        for future in as_completed(future_to_drug):
            drug = future_to_drug[future]
            completed += 1
            try:
                outcome = future.result()
                successes.append(outcome)
                print(
                    f"[{completed}/{total}] {drug}: ✅ success | "
                    f"litigations={outcome['total_litigations']} | "
                    f"bq_rows={outcome['bq_rows_written']} | "
                    f"json={outcome['json_file']} | xlsx={outcome['excel_file']}"
                )
            except Exception as exc:
                failures.append({"drug_name": drug, "error": str(exc)})
                print(f"[{completed}/{total}] {drug}: ❌ failed | {exc}")
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("Run Summary")
    print(f"  Successful : {len(successes)}")
    print(f"  Failed     : {len(failures)}")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f['drug_name']}: {f['error']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
