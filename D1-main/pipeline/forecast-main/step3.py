"""
forecast_s3.py
──────────────
Merged forecasting + phase processing pipeline.

Reads from BigQuery instead of Excel:
  - INPUT 1 (patent data):  cognito-prod-394707.cognito_prod_datamart.loe_table
  - INPUT 2 (phase data):   cognito-prod-394707.cognito_prod_datamart.clinical_efficacy
  - Drug aliases:           Derived from clinical_efficacy
  - Regulatory submissions: Fetched via Gemini Search

Outputs a SINGLE flat table (forecast_s3) with snake_case columns, suitable
for direct upload to BigQuery as:
  cognito-prod-394707.cognito_prod_datamart.forecast_s3

Output columns:
  drug_name, innovator, jurisdiction, layer, layer_reason, parent_patent,
  category, patent_number, filing_date, phase_at_filing,
  duration_from_parent, approval_date, description, insights,
  api_patent, formulation, device, method, dosing, combination,
  typical_filing_stage, filing_stage_reason

Steps:
  Step 1: IP Landscape — classifies each patent into 6 categories
  Step 2: Patent Layering — assigns layers, parents, phase-at-filing
  Step 3: Filing Timing vs. Development Milestones — calculates Phase 1/2
          from Phase 3, fetches approval dates, analyses filing timing
  Merge:  Joins Steps 1+2+3 into a single flat table

Usage:
    python forecast_s3.py               # Run all steps, save Excel
    python forecast_s3.py --upload      # Run all steps + upload to BigQuery
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run
from google import genai
from google.genai import types
from google.cloud import bigquery

load_dotenv(override=True)

# ── Make local modules importable when run as a standalone script ─────────────
_here   = Path(__file__).resolve().parent
_parent = _here.parent
_pkg    = _here.name

for _p in [str(_here), str(_parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_init = _here / "__init__.py"
if not _init.exists():
    _init.touch()

import importlib
try:
    _indexer  = importlib.import_module(f"{_pkg}.indexer")
    _analyser = importlib.import_module(f"{_pkg}.blocking_analyser")
    _exporter = importlib.import_module(f"{_pkg}.excel_exporter")
    get_or_create_collection = _indexer.get_or_create_collection
    get_all_chunks           = _analyser.get_all_chunks
    _CHROMA_AVAILABLE        = True
except Exception as _e:
    print(f"[WARNING] ChromaDB/local imports unavailable: {_e}")
    print("[WARNING] Step 2 will use Excel context only — no full patent text.")
    _CHROMA_AVAILABLE = False
    def get_or_create_collection(*a, **kw): return None
    def get_all_chunks(*a, **kw): return []


# ─────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────

_api_key      = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
_gemini       = genai.Client(api_key=_api_key)
_GEMINI_MODEL = "gemini-2.0-flash"


# ─────────────────────────────────────────────
# BigQuery client & table names
# ─────────────────────────────────────────────

def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    _creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if _creds_path and os.path.exists(_creds_path):
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(_creds_path)
    return None

_BQ_CLIENT = bigquery.Client(
    project=os.getenv("BQ_PROJECT_ID", "cognito-prod-394707"),
    credentials=_get_credentials(),
)

_LOE_TABLE           = "cognito-prod-394707.cognito_prod_datamart.loe_table"
_CLINICAL_TABLE      = "cognito-prod-394707.cognito_prod_datamart.clinical_efficacy"


# ─────────────────────────────────────────────
# Category definitions
# ─────────────────────────────────────────────

_FORECAST_COLS = ["API Patent", "Formulation", "Device", "Method", "Dosing", "Combination"]

_CATEGORY_DESCRIPTIONS = {
    "API Patent":   "Composition of Matter — the core molecule itself (the active pharmaceutical ingredient)",
    "Formulation":  "Formulation — the physical form, excipients, delivery system, or stability of the drug product",
    "Device":       "Device — the delivery device such as auto-injector, pen, or inhaler",
    "Method":       "Method of Treatment — a method of treating a disease or condition using the drug",
    "Dosing":       "Dosage Regimen — a specific dosing schedule, frequency, titration, or amount",
    "Combination":  "Combination — combining this drug with other metabolic agents (e.g. insulin, SGLT2 inhibitors, statins, other GLP-1 agonists) for therapeutic effect",
}


# ─────────────────────────────────────────────
# Load data from BigQuery (replaces load_excel)
# ─────────────────────────────────────────────

def load_loe_table() -> pd.DataFrame:
    """Loads patent data from BigQuery loe_table (replaces pipeline Excel).
    
    Renames underscore-separated BQ column names to space-separated names
    expected by the rest of the pipeline.
    """
    query = f"SELECT * FROM `{_LOE_TABLE}`"
    try:
        df = _BQ_CLIENT.query(query).to_dataframe()
        df.columns = [c.strip() for c in df.columns]
        print(f"[INPUT 1] Loaded {len(df)} row(s) from: {_LOE_TABLE}")
        print(f"[INPUT 1] Raw columns: {list(df.columns)}")

        # ── Map BQ column names (underscores) → pipeline names (spaces) ──
        _BQ_COL_MAP = {
            "Drug_Name":                                "Drug Name",
            "Patent_Number":                            "Patent Number",
            "Jurisdiction":                             "Jurisdiction",
            "Tag":                                      "Tag",
            "Blocking_Category":                        "Blocking Category",
            "Reason":                                   "Reason",
            "Step_1_Claim_Category":                    "Step 1 Claim Category",
            "Step_2_Matched_Elements":                  "Step 2 Matched Elements",
            "S2_Active_Ingredient_Form":                "S2 Active Ingredient Form",
            "S2_Formulation_Details":                   "S2 Formulation Details",
            "S2_Route_of_Administration":               "S2 Route of Administration",
            "S2_Device_Description":                    "S2 Device Description",
            "S2_Combination_Tech_Process":              "S2 Combination Tech Process",
            "Step_3_Technical_Barrier":                 "Step 3 Technical Barrier",
            "Step_3_Confidence":                        "Step 3 Confidence",
            "Step_3_Evidence_Type":                     "Step 3 Evidence Type",
            "Step_3_Evidence_Summary":                  "Step 3 Evidence Summary",
            "Step_4_Blocking_Indicator":                "Step 4 Blocking Indicator",
            "Step_4_Confidence":                        "Step 4 Confidence",
            "Step_4_Regulatory_Failure_if_Removed":     "Step 4 Regulatory Failure if Removed",
            "Step_4_Bridging_Studies_Required":          "Step 4 Bridging Studies Required",
            "Step_4_Formulation_Consistent_Across_Phases": "Step 4 Formulation Consistent Across Phases",
            "Step_4_Reason":                            "Step 4 Reason",
            "Step_5_Novel_Difficult":                   "Step 5 Novel Difficult",
            "Step_5_Novelty_Signal":                    "Step 5 Novelty Signal",
            "Step_5_First_in_Class":                    "Step 5 First in Class",
            "Step_5_Prior_Failed_Attempts":             "Step 5 Prior Failed Attempts",
            "Step_5_Complex_Implementation":            "Step 5 Complex Implementation",
            "Step_5_Confidence":                        "Step 5 Confidence",
            "Step_5_Reason":                            "Step 5 Reason",
            "Filing_Date":                              "Filing Date",
            "Grant_Date":                               "Grant Date",
            "PTE_months":                               "PTE months",
            "Pediatric_Exclusivity":                    "Pediatric Exclusivity",
            "Phase":                                    "Phase",
            "Launch_Date":                              "Launch Date",
            "Approval_Date":                            "Approval Date",
            "Approval_Date_Source":                      "Approval Date Source",
            "Est_Approval_Year":                        "Est Approval Year",
            "Exclusivity_Year":                         "Exclusivity Year",
            "Controlling_Patent_Expiry_Year":           "Controlling Patent Expiry Year",
            "Years_to_Entry":                           "Years to Entry",
            "Avg_Years_to_Entry":                       "Avg Years to Entry",
            "Score":                                    "Score",
            "Avg_Years_to_Entry_US_EP":                 "Avg Years to Entry US EP",
            "IP_Dimension_1_Score":                     "IP Dimension 1 Score",
            "Source_File":                              "Source File",
        }

        # Apply rename — only for columns that exist in the DataFrame
        rename_map = {k: v for k, v in _BQ_COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        print(f"[INPUT 1] Renamed columns: {list(df.columns)}\n")
        return df
    except Exception as e:
        print(f"[INPUT 1] Failed to load loe_table: {e}")
        return pd.DataFrame()


def load_clinical_efficacy() -> pd.DataFrame:
    """Loads phase/clinical data from BigQuery clinical_efficacy table."""
    query = f"""
        SELECT molecule_name, phase, trial_start_date
        FROM `{_CLINICAL_TABLE}`
        WHERE molecule_name IS NOT NULL
          AND phase IS NOT NULL
    """
    try:
        df = _BQ_CLIENT.query(query).to_dataframe()
        df.columns = [c.strip() for c in df.columns]
        # Rename to standard column names used throughout the pipeline
        df = df.rename(columns={
            "molecule_name":    "Molecule",
            "phase":            "Phase",
            "trial_start_date": "Start Date",
        })
        print(f"[INPUT 2] Loaded {len(df)} row(s) from: {_CLINICAL_TABLE}")
        print(f"[INPUT 2] Columns: {list(df.columns)}\n")
        return df
    except Exception as e:
        print(f"[INPUT 2] Failed to load clinical_efficacy: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Drug aliases — derived from clinical_efficacy
# ─────────────────────────────────────────────

def _build_drug_aliases(clinical_df: pd.DataFrame) -> Dict[str, str]:
    """
    Builds drug alias map from clinical_efficacy molecule names.
    Groups names that normalise to the same base (e.g. 'aleniglipron l-arginine' → 'aleniglipron').
    """
    aliases = {}
    if clinical_df.empty:
        return aliases

    names = clinical_df["Molecule"].dropna().unique().tolist()
    # Group by the first word (base molecule name)
    from collections import defaultdict
    groups = defaultdict(list)
    for name in names:
        base = name.strip().split()[0].lower()
        groups[base].append(name.strip())

    for base, variants in groups.items():
        if len(variants) > 1:
            # Use the shortest name as canonical
            canonical = min(variants, key=len)
            for v in variants:
                if v.lower() != canonical.lower():
                    aliases[v.lower()] = canonical
    return aliases


_DRUG_ALIASES: Dict[str, str] = {}


def _canonicalise_drug(name: str) -> str:
    """Resolves drug aliases."""
    lower = name.strip().lower()
    return _DRUG_ALIASES.get(lower, name.strip())


# ─────────────────────────────────────────────
# Regulatory submission dates — via Gemini Search
# ─────────────────────────────────────────────

async def _fetch_regulatory_submission_date(drug_name: str) -> Optional[str]:
    """Uses Gemini web search to find the regulatory submission (NDA/BLA/MAA) date for a drug."""
    prompt = f"""What is the earliest regulatory submission date (NDA, BLA, or MAA filing date) for the drug "{drug_name}"?

Search the web and return ONLY the date in YYYY-MM-DD format. If you cannot find an exact date, return your best estimate.
If no submission has been made, return "None".

Examples of expected output: "2016-12-05", "2021-09-15", "None"
Return ONLY the date string, nothing else.
"""
    try:
        response = await _gemini.aio.models.generate_content(
            model    = _GEMINI_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                tools       = [types.Tool(google_search=types.GoogleSearch())],
                temperature = 0.1,
            ),
        )
        raw = (response.text or "").strip().strip(".")
        if raw.lower() in ("none", "n/a", "unknown", ""):
            return None
        # Validate it looks like a date
        pd.to_datetime(raw, errors="raise")
        print(f"[SUBMISSION] '{drug_name}': {raw}")
        return raw
    except Exception as e:
        print(f"[SUBMISSION] Failed for '{drug_name}': {e}")
        return None


async def _fetch_all_submission_dates(drug_names: List[str]) -> Dict[str, str]:
    """Fetches regulatory submission dates for all marketed drugs via Gemini Search."""
    results = {}
    for drug in drug_names:
        date_str = await _fetch_regulatory_submission_date(drug)
        if date_str:
            results[drug.lower()] = date_str
    return results


# ─────────────────────────────────────────────
# Phase normalisation
# ─────────────────────────────────────────────

def _normalise_phase(raw) -> Optional[str]:
    """Normalises a phase string: 3a→Phase 3, III→Phase 3, Marketed, etc."""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None

    # Replace pipe characters used as roman numerals
    s = re.sub(r'\|\|\|', 'III', s)
    s = re.sub(r'\|\|',   'II',  s)
    s = re.sub(r'\|',     'I',   s)

    low = s.lower()

    if low in ("marketed", "launched", "approved", "registered"):
        return "Marketed"
    if low in ("pre-registration", "preregistration", "pre registration", "nda/bla", "nda", "bla"):
        return "Pre-registration"
    if low in ("preclinical", "pre-clinical", "discovery"):
        return "Preclinical"

    cleaned = re.sub(r"^phase[\s\-]*", "", low).strip()
    _roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    roman_match = re.match(r"^(i{1,3}v?)\s*[a-z]?\s*$", cleaned, re.IGNORECASE)
    if roman_match:
        num = _roman.get(roman_match.group(1).lower())
        if num:
            return f"Phase {num}"

    num_match = re.match(r"^(\d)", cleaned)
    if num_match:
        return f"Phase {num_match.group(1)}"

    return None


def _parse_date_or_year(val):
    """Parses a date or year-only value (2008 → Jan 1 2008)."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return pd.NaT
    try:
        year = int(float(s))
        if 1900 <= year <= 2100:
            return pd.Timestamp(year=year, month=1, day=1)
    except (ValueError, TypeError):
        pass
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


# ─────────────────────────────────────────────
# Approval date fetcher
# ─────────────────────────────────────────────

_APPROVAL_FETCH_AVAILABLE = False
_fetch_approval_for_molecules = None
try:
    _pkg_name_phase = Path(__file__).resolve().parent.name
    _fetcher_mod = importlib.import_module(f"{_pkg_name_phase}.approval_date_fetcher")
    _raw_fetch = _fetcher_mod.fetch_approval_dates

    async def _fetch_approval_for_molecules(molecules: list) -> dict:
        results = {}
        for mol in molecules:
            print(f"  [APPROVAL] Fetching for '{mol}'...")
            try:
                approval = await _raw_fetch(drug_name=mol, bq_companies=[], bq_brands=[], fetch_us=True, fetch_eu=True)
                results[mol] = {"US": approval.get("US", {}).get("date"), "EU": approval.get("EU", {}).get("date")}
            except Exception as e:
                print(f"  [APPROVAL] Failed for '{mol}': {e}")
                results[mol] = {"US": None, "EU": None}
        return results

    _APPROVAL_FETCH_AVAILABLE = True
except Exception:
    pass


# ─────────────────────────────────────────────
# GCS drug listing for filtering
# ─────────────────────────────────────────────

_GCS_DRUG_LIST_AVAILABLE = False
_list_gcs_drug_names_fn = None
try:
    _gcs_mod = importlib.import_module(f"{_pkg_name_phase}.gcs_lister")
    _gcs_client_fn     = _gcs_mod.get_gcs_client
    _gcs_bucket        = _gcs_mod.GCS_BUCKET_NAME
    _gcs_prefix        = _gcs_mod.GCS_PATENTS_PREFIX

    def _list_gcs_drug_names_fn() -> list:
        if not _gcs_bucket:
            return []
        try:
            client = _gcs_client_fn()
            prefix = _gcs_prefix.rstrip("/") + "/"
            all_blobs = list(client.list_blobs(_gcs_bucket, prefix=prefix))
            prefix_depth = len(prefix.split("/")) - 1
            folders = set()
            for blob in all_blobs:
                parts = blob.name.split("/")
                if len(parts) > prefix_depth + 1:
                    f = parts[prefix_depth].strip()
                    if f:
                        folders.add(f)
            return sorted(folders)
        except Exception as e:
            print(f"[GCS] Failed: {e}")
            return []

    _GCS_DRUG_LIST_AVAILABLE = True
except Exception:
    pass


# ─────────────────────────────────────────────
# Phase processing (merged from phase_processor.py)
# ─────────────────────────────────────────────

def _process_phase_input(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes phase data from clinical_efficacy BigQuery table.

    Steps:
      1. Normalise phases (3a→3, III→3)
      2. Keep earliest date per molecule per phase
      3. Drop Phase 4 (handled via Marketed)
      4. Use real Phase 1 / Phase 2 trial_start_date when present;
         only derive missing ones from Phase 3
         (Phase 2 = P3 − 3 yrs, Phase 1 = P3 − 4 yrs)
      5. Fetch approval dates for marketed drugs only
      6. Filter to GCS drugs only

    Expects columns: Molecule, Phase, Start Date
    """
    from dateutil.relativedelta import relativedelta

    if clinical_df.empty:
        print("[PHASE] No clinical data to process.")
        return pd.DataFrame()

    df = clinical_df.copy()
    mol_col   = "Molecule"
    phase_col = "Phase"
    date_col  = "Start Date"

    print(f"[PHASE] Processing {len(df)} rows from clinical_efficacy")

    # Normalise phases
    df["Normalised Phase"] = df[phase_col].apply(_normalise_phase)
    df = df.dropna(subset=["Normalised Phase"])

    # Parse dates/years
    df["_parsed_date"] = df[date_col].apply(_parse_date_or_year)

    # Earliest date per molecule per phase
    df = df.sort_values("_parsed_date", na_position="last")
    result = df.groupby([mol_col, "Normalised Phase"], sort=False).first().reset_index()
    result = result.drop(columns=["_parsed_date"])

    # Drop Phase 4 (handled separately via Marketed). Keep Phase 1 and Phase 2
    # rows if clinical_efficacy provides their own trial_start_date — those
    # are real observed values and should be preferred over derived ones.
    result = result[result["Normalised Phase"] != "Phase 4"]

    # Track which (molecule, phase) combinations already have a REAL start
    # date from clinical_efficacy. We will only synthesise Phase 1 or Phase 2
    # rows for drugs where the real value is missing.
    real_phases_by_mol: Dict[str, set] = {}
    for _, row in result.iterrows():
        m = str(row[mol_col]).strip()
        p = str(row["Normalised Phase"]).strip()
        if p in ("Phase 1", "Phase 2"):
            real_phases_by_mol.setdefault(m, set()).add(p)

    # Calculate Phase 1/2 from Phase 3 ONLY when the real value is missing
    # for that drug. Phase 2 = Phase 3 − 3 years, Phase 1 = Phase 3 − 4 years.
    phase3_rows = result[result["Normalised Phase"] == "Phase 3"].copy()
    new_rows = []
    derived_count = 0
    skipped_count = 0
    for _, row in phase3_rows.iterrows():
        p3_date = _parse_date_or_year(row[date_col])
        if pd.isna(p3_date):
            continue
        mol_name = row[mol_col]
        is_year = False
        try:
            y = int(float(str(row[date_col]).strip()))
            is_year = 1900 <= y <= 2100
        except (ValueError, TypeError):
            pass

        p2_date = p3_date - relativedelta(years=3)
        p1_date = p3_date - relativedelta(years=4)

        have_real = real_phases_by_mol.get(str(mol_name).strip(), set())

        if "Phase 2" not in have_real:
            p2_row = row.copy()
            p2_row["Normalised Phase"] = "Phase 2"
            p2_row[date_col] = p2_date.year if is_year else p2_date
            p2_row[phase_col] = "Phase 2 (calculated)"
            new_rows.append(p2_row)
            derived_count += 1
        else:
            skipped_count += 1

        if "Phase 1" not in have_real:
            p1_row = row.copy()
            p1_row["Normalised Phase"] = "Phase 1"
            p1_row[date_col] = p1_date.year if is_year else p1_date
            p1_row[phase_col] = "Phase 1 (calculated)"
            new_rows.append(p1_row)
            derived_count += 1
        else:
            skipped_count += 1

        if is_year:
            print(
                f"  [PHASE] {mol_name}: P3={p3_date.year} "
                f"→ P2={p2_date.year}{' (real exists, skipped)' if 'Phase 2' in have_real else ''} "
                f"→ P1={p1_date.year}{' (real exists, skipped)' if 'Phase 1' in have_real else ''}"
            )
        else:
            print(
                f"  [PHASE] {mol_name}: P3={p3_date.strftime('%d-%m-%Y')} "
                f"→ P2={p2_date.strftime('%d-%m-%Y')}{' (real exists, skipped)' if 'Phase 2' in have_real else ''} "
                f"→ P1={p1_date.strftime('%d-%m-%Y')}{' (real exists, skipped)' if 'Phase 1' in have_real else ''}"
            )

    if new_rows:
        result = pd.concat([result, pd.DataFrame(new_rows)], ignore_index=True)
        print(
            f"[PHASE] Added {derived_count} calculated Phase 1/2 row(s); "
            f"kept {skipped_count} real Phase 1/2 date(s) from clinical_efficacy"
        )
    elif skipped_count:
        print(
            f"[PHASE] Used {skipped_count} real Phase 1/2 date(s) from clinical_efficacy "
            f"— no derivation needed"
        )

    # Fetch approval dates for marketed drugs only
    if _APPROVAL_FETCH_AVAILABLE:
        all_marketed = set()
        for _, row in df.iterrows():
            if _normalise_phase(str(row.get(phase_col, ""))) == "Marketed":
                mol = str(row.get(mol_col, "")).strip()
                if mol and mol.lower() not in ("nan", "none", ""):
                    all_marketed.add(mol)
        for mol in result[result["Normalised Phase"] == "Marketed"][mol_col].unique():
            all_marketed.add(mol)

        if all_marketed:
            print(f"[PHASE] Fetching approval dates for {len(all_marketed)} marketed drug(s)...")

            def _run_sync(coro):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError
                    return loop.run_until_complete(coro)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(coro)

            approval_map = _run_sync(_fetch_approval_for_molecules(sorted(all_marketed)))
            result["Approval Date"] = ""

            marketed_rows = []
            for mol_name, dates in approval_map.items():
                parsed_dates = []
                for geo in ("US", "EU"):
                    raw = dates.get(geo)
                    if raw:
                        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
                        if pd.notna(parsed):
                            parsed_dates.append(parsed)
                if not parsed_dates:
                    continue
                earliest = min(parsed_dates)
                existing = result[(result[mol_col] == mol_name) & (result["Normalised Phase"] == "Marketed")]
                if not existing.empty:
                    result.loc[existing.index, "Approval Date"] = earliest
                else:
                    marketed_row = {col: "" for col in result.columns}
                    marketed_row[mol_col] = mol_name
                    marketed_row["Normalised Phase"] = "Marketed"
                    marketed_row[phase_col] = "Marketed"
                    marketed_row[date_col] = earliest
                    marketed_row["Approval Date"] = earliest
                    marketed_rows.append(marketed_row)

            if marketed_rows:
                result = pd.concat([result, pd.DataFrame(marketed_rows)], ignore_index=True)
        else:
            result["Approval Date"] = ""
    else:
        result["Approval Date"] = ""

    # Filter to GCS drugs
    if _GCS_DRUG_LIST_AVAILABLE and _list_gcs_drug_names_fn:
        gcs_drugs = _list_gcs_drug_names_fn()
        if gcs_drugs:
            gcs_norm = {d.strip().lower().replace(" ", "").replace("-", "").replace("_", "") for d in gcs_drugs}
            def _in_gcs(mol):
                canonical = _canonicalise_drug(str(mol))
                norm = canonical.lower().replace(" ", "").replace("-", "").replace("_", "")
                if norm in gcs_norm:
                    return True
                return str(mol).strip().lower().replace(" ", "").replace("-", "").replace("_", "") in gcs_norm
            before = len(result)
            result = result[result[mol_col].apply(_in_gcs)].reset_index(drop=True)
            print(f"[PHASE] GCS filter: {before} → {len(result)} rows ({result[mol_col].nunique()} drugs)")

    # Sort
    _phase_order = {"Preclinical": 0, "Phase 1": 1, "Phase 2": 2, "Phase 3": 3, "Pre-registration": 4, "Marketed": 5}
    result["_sort"] = result["Normalised Phase"].map(_phase_order).fillna(99)
    result = result.sort_values([mol_col, "_sort"]).drop(columns=["_sort"]).reset_index(drop=True)

    print(f"[PHASE] Result: {len(result)} rows ({result[mol_col].nunique()} drugs)")
    return result


def _build_phase_timelines(phase_df: pd.DataFrame) -> Dict[str, List]:
    """Builds per-drug timelines from the processed phase DataFrame."""
    timelines: Dict[str, List] = {}

    if phase_df.empty:
        return timelines

    mol_col  = "Molecule"
    date_col = "Start Date"
    approval_col = "Approval Date" if "Approval Date" in phase_df.columns else None

    _VALID_PHASES = {"Phase 1", "Phase 2", "Phase 3", "After Submission", "Marketed"}

    for _, row in phase_df.iterrows():
        mol   = str(row.get(mol_col, "")).strip()
        phase = str(row.get("Normalised Phase", "")).strip()
        raw_date = row.get(date_col)

        if not mol or mol.lower() in ("nan", "none", ""):
            continue
        if phase not in _VALID_PHASES:
            continue

        if phase == "Marketed" and approval_col:
            approval_raw = row.get(approval_col)
            parsed = pd.to_datetime(approval_raw, errors="coerce", dayfirst=True)
            if pd.notna(parsed):
                raw_date = parsed

        parsed = _parse_date_or_year(raw_date)
        if pd.isna(parsed):
            continue

        key = mol.lower()
        if key not in timelines:
            timelines[key] = []
        timelines[key].append((parsed, phase))

    # Sort each timeline
    for key in timelines:
        timelines[key].sort(key=lambda x: x[0])

    print(f"[PHASE] Built timelines for {len(timelines)} drug(s)")
    return timelines


def _resolve_phase_at_filing(
    timelines: Dict[str, List],
    drug_name: str,
    filing_date: str,
) -> str:
    """
    Given a drug's phase timeline and a patent filing date, returns the
    phase the drug was in at the time of filing.
    """
    key = drug_name.strip().lower()
    timeline = timelines.get(key, [])

    if not timeline:
        return "—"

    try:
        filing_dt = pd.to_datetime(filing_date, errors="coerce")
    except Exception:
        return "—"

    if pd.isna(filing_dt):
        return "—"

    result_phase = None
    for milestone_date, phase_label in timeline:
        if milestone_date <= filing_dt:
            result_phase = phase_label
        else:
            break

    if result_phase is None:
        return "Pre-Phase 1"

    return result_phase


# ─────────────────────────────────────────────
# LLM classification — per patent row (Step 1)
# ─────────────────────────────────────────────

_COMBO_KEYWORDS = (
    "combination", "co-administer", "co-therapy", "combined with",
    "together with", "adjunct", "metabolic agent", "dual therapy",
)


_LOE_CATEGORY_TO_FORECAST = {
    "composition of matter":  "API Patent",
    "salt/polymorph":         "API Patent",
    "formulation":            "Formulation",
    "manufacturing process":  "Formulation",
    "method of treatment":    "Method",
    "device":                 "Device",
    "dosage regimen":         "Dosing",
}


def _classify_patent_from_loe(patent_info: Dict) -> Dict[str, str]:
    """
    Deterministic Step 1 classification using the existing
    `Step 1 Claim Category` field from loe_table (produced by the
    upstream pipeline). No LLM call required.

    Returns a dict like {"API Patent": "✓ (Composition of Matter)", ...}
    where keys are a subset of _FORECAST_COLS.

    Mapping (loe -> forecast bucket):
      Composition of Matter  -> API Patent
      Salt/Polymorph         -> API Patent
      Formulation            -> Formulation
      Manufacturing Process  -> Formulation
      Method of Treatment    -> Method
      Device                 -> Device
      Dosage Regimen         -> Dosing

    Combination is not a loe category — it is inferred from the
    Blocking Category / Reason text only when an explicit co-therapy
    keyword is present (matches the prior rule in _classify_patent).
    """
    result: Dict[str, str] = {}

    raw_cat = str(patent_info.get("Step 1 Claim Category", "")).strip()
    if raw_cat and raw_cat.lower() not in ("n/a", "nan", "none", ""):
        bucket = _LOE_CATEGORY_TO_FORECAST.get(raw_cat.lower())
        if bucket:
            result[bucket] = f"✓ ({raw_cat})"

    # Combination: only mark when an explicit co-therapy signal appears
    # in Blocking Category or Reason. Mirrors the prior LLM rule.
    combo_text = " ".join(
        str(patent_info.get(k, "")) for k in ("Blocking Category", "Reason")
    ).lower()
    if any(kw in combo_text for kw in _COMBO_KEYWORDS):
        result["Combination"] = "✓ (co-administration with another agent)"

    patent_num = patent_info.get("Patent Number", "")
    print(f"[STEP 1]   {patent_num} → {list(result.keys()) or 'no match'} "
          f"(loe category: {raw_cat or '—'})")
    return result


async def _classify_patent(drug_name: str, patent_info: Dict) -> Dict[str, str]:
    """
    DEPRECATED: previously sent each patent to Gemini for 6-category
    classification. Now delegates to the deterministic mapping based on
    the `Step 1 Claim Category` already present in loe_table.

    Kept as an async wrapper so existing call sites continue to work.
    The `drug_name` argument is unused but retained for signature
    compatibility.
    """
    del drug_name  # unused
    return _classify_patent_from_loe(patent_info)


async def _lookup_innovator(drug_name: str) -> str:
    """Uses Gemini web search to find the innovator/originator company for a drug."""
    prompt = f"""Who is the original innovator / originator pharmaceutical company that developed and holds the primary patents for the drug "{drug_name}"?

Search the web and return ONLY the company name. No explanation, no extra text. Just the company name.
Examples: "Novo Nordisk", "Eli Lilly", "AstraZeneca"
"""
    try:
        response = await _gemini.aio.models.generate_content(
            model    = _GEMINI_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                tools       = [types.Tool(google_search=types.GoogleSearch())],
                temperature = 0.1,
            ),
        )
        name = (response.text or "").strip().strip(".")
        print(f"[STEP 1] Innovator for '{drug_name}': {name}")
        return name or "Unknown"
    except Exception as e:
        print(f"[STEP 1] Innovator lookup failed for '{drug_name}': {e}")
        return "Unknown"


async def step1_ip_landscape() -> pd.DataFrame:
    """
    Step 1: IP Landscape of All Drugs.

    Reads from BigQuery loe_table. For each drug:
      1. Filters patents belonging to that drug.
      2. Sends each patent's analysis info to Gemini for 6-category classification.
      3. Aggregates into one row per drug.
    """
    df = load_loe_table()
    if df.empty:
        return pd.DataFrame()

    all_drugs = sorted(df["Drug Name"].dropna().unique().tolist())
    print(f"[STEP 1] Drugs found: {all_drugs}\n")

    if not all_drugs:
        print("[STEP 1] No drugs found.")
        return pd.DataFrame()

    _CONTEXT_COLS = [
        "Patent Number", "Jurisdiction", "Tag",
        "Blocking Category", "Step 1 Claim Category", "Reason",
        "Step 2 Matched Elements", "Step 3 Technical Barrier",
        "Step 3 Evidence Summary", "Step 5 Novelty Signal", "Step 5 Reason",
    ]
    context_cols = [c for c in _CONTEXT_COLS if c in df.columns]

    results = []

    for drug_name in all_drugs:
        drug_df = df[df["Drug Name"] == drug_name]
        print(f"[STEP 1] ── {drug_name} — {len(drug_df)} patent row(s) ──")

        innovator = await _lookup_innovator(drug_name)

        category_entries: Dict[str, List[str]] = {col: [] for col in _FORECAST_COLS}

        for _, row in drug_df.iterrows():
            patent_num = str(row.get("Patent Number", "")).strip()
            if not patent_num or patent_num in ("N/A", "nan", "None", ""):
                continue

            patent_info = {
                col: str(row.get(col, "")).strip()
                for col in context_cols
            }

            classification = await _classify_patent(drug_name, patent_info)

            for col, label in classification.items():
                if label not in category_entries[col]:
                    category_entries[col].append(label)

        row_out = {"Drug Name": drug_name, "Innovator": innovator}
        for col in _FORECAST_COLS:
            entries      = category_entries[col]
            row_out[col] = "\n".join(entries) if entries else "—"

        results.append(row_out)
        print(
            f"[STEP 1] {drug_name} summary: "
            + " | ".join(
                f"{col}={len(category_entries[col])}"
                for col in _FORECAST_COLS
            ) + "\n"
        )

    result_df = pd.DataFrame(results, columns=["Drug Name", "Innovator"] + _FORECAST_COLS)
    print(f"[STEP 1] Complete — {len(result_df)} drug(s).")
    return result_df


# ─────────────────────────────────────────────
# Step 2: Patent Layering Pattern (with ChromaDB + branching)
# ─────────────────────────────────────────────


def _get_patent_text_from_chroma(drug_name: str, patent_filename: str) -> str:
    """Fetches full patent text from ChromaDB for a given drug + filename."""
    try:
        collection = get_or_create_collection(drug_name)
        chunks     = get_all_chunks(collection, patent_filename)
        return "\n\n".join(chunks) if chunks else ""
    except Exception as e:
        print(f"[STEP 2] ChromaDB fetch failed for {patent_filename}: {e}")
        return ""


async def _analyse_patent_full(
    drug_name:      str,
    patent_num:     str,
    patent_filename: str,
    excel_context:  Dict,
    known_patents:  List[Dict],
) -> Dict:
    """
    Sends full patent text (from ChromaDB) + context to Gemini.
    Returns: layer, category, description, parent_patent_number.
    """
    patent_text = _get_patent_text_from_chroma(drug_name, patent_filename)

    if known_patents:
        known_block = "\n".join(
            f"  - {p['Patent Number']} (Layer {p['Layer']}, {p['Category']}): {p['Description']}"
            for p in known_patents
        )
    else:
        known_block = "  None yet — this may be the first patent (Layer 1 CoM)."

    excel_block = "\n".join(
        f"  {k}: {v}"
        for k, v in excel_context.items()
        if v and str(v).strip() not in ("N/A", "nan", "None", "")
    )

    patent_section = f"\nFULL PATENT TEXT:\n{patent_text[:35000]}" if patent_text else "\n(No full text available — use context only)"

    prompt = f"""You are a pharmaceutical patent analyst building a patent protection tree for {drug_name}.

PATENT TO ANALYSE:
  Patent Number: {patent_num}

ANALYSIS CONTEXT:
{excel_block}
{patent_section}

PATENTS ALREADY IDENTIFIED FOR THIS DRUG (for branching):
{known_block}

YOUR TASK:
Determine the following for patent {patent_num}:

1. LAYER — which protection layer does this patent belong to?
   Layer 1 = Composition of Matter ONLY (the core API molecule — ONLY ONE patent per drug per jurisdiction can be Layer 1. This is strictly the patent that claims the novel chemical compound itself. Salt forms, polymorphs, formulations, and all other patent types are NOT Layer 1.)
   Layer 2 = Salt/Polymorph OR Formulation (directly derived from the API — these protect how the molecule is physically formed or delivered)
   Layer 3 = Device, Method of Treatment, OR Dosage Regimen (built on top of Layer 2)
   Layer 4 = Combination therapy (combining with other metabolic agents)

   CRITICAL RULE: If the Step 1 Claim Category in the context is NOT "Composition of Matter", this patent CANNOT be Layer 1. Only a patent whose claim category is "Composition of Matter" may be assigned Layer 1.

2. DESCRIPTION — one short phrase (max 10 words) describing what this patent protects.

3. PARENT_PATENT — which already-identified patent does this one most directly build upon?
   - For Layer 1 (CoM): write "None" — it is the root patent.
   - For Layer 2 (Salt/Polymorph/Formulation): the parent MUST be the Layer 1 Composition of Matter patent.
   - For Layer 3 (Device/Method/Dosing): choose the most relevant Layer 2 patent from the known list.
   - For Layer 4 (Combination): choose the most relevant parent from the known list.
   - If no close parent exists, write the closest one or "None".

4. LAYER_REASON — explain in 1-2 sentences: (a) why this patent belongs to the assigned layer, and (b) why the chosen parent patent is the correct parent.

Respond ONLY with valid JSON, no markdown:
{{
  "layer":          <1|2|3|4>,
  "description":    "<short phrase>",
  "parent_patent":  "<patent number or None>",
  "layer_reason":   "<1-2 sentences>"
}}
"""

    try:
        response = await _gemini.aio.models.generate_content(
            model    = _GEMINI_MODEL,
            contents = prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        layer        = int(parsed.get("layer", 99))
        description  = str(parsed.get("description", "—")).strip().strip(".")
        parent       = str(parsed.get("parent_patent", "None")).strip()
        layer_reason = str(parsed.get("layer_reason", "—")).strip()

        # Enforce: only Composition of Matter can be Layer 1
        claim_cat = excel_context.get("Step 1 Claim Category", "").strip().lower()
        is_com = "composition of matter" in claim_cat or "com " in claim_cat or claim_cat == "com"

        if layer == 1 and not is_com:
            layer = 2
            layer_reason = f"[Corrected from Layer 1 → Layer 2: claim category is '{excel_context.get('Step 1 Claim Category', '')}', not Composition of Matter] " + layer_reason

        # Enforce: Formulation/Salt/Polymorph parent must be CoM patent
        is_formulation = any(
            kw in claim_cat
            for kw in ("formulation", "salt", "polymorph")
        )
        if is_formulation and layer == 2:
            com_parent = next(
                (p["Patent Number"] for p in known_patents if p.get("Layer") == 1),
                None,
            )
            if com_parent and parent != com_parent:
                parent = com_parent
                layer_reason = layer_reason.rstrip() + f" [Parent corrected to CoM patent {com_parent}]"

        return {
            "Layer":          layer,
            "Description":    description,
            "Parent Patent":  parent,
            "Layer Reason":   layer_reason,
        }
    except Exception as e:
        print(f"[STEP 2] LLM failed for {patent_num}: {e}")
        return {
            "Layer":         99,
            "Description":   "—",
            "Parent Patent": "None",
            "Layer Reason":  "—",
        }


async def step2_patent_layering() -> pd.DataFrame:
    """
    Step 2: Patent Layering Pattern with ChromaDB + LLM branching.

    Reads from BigQuery loe_table. For each drug + jurisdiction:
      1. Reads each patent's full text from ChromaDB.
      2. Sends full text + context to Gemini.
      3. LLM assigns: layer, category, description, parent patent.
      4. Builds the branching tree sorted by layer then filing date.

    Returns a SINGLE concatenated DataFrame (forecast_s3) with all drugs,
    with columns:
        Drug Name | Jurisdiction | Layer | Layer Reason | Parent Patent |
        Category | Patent Number | Filing Date | Phase at Filing |
        Duration from Parent | Approval Date | Description | Insights
    """
    df = load_loe_table()
    if df.empty:
        return pd.DataFrame()

    all_drugs = sorted(df["Drug Name"].dropna().unique().tolist())
    print(f"[STEP 2] Drugs: {all_drugs}\n")

    if not all_drugs:
        print("[STEP 2] No drugs found.")
        return pd.DataFrame()

    # Load phase timeline via Step 3 (owns phase processing)
    phase_timelines = await step3_build_phase_timelines(all_drugs)
    if phase_timelines:
        print(f"[STEP 2] Phase timeline loaded from Step 3 — will resolve Phase at Filing per patent")
    else:
        print(f"[STEP 2] No phase timeline available — 'Phase at Filing' will be empty")

    _CONTEXT_COLS = [
        "Patent Number", "Jurisdiction", "Filing Date", "Source File",
        "Step 1 Claim Category", "Reason", "Blocking Category",
        "Step 3 Evidence Summary", "Step 5 Reason",
    ]
    context_cols = [c for c in _CONTEXT_COLS if c in df.columns]

    all_drug_rows: List[Dict] = []   # single list for all drugs

    for drug_name in all_drugs:
        drug_df   = df[df["Drug Name"] == drug_name].copy()
        canonical = drug_name.strip().lower().replace(" ", "_")

        # Extract approval date per jurisdiction
        approval_dates: Dict[str, str] = {}
        for jur in ["US", "EP"]:
            jur_rows = drug_df[
                drug_df["Jurisdiction"].astype(str).str.upper().str.strip() == jur
            ]
            if "Approval Date" in jur_rows.columns:
                dates = jur_rows["Approval Date"].dropna().astype(str).str.strip()
                dates = dates[~dates.isin(["N/A", "nan", "None", "Unknown", "NaT", ""])]
                if not dates.empty:
                    raw = dates.iloc[0]
                    if " 00:00:00" in raw:
                        raw = raw.replace(" 00:00:00", "")
                    approval_dates[jur] = raw
            if jur in approval_dates:
                print(f"[STEP 2] {drug_name} | {jur} Approval Date: {approval_dates[jur]}")

        print(f"\n[STEP 2] ══ {drug_name} — {len(drug_df)} patent row(s) ══")

        drug_rows: List[Dict] = []

        for jurisdiction in ["US", "EP"]:
            jur_df = drug_df[
                drug_df["Jurisdiction"].astype(str).str.upper().str.strip() == jurisdiction
            ]
            print(f"\n[STEP 2]   [{jurisdiction}] — {len(jur_df)} patent(s)")

            if jur_df.empty:
                continue

            jur_df = jur_df.copy()
            jur_df["_date_sort"] = pd.to_datetime(
                jur_df["Filing Date"].astype(str), errors="coerce"
            )
            jur_df = jur_df.sort_values("_date_sort", na_position="last")

            known_patents: List[Dict] = []

            for _, row in jur_df.iterrows():
                patent_num = str(row.get("Patent Number", "")).strip()
                if not patent_num or patent_num in ("N/A", "nan", "None", ""):
                    continue

                filing_date = str(row.get("Filing Date", "")).strip()
                if " 00:00:00" in filing_date:
                    filing_date = filing_date.replace(" 00:00:00", "")
                if filing_date in ("nan", "None", "Unknown", "NaT", ""):
                    filing_date = "Unknown"

                source_file = str(row.get("Source File", "")).strip()

                row_context = {c: str(row.get(c, "")).strip() for c in context_cols}

                print(f"[STEP 2]     Analysing {patent_num} ({filing_date})...")
                result = await _analyse_patent_full(
                    drug_name        = drug_name,
                    patent_num       = patent_num,
                    patent_filename  = source_file,
                    excel_context    = row_context,
                    known_patents    = known_patents,
                )

                claim_category = str(row.get("Step 1 Claim Category", "")).strip()
                if not claim_category or claim_category in ("N/A", "nan", "None", ""):
                    claim_category = "Unknown"

                row_out = {
                    "Drug Name":        drug_name,
                    "Jurisdiction":     jurisdiction,
                    "Layer":            result["Layer"],
                    "Layer Reason":     result["Layer Reason"],
                    "Parent Patent":    result["Parent Patent"],
                    "Category":         claim_category,
                    "Patent Number":    patent_num,
                    "Filing Date":      filing_date,
                    "Phase at Filing":  _resolve_phase_at_filing(phase_timelines, drug_name, filing_date),
                    "Approval Date":    approval_dates.get(jurisdiction, "—"),
                    "Description":      result["Description"],
                }
                drug_rows.append(row_out)
                known_patents.append({
                    "Patent Number": patent_num,
                    "Layer":         result["Layer"],
                    "Category":      claim_category,
                    "Description":   result["Description"],
                })
                print(
                    f"[STEP 2]     → Layer {result['Layer']} | {claim_category} "
                    f"| Parent: {result['Parent Patent']} | {result['Description']}"
                    f" | Phase: {row_out['Phase at Filing']}"
                )

        if not drug_rows:
            print(f"[STEP 2] No rows for '{drug_name}'")
            continue

        # Build temporary per-drug DF for duration calculation and insights
        drug_result_df = pd.DataFrame(drug_rows)

        # Sort: jurisdiction, then layer, then filing date
        drug_result_df["_date_sort"] = pd.to_datetime(
            drug_result_df["Filing Date"], errors="coerce"
        )
        drug_result_df = drug_result_df.sort_values(
            by=["Jurisdiction", "Layer", "_date_sort"], na_position="last"
        ).reset_index(drop=True)

        # Duration: time between parent filing date and this patent
        date_lookup: Dict[str, pd.Timestamp] = {}
        for _, r in drug_result_df.iterrows():
            parsed = pd.to_datetime(r["Filing Date"], errors="coerce")
            if pd.notna(parsed):
                date_lookup[r["Patent Number"]] = parsed

        durations = []
        for _, r in drug_result_df.iterrows():
            parent = str(r["Parent Patent"]).strip()
            if parent in ("None", "—", "", "nan") or parent not in date_lookup:
                durations.append("—")
                continue
            current_dt = date_lookup.get(r["Patent Number"])
            parent_dt  = date_lookup.get(parent)
            if current_dt is None or parent_dt is None:
                durations.append("—")
                continue
            delta_days   = (current_dt - parent_dt).days
            delta_months = round(delta_days / 30.44)
            delta_years  = delta_days / 365.25
            if abs(delta_years) >= 1:
                durations.append(f"{delta_years:+.1f} yrs")
            else:
                durations.append(f"{delta_months:+d} months")

        drug_result_df.insert(
            drug_result_df.columns.get_loc("Phase at Filing") + 1,
            "Duration from Parent",
            durations,
        )
        drug_result_df = drug_result_df.drop(columns=["_date_sort"])

        # Insights: LLM summary for this drug
        print(f"[STEP 2] Generating insights for '{drug_name}'...")
        insights = await _generate_insights(drug_name, drug_result_df)
        drug_result_df["Insights"] = ""
        if not drug_result_df.empty:
            drug_result_df.at[drug_result_df.index[0], "Insights"] = insights

        all_drug_rows.append(drug_result_df)
        print(f"\n[STEP 2] {drug_name}: {len(drug_result_df)} patent(s) layered.")

    if not all_drug_rows:
        print("[STEP 2] No data produced.")
        return pd.DataFrame()

    # Concatenate all drugs into a single table
    forecast_s3 = pd.concat(all_drug_rows, ignore_index=True)
    print(f"\n[STEP 2] Complete — forecast_s3 has {len(forecast_s3)} rows across {forecast_s3['Drug Name'].nunique()} drug(s).")
    return forecast_s3


# ─────────────────────────────────────────────
# Step 3: Filing Timing vs. Development Milestones
# ─────────────────────────────────────────────
# Purpose: Analyse when innovators file each patent type relative to
# clinical development phases (Phase 1 → Phase 2 → Phase 3 → Marketed).
#
# This step OWNS the phase timeline construction:
#   1. Loads clinical trial data from BigQuery (clinical_efficacy)
#   2. Normalises phases (3a→3, III→3, etc.)
#   3. Drops Phase 4 (handled via Marketed). Keeps real Phase 1 / Phase 2
#      trial_start_date from clinical_efficacy when present, and only
#      derives missing ones from Phase 3 (Phase 2 = P3 − 3 yrs, Phase 1 = P3 − 4 yrs)
#   4. Fetches live approval dates for marketed drugs (approval_date_fetcher)
#   5. Fetches regulatory submission dates via Gemini Search
#   6. Filters to GCS drugs
#   7. Builds per-drug timelines and resolves "Phase at Filing" per patent
#   8. LLM analysis: determines typical filing stage per patent type

async def step3_build_phase_timelines(drug_names: List[str]) -> Dict[str, List]:
    """
    Step 3 core: Builds phase timelines from clinical_efficacy BigQuery table.

    Pipeline:
      1. Load clinical_efficacy from BigQuery
      2. Normalise phases. Keep real Phase 1/2 trial_start_date when
         present; only calculate from Phase 3 (P2=P3−3y, P1=P3−4y) when missing.
      3. Fetch approval dates for marketed drugs
      4. Fetch regulatory submission dates via Gemini Search
      5. Filter to GCS drugs
      6. Build per-drug (date, phase) timelines

    Returns: {drug_name_lower: [(date, phase_label), ...]}
    """
    clinical_df = load_clinical_efficacy()
    if clinical_df.empty:
        print("[STEP 3] No clinical data loaded from clinical_efficacy.")
        return {}

    # Build drug aliases from clinical data
    global _DRUG_ALIASES
    _DRUG_ALIASES = _build_drug_aliases(clinical_df)
    if _DRUG_ALIASES:
        print(f"[STEP 3 ALIASES] Built {len(_DRUG_ALIASES)} drug alias(es): {_DRUG_ALIASES}")

    # Process phase input: normalise, calculate Phase 1/2, fetch approvals, GCS filter
    phase_df = _process_phase_input(clinical_df)
    if phase_df.empty:
        return {}

    # Build base timelines
    timelines = _build_phase_timelines(phase_df)

    # Fetch regulatory submission dates via Gemini Search for known drugs
    marketed_drugs = [d for d in drug_names if d.lower() in timelines]
    if marketed_drugs:
        print(f"[STEP 3 SUBMISSION] Fetching regulatory submission dates for: {marketed_drugs}")
        submission_dates = await _fetch_all_submission_dates(marketed_drugs)

        for drug, date_str in submission_dates.items():
            parsed = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(parsed):
                continue
            key = drug.lower()
            if key not in timelines:
                timelines[key] = []
            existing_phases = {p for _, p in timelines[key]}
            if "After Submission" not in existing_phases:
                timelines[key].append((parsed, "After Submission"))
                print(f"[STEP 3 SUBMISSION] Injected 'After Submission' for '{drug}': {date_str}")

        # Re-sort timelines
        for key in timelines:
            timelines[key].sort(key=lambda x: x[0])

    return timelines


async def step3_filing_analysis(forecast_s3: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 analysis: For each drug, uses the LLM to determine the typical
    clinical phase at which each patent type is strategically filed.

    Requires "Phase at Filing" column to be populated in the forecast_s3 table
    (done by step3_build_phase_timelines + _resolve_phase_at_filing in Step 2).

    Returns a single DataFrame with columns:
        Drug Name | Patent Type | Typical Filing Stage | Reason
    """
    if forecast_s3.empty:
        return pd.DataFrame()

    all_pattern_rows: List[Dict] = []

    for drug_name in forecast_s3["Drug Name"].unique():
        df = forecast_s3[forecast_s3["Drug Name"] == drug_name].copy()
        print(f"\n[STEP 3] ══ {drug_name} ══")

        if df.empty:
            continue

        valid = df[
            (df["Phase at Filing"].astype(str).str.strip() != "—") &
            (df["Phase at Filing"].astype(str).str.strip() != "")
        ].copy()

        if valid.empty:
            print(f"[STEP 3] No Phase at Filing data for '{drug_name}'")
            continue

        patent_lines = []
        for _, row in valid.iterrows():
            patent_lines.append(
                f"  {row['Patent Number']} | {row['Category']} | "
                f"Filed: {row['Filing Date']} | Phase: {row['Phase at Filing']} | "
                f"Layer {row['Layer']} | [{row['Jurisdiction']}] | {row['Description']}"
            )
        patent_listing = "\n".join(patent_lines)

        grouped = valid.groupby(["Category", "Phase at Filing"]).agg(
            Count=("Patent Number", "count"),
            Patents=("Patent Number", lambda x: ", ".join(x.astype(str))),
        ).reset_index()

        stats_lines = []
        for category in sorted(grouped["Category"].unique()):
            cat_rows = grouped[grouped["Category"] == category]
            phase_details = " | ".join(
                f"{r['Phase at Filing']}: {int(r['Count'])} ({r['Patents']})"
                for _, r in cat_rows.iterrows()
            )
            stats_lines.append(f"  {category}: {phase_details}")
        stats_block = "\n".join(stats_lines)

        categories = sorted(valid["Category"].unique().tolist())

        prompt = f"""You are a pharmaceutical patent strategy analyst.

Drug: {drug_name}

Below is every patent for {drug_name} with its category, filing date, and the clinical phase the drug was in when the patent was filed:

{patent_listing}

Here are the filing counts per patent type per clinical phase:

{stats_block}

YOUR TASK:
For each patent type below, determine the TYPICAL FILING STAGE — the clinical development phase when this type of patent is strategically expected to be filed by an innovator.

Do NOT just pick the phase with the most patents. Consider:
- The strategic purpose of this patent type
- The timing relative to clinical milestones
- Whether filings in a particular phase represent the innovator's deliberate strategy vs. opportunistic additions
- Industry norms for when this type of patent is typically filed

Patent types to analyse: {", ".join(categories)}

Respond ONLY with valid JSON, no markdown:
{{
  "patterns": [
    {{
      "patent_type": "<category name>",
      "typical_filing_stage": "<Phase 1 | Phase 2 | Phase 3 | Pre-Phase 1 | Marketed>",
      "reason": "<1 sentence explaining why>"
    }}
  ]
}}
"""
        try:
            print(f"[STEP 3] Asking LLM for filing patterns...")
            response = await _gemini.aio.models.generate_content(
                model    = _GEMINI_MODEL,
                contents = prompt,
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            parsed = json.loads(raw)

            for item in parsed.get("patterns", []):
                patent_type   = str(item.get("patent_type", "Unknown")).strip()
                typical_stage = str(item.get("typical_filing_stage", "—")).strip()
                reason        = str(item.get("reason", "")).strip()
                all_pattern_rows.append({
                    "Drug Name":            drug_name,
                    "Patent Type":          patent_type,
                    "Typical Filing Stage": typical_stage,
                    "Reason":               reason,
                })
                print(f"[STEP 3]   {patent_type:<25} → {typical_stage:<15} | {reason}")

        except Exception as e:
            print(f"[STEP 3] LLM failed for {drug_name}: {e}")
            for category in categories:
                cat_rows = grouped[grouped["Category"] == category]
                top = cat_rows.sort_values("Count", ascending=False).iloc[0]
                all_pattern_rows.append({
                    "Drug Name":            drug_name,
                    "Patent Type":          category,
                    "Typical Filing Stage": top["Phase at Filing"],
                    "Reason":               f"Most frequent ({int(top['Count'])} patents)",
                })

    result = pd.DataFrame(all_pattern_rows)
    print(f"\n[STEP 3] Complete — {len(result)} pattern rows across {result['Drug Name'].nunique() if not result.empty else 0} drug(s).")
    return result


# ─────────────────────────────────────────────
# Insights generation
# ─────────────────────────────────────────────

async def _generate_insights(drug_name: str, df: pd.DataFrame) -> str:
    """Asks Gemini to generate insights about the patent layering strategy for a drug."""
    summary_lines = []
    for _, row in df.iterrows():
        phase_info = f" | Phase at Filing: {row['Phase at Filing']}" if row.get('Phase at Filing', '—') != '—' else ""
        summary_lines.append(
            f"  [{row['Jurisdiction']}] Layer {row['Layer']} | {row['Category']} | "
            f"{row['Patent Number']} | Filed: {row['Filing Date']}{phase_info} | "
            f"Duration from parent: {row['Duration from Parent']} | "
            f"Parent: {row['Parent Patent']} | {row['Description']}"
        )
    summary = "\n".join(summary_lines)

    prompt = f"""You are a pharmaceutical patent strategy analyst.

Drug: {drug_name}

Below is the full patent layering tree for {drug_name}, showing how IP protection was built up over time:

{summary}

Based on this layering pattern, provide strategic insights covering:
1. How quickly the innovator built protection around the core API (Layer 1)
2. Which layers have the most patents and why
3. Notable gaps in the protection strategy (e.g. missing layers in one jurisdiction)
4. The time gaps between layers — what does the filing timeline reveal about their strategy?
5. Any asymmetry between US and EP patent estates
6. PHASE-ALIGNED FILING STRATEGY: Analyse the "Phase at Filing" data for each patent.

Write 5-8 concise bullet points. Be specific — reference actual patent numbers, dates, phases, and durations where relevant.
Respond with plain text bullet points only, no headers, no markdown.
"""
    try:
        response = await _gemini.aio.models.generate_content(
            model    = _GEMINI_MODEL,
            contents = prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[STEP 2] Insights LLM failed for {drug_name}: {e}")
        return "—"


def _compute_time_statistics(
    forecast_s3: pd.DataFrame,
    drug_names: List[str],
    innovator: str,
) -> str:
    """
    Pre-computes detailed chronological statistics from patent filing dates.
    Works on the single concatenated forecast_s3 table.
    """
    sections = []

    for drug_name in drug_names:
        df = forecast_s3[forecast_s3["Drug Name"] == drug_name].copy()
        if df.empty:
            sections.append(f"\n=== {drug_name}: No data ===")
            continue

        df["_parsed_date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
        has_dates = df.dropna(subset=["_parsed_date"])

        if has_dates.empty:
            sections.append(f"\n=== {drug_name}: No parseable filing dates ===")
            continue

        sections.append(f"\n{'='*60}")
        sections.append(f"  TIME STATISTICS FOR: {drug_name}")
        sections.append(f"{'='*60}")

        for jur in ["US", "EP"]:
            jur_df = has_dates[has_dates["Jurisdiction"] == jur].sort_values("_parsed_date")
            if jur_df.empty:
                sections.append(f"\n  [{jur}] No patents with parseable dates.")
                continue

            sections.append(f"\n  [{jur}] — {len(jur_df)} patent(s)")

            sections.append(f"  Chronological order:")
            for i, (_, r) in enumerate(jur_df.iterrows(), 1):
                phase_info = f" | Phase: {r['Phase at Filing']}" if r.get('Phase at Filing', '—') != '—' else ""
                sections.append(
                    f"    {i}. {r['Patent Number']} | Filed: {r['_parsed_date'].strftime('%Y-%m-%d')}"
                    f"{phase_info} | Layer {r['Layer']} | {r['Category']} | {r['Description']}"
                )

            first_dt = jur_df["_parsed_date"].min()
            last_dt  = jur_df["_parsed_date"].max()
            span_days  = (last_dt - first_dt).days
            span_years = round(span_days / 365.25, 2)
            sections.append(f"  First filing: {first_dt.strftime('%Y-%m-%d')} ({jur_df.iloc[0]['Patent Number']})")
            sections.append(f"  Last filing:  {last_dt.strftime('%Y-%m-%d')} ({jur_df.iloc[-1]['Patent Number']})")
            sections.append(f"  Total span:   {span_days} days ({span_years} years)")

            dates_sorted   = jur_df["_parsed_date"].tolist()
            patents_sorted = jur_df["Patent Number"].tolist()
            if len(dates_sorted) >= 2:
                sections.append(f"  Consecutive filing gaps:")
                gaps = []
                for i in range(1, len(dates_sorted)):
                    gap_days  = (dates_sorted[i] - dates_sorted[i-1]).days
                    gap_years = round(gap_days / 365.25, 2)
                    gap_months = round(gap_days / 30.44, 1)
                    gaps.append(gap_days)
                    sections.append(
                        f"    {patents_sorted[i-1]} → {patents_sorted[i]}: "
                        f"{gap_days} days ({gap_years} yrs / {gap_months} months)"
                    )
                avg_gap = round(sum(gaps) / len(gaps), 1)
                min_gap = min(gaps)
                max_gap = max(gaps)
                sections.append(f"  Average gap: {avg_gap} days ({round(avg_gap/365.25, 2)} yrs)")
                sections.append(f"  Shortest gap: {min_gap} days ({round(min_gap/365.25, 2)} yrs)")
                sections.append(f"  Longest gap:  {max_gap} days ({round(max_gap/365.25, 2)} yrs)")

                dormant = [(i, g) for i, g in enumerate(gaps) if g >= 730]
                if dormant:
                    sections.append(f"  DORMANT PERIODS (2+ years):")
                    for idx, g in dormant:
                        sections.append(
                            f"    {patents_sorted[idx]} ({dates_sorted[idx].strftime('%Y-%m-%d')}) → "
                            f"{patents_sorted[idx+1]} ({dates_sorted[idx+1].strftime('%Y-%m-%d')}): "
                            f"{g} days ({round(g/365.25, 2)} yrs) — NO FILINGS"
                        )
                else:
                    sections.append(f"  No dormant periods (all gaps < 2 years).")

            com_df = jur_df[jur_df["Layer"] == 1]
            if not com_df.empty:
                com_date = com_df["_parsed_date"].min()
                sections.append(f"  Layer timing (from CoM filed {com_date.strftime('%Y-%m-%d')}):")
                for layer in sorted(jur_df["Layer"].unique()):
                    layer_df = jur_df[jur_df["Layer"] == layer].sort_values("_parsed_date")
                    first_in_layer = layer_df.iloc[0]
                    delta_days  = (first_in_layer["_parsed_date"] - com_date).days
                    delta_years = round(delta_days / 365.25, 2)
                    sections.append(
                        f"    Layer {layer}: first filed {first_in_layer['_parsed_date'].strftime('%Y-%m-%d')} "
                        f"({first_in_layer['Patent Number']}, {first_in_layer['Category']}) — "
                        f"{delta_days} days / {delta_years} yrs after CoM"
                    )

            year_counts = jur_df["_parsed_date"].dt.year.value_counts().sort_index()
            sections.append(f"  Filings per year:")
            for yr, cnt in year_counts.items():
                yr_patents = jur_df[jur_df["_parsed_date"].dt.year == yr]
                details = ", ".join(
                    f"{r['Patent Number']} ({r['Category']}"
                    + (f", {r['Phase at Filing']}" if r.get('Phase at Filing', '—') != '—' else "")
                    + ")"
                    for _, r in yr_patents.iterrows()
                )
                sections.append(f"    {yr}: {cnt} filing(s) — {details}")

            peak_year = year_counts.idxmax()
            sections.append(f"  Peak filing year: {peak_year} ({year_counts[peak_year]} filings)")

            if "Phase at Filing" in jur_df.columns:
                phase_counts = jur_df["Phase at Filing"].value_counts()
                valid_phases = phase_counts[phase_counts.index != "—"]
                if not valid_phases.empty:
                    sections.append(f"  Patents by clinical phase at time of filing:")
                    for phase, cnt in valid_phases.items():
                        phase_patents = jur_df[jur_df["Phase at Filing"] == phase]
                        patent_list = ", ".join(
                            f"{r['Patent Number']} ({r['Category']})"
                            for _, r in phase_patents.iterrows()
                        )
                        sections.append(f"    {phase}: {cnt} patent(s) — {patent_list}")

        us_dates = has_dates[has_dates["Jurisdiction"] == "US"]
        ep_dates = has_dates[has_dates["Jurisdiction"] == "EP"]
        if not us_dates.empty and not ep_dates.empty:
            us_first = us_dates["_parsed_date"].min()
            ep_first = ep_dates["_parsed_date"].min()
            lag_days = (ep_first - us_first).days
            sections.append(f"\n  US vs EP first filing lag: {lag_days} days ({round(lag_days/365.25, 2)} yrs)")
            sections.append(f"    US first: {us_first.strftime('%Y-%m-%d')} | EP first: {ep_first.strftime('%Y-%m-%d')}")

            us_span = (us_dates["_parsed_date"].max() - us_dates["_parsed_date"].min()).days
            ep_span = (ep_dates["_parsed_date"].max() - ep_dates["_parsed_date"].min()).days
            us_rate = round(len(us_dates) / max(us_span / 365.25, 0.1), 2) if us_span > 0 else len(us_dates)
            ep_rate = round(len(ep_dates) / max(ep_span / 365.25, 0.1), 2) if ep_span > 0 else len(ep_dates)
            sections.append(f"  Filing velocity: US = {us_rate} patents/year | EP = {ep_rate} patents/year")

    if len(drug_names) >= 2:
        sections.append(f"\n{'='*60}")
        sections.append(f"  CROSS-DRUG VELOCITY COMPARISON")
        sections.append(f"{'='*60}")
        for drug_name in drug_names:
            df = forecast_s3[forecast_s3["Drug Name"] == drug_name].copy()
            if df.empty:
                continue
            df["_parsed_date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
            has = df.dropna(subset=["_parsed_date"])
            if has.empty:
                continue
            total    = len(has)
            first_dt = has["_parsed_date"].min()
            last_dt  = has["_parsed_date"].max()
            span     = (last_dt - first_dt).days
            rate     = round(total / max(span / 365.25, 0.1), 2) if span > 0 else total
            com_to_max_layer = "N/A"
            com = has[has["Layer"] == 1]
            if not com.empty:
                com_dt    = com["_parsed_date"].min()
                max_layer = has["Layer"].max()
                last_layer_dt = has[has["Layer"] == max_layer]["_parsed_date"].max()
                cl_days   = (last_layer_dt - com_dt).days
                com_to_max_layer = f"{cl_days} days ({round(cl_days/365.25, 2)} yrs) from Layer 1 to Layer {max_layer}"
            sections.append(
                f"  {drug_name}: {total} patents over {round(span/365.25, 2)} yrs | "
                f"{rate} patents/year | {com_to_max_layer}"
            )

    return "\n".join(sections)


async def _generate_innovator_insights(
    forecast_s3: pd.DataFrame,
    drug_names: List[str],
    innovator: str,
) -> pd.DataFrame:
    """
    Compares the patent layering patterns of multiple drugs from the same innovator.
    Works on the single concatenated forecast_s3 table.
    """
    summaries = []
    for drug_name in drug_names:
        df = forecast_s3[forecast_s3["Drug Name"] == drug_name]
        if df.empty:
            summaries.append(f"\n{drug_name}: No data available.")
            continue
        lines = [f"\n{drug_name}:"]
        for _, row in df.iterrows():
            phase_info = f" | Phase: {row['Phase at Filing']}" if row.get('Phase at Filing', '—') != '—' else ""
            lines.append(
                f"  [{row['Jurisdiction']}] Layer {row['Layer']} | {row['Category']} | "
                f"{row['Patent Number']} | Filed: {row['Filing Date']}{phase_info} | "
                f"Duration: {row['Duration from Parent']} | Parent: {row['Parent Patent']} | "
                f"{row['Description']}"
            )
        summaries.append("\n".join(lines))

    combined_summary = "\n".join(summaries)
    time_stats_block = _compute_time_statistics(forecast_s3, drug_names, innovator)

    prompt = f"""You are a pharmaceutical patent strategy analyst.

Innovator: {innovator}
Drugs being compared: {", ".join(drug_names)}

Below are the patent layering trees for each drug from {innovator}:

{combined_summary}

These drugs are from the same innovator. Analyse the cross-drug patent strategy and provide insights on:

1. SIMILARITIES: What is structurally similar between the patent estates of {" and ".join(drug_names)}?
2. COMMON PATTERNS: What protection layers does {innovator} consistently file across both drugs?
3. TIMING STRATEGY: How long after the CoM patent does {innovator} typically file Layer 2, Layer 3, and Layer 4 patents?
4. JURISDICTIONAL STRATEGY: Does {innovator} file US and EP patents simultaneously or stagger them?
5. EVOLUTION: How did their strategy evolve from {drug_names[0]} to {drug_names[1] if len(drug_names) > 1 else drug_names[0]}?
6. GAPS & OPPORTUNITIES: What layers are missing or weak?
7. PREDICTED NEXT MOVES: Based on this pattern, what types of patents would {innovator} likely file next?
8. TIME-BASED INSIGHTS: Below are pre-computed chronological statistics. Use ALL of these numbers exactly as given.

{time_stats_block}

Using the statistics above, write a deeply detailed chronological analysis covering:
8a. FILING TIMELINE OVERVIEW
8b. FILING CADENCE & CLUSTERING
8c. CONSECUTIVE FILING GAPS — PATTERN ANALYSIS
8d. PEAK FILING YEARS & LIFECYCLE CORRELATION
8e. FILING VELOCITY & STRATEGIC AGGRESSION
8f. LAYER BUILD-OUT TIMING — CROSS-DRUG PATTERN
8g. DORMANT PERIODS & STRATEGIC INFLECTION POINTS
8h. JURISDICTIONAL TIMING PATTERNS & FILING SEQUENCE
8i. RECURRING PATTERNS & INNOVATOR SIGNATURE
8j. PHASE-ALIGNED FILING STRATEGY

Be specific — reference actual patent numbers, exact filing dates, clinical phases, and precise durations.
Write sections 1-7 as clear paragraphs.
Write section 8 as multiple detailed paragraphs, one per sub-section (8a through 8j). Each sub-section header must appear on its own line.
"""

    try:
        print(f"[STEP 2] Generating innovator insights for {innovator} ({', '.join(drug_names)})...")
        response = await _gemini.aio.models.generate_content(
            model    = _GEMINI_MODEL,
            contents = prompt,
        )
        raw = response.text.strip()

        rows = [{"Section": "Innovator", "Content": innovator}]
        rows.append({"Section": "Drugs Compared", "Content": ", ".join(drug_names)})
        rows.append({"Section": "", "Content": ""})
        rows.append({"Section": "── GENERAL ANALYSIS ──", "Content": ""})

        current_section = ""
        current_num     = ""
        current_lines   = []
        separator_inserted = False

        for line in raw.splitlines():
            line = line.strip().strip("*").strip()
            if not line:
                if current_section and current_lines:
                    rows.append({"Section": current_section, "Content": " ".join(current_lines)})
                    current_section = ""
                    current_num     = ""
                    current_lines   = []
                continue

            header_match = re.match(r"^(\d+[a-z]?)\.\s+([A-Za-z &'\-]+?)[\s:.]?\s*$", line)
            if not header_match:
                header_match = re.match(r"^(\d+[a-z]?)\.\s+([A-Za-z &'\-]+?):\s+(.+)$", line)
                if header_match:
                    if current_section and current_lines:
                        rows.append({"Section": current_section, "Content": " ".join(current_lines)})

                    new_num = header_match.group(1)
                    if not separator_inserted and new_num.startswith("8"):
                        rows.append({"Section": "", "Content": ""})
                        rows.append({"Section": "── TIME-BASED ANALYSIS ──", "Content": ""})
                        separator_inserted = True

                    current_num     = new_num
                    current_section = header_match.group(2).strip().upper()
                    current_lines   = [header_match.group(3).strip()]
                    continue

            if header_match:
                if current_section and current_lines:
                    rows.append({"Section": current_section, "Content": " ".join(current_lines)})

                new_num = header_match.group(1)
                if not separator_inserted and new_num.startswith("8"):
                    rows.append({"Section": "", "Content": ""})
                    rows.append({"Section": "── TIME-BASED ANALYSIS ──", "Content": ""})
                    separator_inserted = True

                current_num     = new_num
                current_section = header_match.group(2).strip().upper()
                current_lines   = []
            else:
                current_lines.append(line)

        if current_section and current_lines:
            rows.append({"Section": current_section, "Content": " ".join(current_lines)})

        result = pd.DataFrame(rows, columns=["Section", "Content"])

        content_rows = result[result["Section"].str.strip() != ""]
        if len(content_rows) < 6:
            print(f"[STEP 2] Parser only found {len(content_rows)} sections — using raw fallback")
            rows = [
                {"Section": "Innovator",      "Content": innovator},
                {"Section": "Drugs Compared", "Content": ", ".join(drug_names)},
                {"Section": "",               "Content": ""},
                {"Section": "Full Response",  "Content": raw},
            ]
            result = pd.DataFrame(rows, columns=["Section", "Content"])

        return result

    except Exception as e:
        print(f"[STEP 2] Innovator insights failed: {e}")
        return pd.DataFrame([{"Section": "Error", "Content": str(e)}])


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

def print_step1(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data to display.")
        return

    col_widths = {}
    for col in df.columns:
        max_content = df[col].astype(str).apply(
            lambda x: max((len(line) for line in x.split("\n")), default=0)
        ).max()
        col_widths[col] = max(max_content, len(col)) + 2

    sep    = "─" * sum(col_widths.values())
    header = "".join(f"{col:<{col_widths[col]}}" for col in df.columns)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for _, row in df.iterrows():
        lines_per_col = {col: str(row[col]).split("\n") for col in df.columns}
        max_lines     = max(len(v) for v in lines_per_col.values())
        for line_idx in range(max_lines):
            line_out = ""
            for col in df.columns:
                lines    = lines_per_col[col]
                cell     = lines[line_idx] if line_idx < len(lines) else ""
                line_out += f"{cell:<{col_widths[col]}}"
            print(line_out)
        print()

    print(sep)
    print(f"  {len(df)} drug(s)\n")


def print_forecast_s3(df: pd.DataFrame) -> None:
    """Prints the unified forecast_s3 table."""
    if df.empty:
        print("No data to display.")
        return

    for drug_name in df["Drug Name"].unique():
        drug_df = df[df["Drug Name"] == drug_name]
        print(f"\n{'═'*80}")
        print(f"  {drug_name}")
        print(f"{'═'*80}")

        for jur in ["US", "EP"]:
            jur_df = drug_df[drug_df["Jurisdiction"] == jur]
            if jur_df.empty:
                continue
            print(f"\n  [{jur}]")
            display_cols = [c for c in jur_df.columns if c not in ("Drug Name", "Jurisdiction")]
            col_widths = {
                col: max(jur_df[col].astype(str).map(len).max(), len(col)) + 2
                for col in display_cols
            }
            sep    = "  " + "─" * sum(col_widths.values())
            header = "  " + "".join(f"{col:<{col_widths[col]}}" for col in display_cols)
            print(sep)
            print(header)
            print(sep)
            for _, row in jur_df.iterrows():
                print("  " + "".join(f"{str(row[col]):<{col_widths[col]}}" for col in display_cols))
            print(sep)


def print_step3(df: pd.DataFrame) -> None:
    """Prints Step 3 filing pattern results."""
    if df.empty:
        print("No Step 3 data to display.")
        return

    for drug_name in df["Drug Name"].unique():
        drug_df = df[df["Drug Name"] == drug_name]
        print(f"\n{'═'*70}")
        print(f"  {drug_name} — Filing Timing vs. Development Milestones")
        print(f"{'═'*70}")
        if drug_df.empty:
            print("  No data.")
            continue
        print(f"  {'Patent Type':<25} {'Typical Filing Stage':<20} {'Reason'}")
        print(f"  {'─'*25} {'─'*20} {'─'*40}")
        for _, row in drug_df.iterrows():
            print(
                f"  {row['Patent Type']:<25} {row['Typical Filing Stage']:<20} "
                f"{row.get('Reason', '')}"
            )


# ─────────────────────────────────────────────
# Output paths & BQ target
# ─────────────────────────────────────────────

OUTPUT_DIR        = None  # No longer used — output goes to GCS
OUTPUT_FILE       = None
_FORECAST_S3_BQ   = "cognito-prod-394707.cognito_prod_datamart.forecast_s3"


def _save_excel(df: pd.DataFrame, path: str, sheet_name: str) -> None:
    """Write an Excel file to GCS instead of local disk."""
    try:
        import sys, io
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cog import gcs_cache

        filename = Path(path).name
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "B2"
            for col_cells in ws.columns:
                length = max(
                    len(str(cell.value)) if cell.value else 0
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(length + 4, 60)
        buf.seek(0)

        uri = gcs_cache.write_bytes(
            "forecast_exports",
            filename,
            buf.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        print(f"Saved to GCS: {uri}")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel to GCS: {e}")
    except Exception as e:
        print(f"Export failed: {e}")


# ─────────────────────────────────────────────
# Merge all steps into a single flat table
# ─────────────────────────────────────────────

def merge_to_forecast_s3(
    step1_df: pd.DataFrame,
    step2_df: pd.DataFrame,
    step3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges Step 1, Step 2, and Step 3 outputs into a single flat table.

    Grain: one row per patent (from Step 2).
    Step 1 columns are joined by Drug Name (drug-level, repeated per patent).
    Step 3 columns are joined by Drug Name + Category (category-level, repeated per patent).

    Final columns (snake_case for BQ):
        drug_name, innovator, jurisdiction, layer, layer_reason, parent_patent,
        category, patent_number, filing_date, phase_at_filing,
        duration_from_parent, approval_date, description, insights,
        api_patent, formulation, device, method, dosing, combination,
        typical_filing_stage, filing_stage_reason
    """
    if step2_df.empty:
        print("[MERGE] Step 2 data is empty — nothing to merge.")
        return pd.DataFrame()

    result = step2_df.copy()

    # ── Join Step 1 (drug-level) ──────────────────────────────────────────
    if not step1_df.empty:
        step1_cols = ["Drug Name", "Innovator", "API Patent", "Formulation",
                      "Device", "Method", "Dosing", "Combination"]
        step1_join = step1_df[[c for c in step1_cols if c in step1_df.columns]].copy()
        # De-dup in case Step 1 has multiple rows per drug (shouldn't, but safety)
        step1_join = step1_join.drop_duplicates(subset=["Drug Name"], keep="first")
        result = result.merge(step1_join, on="Drug Name", how="left")
        print(f"[MERGE] Joined Step 1: +{len(step1_join.columns) - 1} columns")
    else:
        # Add empty Step 1 columns
        for col in ["Innovator", "API Patent", "Formulation", "Device",
                     "Method", "Dosing", "Combination"]:
            result[col] = "—"
        print("[MERGE] Step 1 data empty — added blank columns")

    # ── Join Step 3 (drug+category level) ─────────────────────────────────
    if not step3_df.empty:
        step3_join = step3_df.rename(columns={
            "Patent Type":          "Category",
            "Typical Filing Stage": "Typical Filing Stage",
            "Reason":               "Filing Stage Reason",
        })
        step3_join = step3_join[["Drug Name", "Category", "Typical Filing Stage",
                                  "Filing Stage Reason"]].copy()
        step3_join = step3_join.drop_duplicates(subset=["Drug Name", "Category"], keep="first")
        result = result.merge(step3_join, on=["Drug Name", "Category"], how="left")
        print(f"[MERGE] Joined Step 3: +2 columns")
    else:
        result["Typical Filing Stage"] = "—"
        result["Filing Stage Reason"]  = "—"
        print("[MERGE] Step 3 data empty — added blank columns")

    # ── Reorder columns ───────────────────────────────────────────────────
    _FINAL_COLUMN_ORDER = [
        "Drug Name", "Innovator", "Jurisdiction", "Layer", "Layer Reason",
        "Parent Patent", "Category", "Patent Number", "Filing Date",
        "Phase at Filing", "Duration from Parent", "Approval Date",
        "Description", "Insights",
        "API Patent", "Formulation", "Device", "Method", "Dosing", "Combination",
        "Typical Filing Stage", "Filing Stage Reason",
    ]
    # Only keep columns that exist
    ordered = [c for c in _FINAL_COLUMN_ORDER if c in result.columns]
    # Append any extra columns not in the order list
    extras = [c for c in result.columns if c not in ordered]
    result = result[ordered + extras]

    # ── Snake_case all column names for BQ ────────────────────────────────
    result.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in result.columns
    ]

    # Fill NaN with "—"
    result = result.fillna("—")

    print(f"[MERGE] Final table: {len(result)} rows × {len(result.columns)} columns")
    print(f"[MERGE] Columns: {list(result.columns)}")
    return result


def upload_to_bigquery(df: pd.DataFrame, table_id: str = _FORECAST_S3_BQ) -> None:
    """Uploads the merged forecast_s3 DataFrame to BigQuery, appending to the table."""
    if df.empty:
        print("[BQ UPLOAD] DataFrame is empty — skipping upload.")
        return

    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        )
        job = _BQ_CLIENT.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for completion
        print(f"[BQ UPLOAD] Uploaded {len(df)} rows to: {table_id}")
    except Exception as e:
        print(f"[BQ UPLOAD] Failed: {e}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forecasting pipeline (BigQuery edition).")
    parser.add_argument(
        "--upload", action="store_true", default=False,
        help="Upload the final merged table to BigQuery after processing."
    )
    args = parser.parse_args()

    def _run(coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
            return loop.run_until_complete(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: IP Landscape
    # ══════════════════════════════════════════════════════════════════════
    print("\n── STEP 1 — IP Landscape of Marketed Drugs ──\n")
    step1_result = _run(step1_ip_landscape())
    print_step1(step1_result)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Patent Layering Pattern
    # ══════════════════════════════════════════════════════════════════════
    print("\n── STEP 2 — Patent Layering Pattern ──\n")
    step2_result = _run(step2_patent_layering())
    print_forecast_s3(step2_result)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Filing Timing vs. Development Milestones
    # ══════════════════════════════════════════════════════════════════════
    step3_result = pd.DataFrame()
    if not step2_result.empty:
        print("\n── STEP 3 — Filing Timing vs. Development Milestones ──\n")
        step3_result = _run(step3_filing_analysis(step2_result))
        print_step3(step3_result)

    # ══════════════════════════════════════════════════════════════════════
    # MERGE → Single flat table (forecast_s3)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── MERGING — All steps into forecast_s3 ──\n")
    forecast_s3 = merge_to_forecast_s3(step1_result, step2_result, step3_result)

    if not forecast_s3.empty:
        # Save to GCS
        _save_excel(forecast_s3, "forecast_s3.xlsx", "forecast_s3")

        # Upload to BigQuery if requested
        if args.upload:
            upload_to_bigquery(forecast_s3)
        else:
            print(f"[INFO] To upload to BigQuery, re-run with --upload")

    print("\n── DONE ──")
