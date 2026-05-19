"""
step6_forecast_pipeline.py
──────────────────────────
Merged pipeline: Step 6 (Patent Forecast Generator) + Forecast Scorer.

Reads inputs from BigQuery:
  - Step 2/3 → cognito-prod-394707.cognito_prod_datamart.forecast_s3
  - Step 4   → cognito-prod-394707.cognito_prod_datamart.filing_pattern_table
  - Step 5   → cognito-prod-394707.cognito_prod_datamart.company_analysis_table

Generates patent forecasts via Gemini, scores them, and writes output
to a BigQuery table.

Output:
  - BigQuery table: <BQ_PROJECT_ID>.<BQ_DATASET_ID>.forecasted_loe

Usage:
    python step6_forecast_pipeline.py                        # all drugs
    python step6_forecast_pipeline.py --drug Semaglutide     # single drug
    python step6_forecast_pipeline.py --limit 5              # first 5
    python step6_forecast_pipeline.py --dry-run              # just list
"""

import asyncio
import json
import os
import re as _re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run
from google import genai
from google.genai import types

load_dotenv(override=True)

# ── Make local modules importable ─────────────────────────────────────────────
_here   = Path(__file__).resolve().parent
_parent = _here.parent
_pkg    = _here.name

for _p in [str(_here), str(_parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

_api_key      = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
_gemini       = genai.Client(api_key=_api_key)
_GEMINI_MODEL = "gemini-2.5-flash"

_CURRENT_YEAR  = datetime.now().year
_CURRENT_DATE  = datetime.now().strftime("%Y-%m-%d")

BQ_PROJECT_ID      = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID      = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_TABLE_NAME      = os.getenv("BQ_TABLE_NAME")         # drug-company mapping table
BQ_SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# ── Input BQ tables ──────────────────────────────────────────────────────────
BQ_FORECAST_S3_TABLE       = os.getenv("BQ_FORECAST_S3_TABLE",
    "cognito-prod-394707.cognito_prod_datamart.forecast_s3")
BQ_FILING_PATTERN_TABLE    = os.getenv("BQ_FILING_PATTERN_TABLE",
    "cognito-prod-394707.cognito_prod_datamart.filing_pattern_table")
BQ_COMPANY_ANALYSIS_TABLE  = os.getenv("BQ_COMPANY_ANALYSIS_TABLE",
    "cognito-prod-394707.cognito_prod_datamart.company_analysis_table")

# ── Output BQ table ─────────────────────────────────────────────────────────
BQ_OUTPUT_TABLE = os.getenv("BQ_OUTPUT_TABLE",
    f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.forecasted_loe")


# ─────────────────────────────────────────────
# BigQuery client helper
# ─────────────────────────────────────────────

_bq_client = None

def _get_bq_client():
    global _bq_client
    if _bq_client is not None:
        return _bq_client

    from google.cloud import bigquery
    from google.oauth2 import service_account

    if BQ_SERVICE_ACCOUNT and os.path.exists(BQ_SERVICE_ACCOUNT):
        credentials = service_account.Credentials.from_service_account_file(BQ_SERVICE_ACCOUNT)
        _bq_client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
    else:
        _bq_client = bigquery.Client(project=BQ_PROJECT_ID)

    return _bq_client


# ─────────────────────────────────────────────
# 1. Load drug → company mapping from BigQuery
# ─────────────────────────────────────────────

def _load_drug_company_map() -> Dict[str, Dict]:
    """
    Queries BigQuery for drug → parent company mapping.
    Returns dict: {drug_name_lower: {"company": str, "geographies": [str]}}
    """
    if not all([BQ_PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_NAME]):
        print("[BQ] Missing BQ_TABLE_NAME config — drug-company mapping unavailable")
        return {}

    try:
        client = _get_bq_client()
        fq_table = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_NAME}"

        query = f"""
        WITH base AS (
          SELECT DISTINCT
            cleaned_generic_name,
            Parent_Company_Name,
            Drug_Geography
          FROM `{fq_table}`
          WHERE
            (
              UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
              OR UPPER(cleaned_Target) LIKE '%GLP-1%'
              OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
            )
            OR (
              data_source = 'IPD'
              AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
            )
        )
        SELECT
          b.cleaned_generic_name,
          b.Parent_Company_Name,
          b.Drug_Geography
        FROM base AS b
        WHERE EXISTS (
          SELECT 1
          FROM UNNEST(REGEXP_EXTRACT_ALL(b.Drug_Geography, r'[^,;]+')) AS x
          WHERE TRIM(x) IN ('EU', 'United States', 'China', 'India', 'Brazil',
                             'Australia', 'Russia', 'Canada', 'Japan', 'Mexico',
                             'Taiwan', 'South Korea',
                             'CN', 'IN', 'BR', 'AU', 'RU', 'US', 'CA', 'JP',
                             'MX', 'TW', 'KR', 'EP')
        )
        ORDER BY b.cleaned_generic_name
        """

        print(f"[BQ] Querying drug-company mapping from {fq_table}...")
        df = client.query(query).to_dataframe()

        mapping = {}
        for _, row in df.iterrows():
            drug    = str(row.get("cleaned_generic_name", "")).strip()
            company = str(row.get("Parent_Company_Name", "")).strip()
            geo_raw = str(row.get("Drug_Geography", "")).strip()
            if drug and company and drug.lower() not in ("nan", "none", ""):
                key = drug.lower()
                if key not in mapping:
                    mapping[key] = {"company": company, "geographies": []}
                for geo in geo_raw.replace(";", ",").split(","):
                    geo = geo.strip()
                    if geo and geo not in mapping[key]["geographies"]:
                        mapping[key]["geographies"].append(geo)

        print(f"[BQ] Loaded {len(mapping)} drug→company mapping(s)")
        for d, info in sorted(mapping.items()):
            geos = ", ".join(info["geographies"][:5])
            print(f"  {d} → {info['company']} [{geos}]")

        return mapping

    except Exception as e:
        print(f"[BQ] Drug-company mapping query failed: {e}")
        return {}


# ─────────────────────────────────────────────
# 2. Load Step 2/3 from BQ (forecast_s3)
# ─────────────────────────────────────────────

def _load_step23_from_bq() -> Dict[str, pd.DataFrame]:
    """
    Reads Step 2/3 data from BQ table: forecast_s3.
    Groups rows by drug name and returns {drug_name: DataFrame}.
    """
    try:
        client = _get_bq_client()
        query = f"SELECT * FROM `{BQ_FORECAST_S3_TABLE}`"
        print(f"[STEP 2/3] Loading from {BQ_FORECAST_S3_TABLE}...")
        df = client.query(query).to_dataframe()

        if df.empty:
            print("[STEP 2/3] Table is empty")
            return {}

        # Identify the drug name column (try common names)
        drug_col = None
        for candidate in ["drug_name", "Drug", "drug", "cleaned_generic_name",
                          "Drug Name", "drug_name_clean", "generic_name"]:
            if candidate in df.columns:
                drug_col = candidate
                break

        if drug_col is None:
            # Fall back to first column
            drug_col = df.columns[0]
            print(f"[STEP 2/3] Drug column not found — using '{drug_col}'")

        drug_dfs = {}
        for drug_name, group_df in df.groupby(drug_col):
            name = str(drug_name).strip()
            if name and name.lower() not in ("nan", "none", ""):
                drug_dfs[name] = group_df.reset_index(drop=True)

        print(f"[STEP 2/3] Loaded {len(drug_dfs)} drug(s) from BQ")
        return drug_dfs

    except Exception as e:
        print(f"[STEP 2/3] Failed to load from BQ: {e}")
        return {}


# ─────────────────────────────────────────────
# 3. Load Step 4 from BQ (filing_pattern_table)
# ─────────────────────────────────────────────

def _load_step4_from_bq(drug_name: str) -> str:
    """
    Reads innovator filing pattern data from BQ for a specific drug.
    Returns a text summary.
    """
    try:
        client = _get_bq_client()
        query = f"""
        SELECT *
        FROM `{BQ_FILING_PATTERN_TABLE}`
        WHERE LOWER(drug_name) = LOWER(@drug)
           OR LOWER(drug_name) LIKE CONCAT('%%', LOWER(@drug), '%%')
        """
        job_config = _bq_job_config({"drug": drug_name})
        df = client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return ""

        lines = [f"INNOVATOR FILING PATTERNS FOR {drug_name}:"]
        for _, row in df.iterrows():
            company = row.get("Company", row.get("company", "N/A"))
            char    = row.get("Characterization", row.get("characterization", "N/A"))
            conf    = row.get("Confidence", row.get("confidence", "N/A"))
            rat     = row.get("Rationale", row.get("rationale", "N/A"))
            lines.append(f"  {company}: {char} (confidence: {conf})")
            lines.append(f"    {rat}")

        result = "\n".join(lines)
        print(f"[STEP 4] Loaded filing patterns for {drug_name} from BQ ({len(result)} chars)")
        return result

    except Exception as e:
        print(f"[STEP 4] Failed to load from BQ for {drug_name}: {e}")
        return ""


def _load_all_step4_from_bq() -> Dict[str, str]:
    """Load ALL Step 4 data from BQ in one query, return {drug_name_lower: text}."""
    try:
        client = _get_bq_client()
        query = f"SELECT * FROM `{BQ_FILING_PATTERN_TABLE}`"
        print(f"[STEP 4] Loading all filing patterns from {BQ_FILING_PATTERN_TABLE}...")
        df = client.query(query).to_dataframe()

        if df.empty:
            print("[STEP 4] Table is empty")
            return {}

        # Identify drug column
        drug_col = None
        for candidate in ["drug_name", "Drug", "drug", "cleaned_generic_name"]:
            if candidate in df.columns:
                drug_col = candidate
                break
        if drug_col is None:
            drug_col = df.columns[0]

        result = {}
        for drug_name, group in df.groupby(drug_col):
            name = str(drug_name).strip()
            if not name or name.lower() in ("nan", "none", ""):
                continue

            lines = [f"INNOVATOR FILING PATTERNS FOR {name}:"]
            for _, row in group.iterrows():
                company = row.get("Company", row.get("company", "N/A"))
                char    = row.get("Characterization", row.get("characterization", "N/A"))
                conf    = row.get("Confidence", row.get("confidence", "N/A"))
                rat     = row.get("Rationale", row.get("rationale", "N/A"))
                lines.append(f"  {company}: {char} (confidence: {conf})")
                lines.append(f"    {rat}")

            result[name.lower()] = "\n".join(lines)

        print(f"[STEP 4] Loaded filing patterns for {len(result)} drug(s)")
        return result

    except Exception as e:
        print(f"[STEP 4] Failed to load all from BQ: {e}")
        return {}


# ─────────────────────────────────────────────
# 4. Load Step 5 from BQ (company_analysis_table)
# ─────────────────────────────────────────────

def _load_step5_from_bq(company_name: str) -> str:
    """
    Reads company analysis from BQ for a specific company.
    Returns a text summary.
    """
    if not company_name:
        return ""

    try:
        client = _get_bq_client()
        query = f"""
        SELECT *
        FROM `{BQ_COMPANY_ANALYSIS_TABLE}`
        WHERE LOWER(company_name) = LOWER(@company)
           OR LOWER(company_name) LIKE CONCAT('%%', LOWER(@company), '%%')
        """
        job_config = _bq_job_config({"company": company_name})
        df = client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            print(f"[STEP 5] No company analysis found in BQ for '{company_name}'")
            return ""

        lines = [f"BUSINESS STRATEGY ASSESSMENT FOR {company_name}:"]
        for _, row in df.iterrows():
            section = row.get("Section", row.get("section", ""))
            rating  = row.get("Rating", row.get("rating", ""))
            details = row.get("Details", row.get("details", ""))
            lines.append(f"  {section}: {rating}")
            if details and str(details).strip() not in ("", "nan", "N/A"):
                lines.append(f"    {str(details)[:500]}")

        result = "\n".join(lines)
        print(f"[STEP 5] Loaded company analysis for {company_name} from BQ ({len(result)} chars)")
        return result

    except Exception as e:
        print(f"[STEP 5] Failed to load from BQ for {company_name}: {e}")
        return ""


def _load_all_step5_from_bq() -> Dict[str, str]:
    """Load ALL Step 5 data from BQ in one query, return {company_name_lower: text}."""
    try:
        client = _get_bq_client()
        query = f"SELECT * FROM `{BQ_COMPANY_ANALYSIS_TABLE}`"
        print(f"[STEP 5] Loading all company analyses from {BQ_COMPANY_ANALYSIS_TABLE}...")
        df = client.query(query).to_dataframe()

        if df.empty:
            print("[STEP 5] Table is empty")
            return {}

        # Identify company column
        comp_col = None
        for candidate in ["company_name", "Company", "company", "Parent_Company_Name"]:
            if candidate in df.columns:
                comp_col = candidate
                break
        if comp_col is None:
            comp_col = df.columns[0]

        result = {}
        for company, group in df.groupby(comp_col):
            name = str(company).strip()
            if not name or name.lower() in ("nan", "none", ""):
                continue

            lines = [f"BUSINESS STRATEGY ASSESSMENT FOR {name}:"]
            for _, row in group.iterrows():
                section = row.get("Section", row.get("section", ""))
                rating  = row.get("Rating", row.get("rating", ""))
                details = row.get("Details", row.get("details", ""))
                lines.append(f"  {section}: {rating}")
                if details and str(details).strip() not in ("", "nan", "N/A"):
                    lines.append(f"    {str(details)[:500]}")

            result[name.lower()] = "\n".join(lines)

        print(f"[STEP 5] Loaded company analyses for {len(result)} company(ies)")
        return result

    except Exception as e:
        print(f"[STEP 5] Failed to load all from BQ: {e}")
        return {}


# ─────────────────────────────────────────────
# BQ parameterized query helper
# ─────────────────────────────────────────────

def _bq_job_config(params: Dict[str, str]):
    from google.cloud import bigquery
    query_params = []
    for name, value in params.items():
        query_params.append(
            bigquery.ScalarQueryParameter(name, "STRING", value)
        )
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    return job_config


# ─────────────────────────────────────────────
# Web search fallbacks (Gemini)
# ─────────────────────────────────────────────

async def _search_company_for_drug(drug_name: str) -> str:
    """Uses Gemini with web search to identify the innovator company for a drug."""
    prompt = (
        f'Who is the original innovator / originator pharmaceutical company '
        f'that developed the drug "{drug_name}"?\n\n'
        f'Search the web thoroughly. Return ONLY the company name. No explanation.'
    )
    try:
        response = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            ),
        )
        name = (response.text or "").strip().strip(".")
        print(f"[SEARCH FALLBACK] Company for '{drug_name}': {name}")
        return name or ""
    except Exception as e:
        print(f"[SEARCH FALLBACK] Company search failed for '{drug_name}': {e}")
        return ""


async def _search_innovator_patterns(drug_name: str, company_name: str) -> str:
    """Uses Gemini with web search to gather innovator patent filing behaviour."""
    prompt = f"""You are a pharmaceutical patent intelligence analyst.

Research the patent filing patterns of {company_name or "the innovator company"} for the drug "{drug_name}".

Search Espacenet, Google Patents, and any other patent databases.

Analyse:
1. How many patents does the company hold for {drug_name}? List key patent numbers if found.
2. What types of patents? (composition of matter, formulation, device, method of treatment, dosing regimen, combination, salt/polymorph, manufacturing)
3. Does the company file continuation applications aggressively?
4. Does the company build dense patent thickets or file minimally?
5. Does the company expand protection late in development (after Phase 3 / approval)?
6. What is the company's overall IP strategy characterisation?

Provide a detailed summary with specific patent numbers and filing dates where available.
"""
    try:
        response = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        result = (response.text or "").strip()
        if result:
            print(f"[SEARCH FALLBACK] Innovator patterns for '{drug_name}': {len(result)} chars")
            return f"INNOVATOR FILING PATTERNS FOR {drug_name} (from web search):\n{result}"
        return ""
    except Exception as e:
        print(f"[SEARCH FALLBACK] Innovator patterns search failed: {e}")
        return ""


async def _search_company_analysis(company_name: str, drug_name: str) -> str:
    """Uses Gemini with web search to gather company business strategy intelligence."""
    prompt = f"""You are a pharmaceutical business analyst.

Research {company_name}'s business strategy, focusing on their drug "{drug_name}" and the broader therapeutic area.

Search the web for:
1. {company_name}'s total annual revenue and revenue from {drug_name}
2. Pipeline assets related to {drug_name} or its therapeutic class
3. Recent management statements about {drug_name} strategy
4. Competitive landscape
5. Patent expiry dates and generic/biosimilar threats
6. New indications, formulations, devices, or combinations being developed
7. Lifecycle management strategies

Provide a detailed summary with specific numbers, dates, and quotes where available.
"""
    try:
        response = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        result = (response.text or "").strip()
        if result:
            print(f"[SEARCH FALLBACK] Company analysis for '{company_name}': {len(result)} chars")
            return f"BUSINESS STRATEGY ASSESSMENT FOR {company_name} (from web search):\n{result}"
        return ""
    except Exception as e:
        print(f"[SEARCH FALLBACK] Company analysis search failed: {e}")
        return ""


# ─────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────

def _fix_filing_window(window: str) -> str:
    """Ensure filing_window starts at or after the current year."""
    if not window or not isinstance(window, str):
        return f"{_CURRENT_YEAR}-{_CURRENT_YEAR + 3}"

    years = [int(y) for y in _re.findall(r"\d{4}", window)]
    if not years:
        return f"{_CURRENT_YEAR}-{_CURRENT_YEAR + 3}"

    if len(years) == 1:
        start = max(years[0], _CURRENT_YEAR)
        return f"{start}-{start + 2}"

    start, end = years[0], years[-1]

    if start < _CURRENT_YEAR:
        span = max(end - start, 1)
        start = _CURRENT_YEAR
        end = start + span

    if end < _CURRENT_YEAR:
        end = _CURRENT_YEAR + 2
    if end <= start:
        end = start + 2

    return f"{start}-{end}"


def _normalize_phase(phase_str: str) -> str:
    """Normalize phase strings to simple format."""
    if not phase_str or not isinstance(phase_str, str):
        return "N/A"

    s = phase_str.strip()
    s_upper = s.upper()

    if "APPROVED" in s_upper or "MARKETED" in s_upper or "LAUNCHED" in s_upper:
        return "Approved"
    if "SUBMIT" in s_upper or "BLA" in s_upper or "NDA" in s_upper or "MAA" in s_upper:
        return "Submitted"
    if "PRECLINICAL" in s_upper or "PRE-CLINICAL" in s_upper or "DISCOVERY" in s_upper:
        return "Preclinical"
    if "DISCONTINUE" in s_upper or "WITHDRAW" in s_upper or "TERMINAT" in s_upper:
        return "Discontinued"
    if "NOT APPLICABLE" in s_upper or "N/A" in s_upper or "UNKNOWN" in s_upper:
        return "N/A"
    if "NOT FILED" in s_upper or "NO FILING" in s_upper:
        return "Not filed"

    m = _re.search(r"(?:phase\s*)(IV|III|II|I|[1-4])(?:\s*/\s*(IV|III|II|I|[1-4]))?",
                    s, _re.IGNORECASE)
    if m:
        roman = {"I": "1", "II": "2", "III": "3", "IV": "4"}
        p1 = m.group(1).upper()
        p1 = roman.get(p1, p1)
        if m.group(2):
            p2 = m.group(2).upper()
            p2 = roman.get(p2, p2)
            return f"Phase {p1}/{p2}"
        sub = _re.search(r"(?:phase\s*(?:IV|III|II|I|[1-4])\s*)([ab])",
                          s, _re.IGNORECASE)
        if sub:
            return f"Phase {p1}{sub.group(1).lower()}"
        return f"Phase {p1}"

    return s[:20]


def _validate_forecast(forecast: Dict) -> Dict:
    """Post-process a forecast to fix common issues."""
    if not forecast or "forecast" not in forecast:
        return forecast

    for entry in forecast["forecast"]:
        entry["filing_window"] = _fix_filing_window(entry.get("filing_window", ""))
        if "drug_phase_in_jurisdiction" in entry:
            entry["drug_phase_in_jurisdiction"] = _normalize_phase(
                entry.get("drug_phase_in_jurisdiction", "")
            )
        try:
            ep = int(entry.get("estimated_patents", 1))
            entry["estimated_patents"] = max(ep, 1)
        except (ValueError, TypeError):
            entry["estimated_patents"] = 1

    if "current_phase" in forecast:
        forecast["current_phase"] = _normalize_phase(forecast.get("current_phase", ""))

    return forecast


# ─────────────────────────────────────────────
# Forecast scoring helpers (from forecast_scorer)
# ─────────────────────────────────────────────

def _extract_lower_year(filing_window: str) -> Optional[int]:
    if not filing_window or str(filing_window).strip().lower() in ("n/a", "nan", "none", ""):
        return None
    years = _re.findall(r'(20\d{2})', str(filing_window))
    return min(int(y) for y in years) if years else None


def _calc_score(avg_yte: float) -> int:
    if avg_yte <= 3:
        return 5
    elif avg_yte <= 5:
        return 4
    elif avg_yte <= 8:
        return 3
    elif avg_yte <= 10:
        return 2
    else:
        return 1


def _score_forecast(drug_name: str, forecast: Dict) -> List[Dict]:
    """
    Scores a drug's forecast entries.
    For each forecasted patent:
      controlling_patent_expiry = lower_filing_year + 20
      years_to_entry = controlling_patent_expiry - current_year
    Then:
      avg_years_to_entry = mean of max(US), max(EP/EU)
      score = 1-5
    """
    current_phase = forecast.get("current_phase", "")

    # Est. Approval Year based on phase
    phase_lower = current_phase.strip().lower()
    if "phase 3" in phase_lower:
        est_approval_year = _CURRENT_YEAR + 3
    elif "phase 2" in phase_lower:
        est_approval_year = _CURRENT_YEAR + 5
    elif "approved" in phase_lower:
        est_approval_year = _CURRENT_YEAR  # already approved
    else:
        est_approval_year = None

    patents = []

    for entry in forecast.get("forecast", []):
        patent_type = entry.get("patent_type", "").strip()
        filing_win  = entry.get("filing_window", "").strip()
        jur         = entry.get("jurisdiction", "").strip()
        likelihood  = entry.get("likelihood", "").strip()

        if not jur or jur.lower() in ("nan", "none", ""):
            continue
        if not likelihood or likelihood.lower() in ("nan", "none", ""):
            continue

        lower_year = _extract_lower_year(filing_win)
        if not lower_year:
            continue

        expiry = lower_year + 20
        yte    = expiry - _CURRENT_YEAR

        # Exclusivity Year
        jur_upper = jur.upper()
        if est_approval_year is not None:
            if jur_upper == "US":
                exclusivity_year = est_approval_year + 5
            elif jur_upper in ("EU", "EP"):
                exclusivity_year = est_approval_year + 10
            else:
                exclusivity_year = None
        else:
            exclusivity_year = None

        est_patents_raw = entry.get("estimated_patents")
        est_patents = None if pd.isna(est_patents_raw) else est_patents_raw

        patents.append({
            "drug_name":                        drug_name,
            "company":                          forecast.get("company", ""),
            "global_phase":                     current_phase,
            "drug_class":                       forecast.get("drug_class", ""),
            "patent_number":                    f"{jur}+Forecasted+{patent_type}",
            "tag":                              "BLOCKING",
            "type":                             "Forecasted",
            "step1_claim_category":             patent_type,
            "jurisdiction":                     jur,
            "phase_in_jurisdiction":            entry.get("drug_phase_in_jurisdiction", ""),
            "likelihood":                       likelihood,
            "filing_window":                    filing_win,
            "filing_date_lower":                lower_year,
            "rationale":                        entry.get("rationale", ""),
            "strategic_purpose":                entry.get("strategic_purpose", ""),
            "controlling_patent_expiry_year":   expiry,
            "years_to_entry":                   yte,
            "est_approval_year":                est_approval_year,
            "exclusivity_year":                 exclusivity_year,
            "no_of_forecasted_patents":         est_patents,
            "overall_forecast":                 forecast.get("overall_forecast", ""),
            "portfolio_gaps":                   " | ".join(forecast.get("portfolio_gaps", [])),
            "risk_assessment":                  forecast.get("risk_assessment", ""),
            "existing_patent_summary":          forecast.get("existing_patent_summary", ""),
        })

    if not patents:
        return []

    # Keep highest YTE row per jurisdiction — blank YTE on all other rows
    best_per_jur = {}
    for p in patents:
        jur = p["jurisdiction"].upper()
        if jur not in best_per_jur or p["years_to_entry"] > best_per_jur[jur]["years_to_entry"]:
            best_per_jur[jur] = p

    for p in patents:
        if p is not best_per_jur.get(p["jurisdiction"].upper()):
            p["years_to_entry"] = None

    # Drug-level score: max YTE per jurisdiction, then avg, then score
    us_yte = best_per_jur.get("US", {}).get("years_to_entry")
    ep_yte = next((best_per_jur[j].get("years_to_entry")
                   for j in ("EP", "EU") if j in best_per_jur), None)

    yte_values = [v for v in [us_yte, ep_yte] if v is not None]
    avg_yte = round(sum(yte_values) / len(yte_values), 2) if yte_values else None
    score   = _calc_score(avg_yte) if avg_yte is not None else None

    # Avg across all jurisdictions
    all_yte = [p["years_to_entry"] for p in patents if p["years_to_entry"] is not None]
    avg_yte_all = round(sum(all_yte) / len(all_yte), 2) if all_yte else None

    for p in patents:
        p["avg_years_to_entry_us_ep"] = avg_yte
        p["avg_years_to_entry"]       = avg_yte_all
        p["ip_dimension_1_score"]     = score
        p["scored_at"]                = _CURRENT_DATE

    print(f"  {drug_name}: {len(forecast.get('forecast', []))} forecasts | "
          f"US max YTE={us_yte} | EP max YTE={ep_yte} | "
          f"Avg={avg_yte} | Score={score}")

    return patents


# ─────────────────────────────────────────────
# Generate patent forecast for a drug (Gemini)
# ─────────────────────────────────────────────

async def _generate_drug_forecast(
    drug_name: str,
    company_name: str,
    patent_data: Optional[pd.DataFrame],
    innovator_patterns: str,
    company_analysis: str,
) -> Dict:
    """Uses Gemini to synthesise all inputs and produce a patent filing forecast."""
    # Build patent summary from Step 2/3 data
    patent_summary_lines = []
    if patent_data is not None and not patent_data.empty:
        patent_summary_lines.append(f"EXISTING PATENT PORTFOLIO ({len(patent_data)} patents):")
        if "Category" in patent_data.columns:
            cats = patent_data["Category"].value_counts()
            patent_summary_lines.append(f"  Categories: {dict(cats)}")
        if "Layer" in patent_data.columns:
            layers = patent_data["Layer"].value_counts()
            patent_summary_lines.append(f"  Layers: {dict(layers)}")
        if "Phase at Filing" in patent_data.columns:
            phases = patent_data["Phase at Filing"].value_counts()
            patent_summary_lines.append(f"  Phase at Filing: {dict(phases)}")
        if "Jurisdiction" in patent_data.columns:
            juris = patent_data["Jurisdiction"].unique().tolist()
            patent_summary_lines.append(f"  Jurisdictions: {juris}")
        if "Filing Date" in patent_data.columns:
            dates = pd.to_datetime(patent_data["Filing Date"], errors="coerce").dropna()
            if not dates.empty:
                patent_summary_lines.append(
                    f"  Filing range: {dates.min().strftime('%Y')} — {dates.max().strftime('%Y')}")
        if "Insights" in patent_data.columns:
            insights = patent_data["Insights"].dropna().astype(str)
            insights = insights[insights.str.strip() != ""]
            if not insights.empty:
                patent_summary_lines.append(f"\n  LAYERING INSIGHTS:\n  {insights.iloc[0][:2000]}")

        cols_to_show = [c for c in ["Patent Number", "Category", "Layer", "Filing Date",
                                     "Phase at Filing", "Description"] if c in patent_data.columns]
        if cols_to_show:
            patent_summary_lines.append(f"\n  PATENT DETAILS (first 30):")
            for _, row in patent_data.head(30).iterrows():
                line = " | ".join(f"{c}: {row[c]}" for c in cols_to_show)
                patent_summary_lines.append(f"    {line}")

    patent_summary = ("\n".join(patent_summary_lines)
                      if patent_summary_lines
                      else "No existing patent data available.")

    prompt = f"""You are a senior pharmaceutical patent strategy forecaster with access to web search.

TODAY'S DATE: {_CURRENT_DATE}
CURRENT YEAR: {_CURRENT_YEAR}

DRUG: {drug_name}
COMPANY: {company_name or "Unknown — search the web to identify the innovator company"}

You have 3 sources of intelligence below. IF ANY SOURCE IS MISSING OR SAYS "No data available", YOU MUST USE WEB SEARCH to find that information yourself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 1: EXISTING PATENT PORTFOLIO (from Step 2/3 analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{patent_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 2: INNOVATOR FILING BEHAVIOUR (from Step 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{innovator_patterns or "No innovator filing pattern data available. SEARCH THE WEB for patent filing behaviour of the innovator company for this drug."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE 3: COMPANY BUSINESS STRATEGY (from Step 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{company_analysis or "No company analysis available. SEARCH THE WEB for business strategy, revenue, pipeline, and competitive landscape of the innovator company."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Generate a COMPLETE PATENT FILING FORECAST for {drug_name} covering ALL patent types in ALL jurisdictions.

CRITICAL DATE RULE: Today is {_CURRENT_DATE}. This is a FORECAST of FUTURE patent filings.
- ALL filing_window values MUST start at {_CURRENT_YEAR} or later.
- If a patent type was already filed, forecast the NEXT expected filing starting from {_CURRENT_YEAR} or later.
- If no future filing is expected, still include with likelihood "Low" and filing_window starting at {_CURRENT_YEAR}.

PER-JURISDICTION PHASE REQUIREMENT:
For EACH jurisdiction, determine the drug's current regulatory/development phase IN THAT SPECIFIC JURISDICTION by searching the web.

Consider ALL of the following:
- FROM STEP 1 (IP Landscape): Patent categories already covered vs. missing
- FROM STEP 2 (Patent Layering): Patent protection tree structure and gaps
- FROM STEP 3 (Filing Pattern by Phase): Typical filing phase for each patent type
- FROM STEP 4 (Innovator Filing Behaviour): Dense thickets vs. minimal filing
- FROM STEP 5 (Business Strategy): Commercial importance, revenue dependency, lifecycle management
- SCORING CONTEXT: controlling_patent_expiry = lower_filing_year + 20; years_to_entry = expiry - current_year

PATENT TYPES (exactly 6):
  Device, Dosage Regimen, Formulation, Manufacturing Process, Method of treatment, Salt/Polymorph

JURISDICTIONS (exactly 12):
  CN, IN, BR, AU, RU, US, CA, JP, MX, TW, KR, EP

You MUST produce 6 × 12 = 72 entries.

For EACH entry provide:
- patent_type: exactly one of the 6 labels
- jurisdiction: exactly one of the 12 codes
- drug_phase_in_jurisdiction: simple phase label
- likelihood: Very High / High / Moderate / Low
- filing_window: year range starting at {_CURRENT_YEAR} or later
- rationale: 2-3 sentences with jurisdiction-specific reasoning
- strategic_purpose: business goal in this jurisdiction
- estimated_patents: integer (minimum 1)

Respond with JSON:
{{
  "drug": "{drug_name}",
  "company": "<company name>",
  "current_phase": "<most advanced phase globally>",
  "drug_class": "<e.g., GLP-1 receptor agonist peptide>",
  "existing_patent_summary": "<1-2 sentence summary>",
  "forecast": [
    {{
      "patent_type": "<type>",
      "jurisdiction": "<code>",
      "drug_phase_in_jurisdiction": "<phase>",
      "likelihood": "<level>",
      "filing_window": "<year range>",
      "rationale": "<2-3 sentences>",
      "strategic_purpose": "<goal>",
      "estimated_patents": <integer>
    }}
  ],
  "portfolio_gaps": ["<gap 1>", "<gap 2>"],
  "risk_assessment": "<2-3 sentences>",
  "overall_forecast": "<3-5 sentence executive summary>"
}}

CRITICAL RULES:
- 72 entries (6 types × 12 jurisdictions). No skipping.
- filing_window MUST start at {_CURRENT_YEAR} or later.
- estimated_patents MUST be >= 1.
- "company" field MUST contain the actual company name, never "Unknown".
"""

    try:
        response = await _gemini.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        forecast = json.loads(raw)
        forecast = _validate_forecast(forecast)
        return forecast

    except Exception as e:
        print(f"[FORECAST] Gemini forecast failed for {drug_name}: {e}")
        return {"drug": drug_name, "error": str(e)}


# ─────────────────────────────────────────────
# Write scored results to BigQuery
# ─────────────────────────────────────────────

def _write_to_bq(all_scored: List[Dict], table_id: str = None):
    """Writes all scored forecast rows to a BigQuery table."""
    table_id = table_id or BQ_OUTPUT_TABLE

    if not all_scored:
        print("[BQ OUTPUT] No scored data to write")
        return

    df = pd.DataFrame(all_scored)

    # Ensure consistent column types
    int_cols = ["filing_date_lower", "controlling_patent_expiry_year",
                "no_of_forecasted_patents", "ip_dimension_1_score"]
    float_cols = ["years_to_entry", "avg_years_to_entry_us_ep", "avg_years_to_entry",
                  "est_approval_year", "exclusivity_year"]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        from google.cloud import bigquery

        client = _get_bq_client()

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        )

        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for completion

        table = client.get_table(table_id)
        print(f"\n[BQ OUTPUT] Written {table.num_rows} rows to {table_id}")
        print(f"[BQ OUTPUT] Schema: {len(table.schema)} columns")

    except Exception as e:
        print(f"[BQ OUTPUT] Failed to write to {table_id}: {e}")
        # Fallback: save locally
        fallback_path = "forecasted_loe_fallback.parquet"
        df.to_parquet(fallback_path, index=False)
        print(f"[BQ OUTPUT] Saved fallback to {fallback_path}")


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

async def run_pipeline(drug_names: List[str]):
    """Runs the full forecast + scoring pipeline for all specified drugs."""

    print(f"\n{'═'*80}")
    print(f"  STEP 6 FORECAST + SCORING PIPELINE")
    print(f"  Date: {_CURRENT_DATE}")
    print(f"{'═'*80}\n")

    # ── Load all inputs from BQ ──────────────────────────────────────────────
    print("[PIPELINE] Loading inputs from BigQuery...\n")

    drug_company_map = _load_drug_company_map()
    forecasting_data = _load_step23_from_bq()
    step4_data       = _load_all_step4_from_bq()
    step5_data       = _load_all_step5_from_bq()

    if not forecasting_data:
        print(f"[PIPELINE] No Step 2/3 data found in {BQ_FORECAST_S3_TABLE}")
        print(f"[PIPELINE] Proceeding with web search fallback for patent data")

    # If no drug list provided, use drugs from Step 2/3
    if not drug_names:
        drug_names = sorted(forecasting_data.keys())
        if not drug_names:
            # Try drug-company map
            drug_names = sorted(set(
                info.get("company", k).title() if k else k
                for k, info in drug_company_map.items()
            ))

    if not drug_names:
        print("[PIPELINE] No drugs found. Provide --drug or populate the BQ tables.")
        return

    print(f"\n[PIPELINE] Drugs to process: {len(drug_names)}")
    for i, d in enumerate(drug_names, 1):
        print(f"  {i}. {d}")

    # ── Process each drug ────────────────────────────────────────────────────
    total = len(drug_names)
    all_scored = []
    succeeded = 0
    failed = []

    for i, drug_name in enumerate(drug_names, 1):
        print(f"\n{'█'*80}")
        print(f"  [{i}/{total}] Forecasting & Scoring: {drug_name}")
        print(f"{'█'*80}")

        # Find company and geographies
        drug_info = drug_company_map.get(drug_name.lower(), {})
        if not drug_info:
            drug_norm = drug_name.lower().replace(" ", "").replace("-", "")
            for k, v in drug_company_map.items():
                if drug_norm in k.replace(" ", "").replace("-", ""):
                    drug_info = v
                    break
        company_name = drug_info.get("company", "") if isinstance(drug_info, dict) else ""
        geographies  = drug_info.get("geographies", []) if isinstance(drug_info, dict) else []

        # Web search fallback for company
        if not company_name:
            print(f"  Company not found in BQ — searching web...")
            company_name = await _search_company_for_drug(drug_name)

        print(f"  Company: {company_name or 'Unknown'}")
        if geographies:
            print(f"  Geographies: {', '.join(geographies)}")

        # Load patent data (Step 2/3 from BQ)
        patent_df = None
        for sheet_name, df in forecasting_data.items():
            if sheet_name.lower().strip() == drug_name.lower().strip():
                patent_df = df
                break
        if patent_df is None:
            drug_norm = drug_name.lower().replace(" ", "").replace("-", "")
            for sheet_name, df in forecasting_data.items():
                if drug_norm in sheet_name.lower().replace(" ", "").replace("-", ""):
                    patent_df = df
                    break
        if patent_df is not None:
            print(f"  Step 2/3 (BQ): {len(patent_df)} rows")
        else:
            print(f"  Step 2/3: not found in BQ")

        # Load Step 4 (innovator patterns from BQ)
        innovator_patterns = ""
        drug_lower = drug_name.lower()
        if drug_lower in step4_data:
            innovator_patterns = step4_data[drug_lower]
            print(f"  Step 4 (BQ): {len(innovator_patterns)} chars")
        else:
            # Try fuzzy match
            for k, v in step4_data.items():
                if drug_lower.replace(" ", "").replace("-", "") in k.replace(" ", "").replace("-", ""):
                    innovator_patterns = v
                    print(f"  Step 4 (BQ fuzzy): {len(innovator_patterns)} chars")
                    break

        # Load Step 5 (company analysis from BQ)
        company_analysis = ""
        if company_name:
            comp_lower = company_name.lower()
            if comp_lower in step5_data:
                company_analysis = step5_data[comp_lower]
                print(f"  Step 5 (BQ): {len(company_analysis)} chars")
            else:
                for k, v in step5_data.items():
                    if comp_lower in k or k in comp_lower:
                        company_analysis = v
                        print(f"  Step 5 (BQ fuzzy): {len(company_analysis)} chars")
                        break

        # Web search fallbacks for missing data
        if not innovator_patterns:
            print(f"  Step 4 missing — searching web...")
            innovator_patterns = await _search_innovator_patterns(drug_name, company_name)

        if not company_analysis:
            print(f"  Step 5 missing — searching web...")
            company_analysis = await _search_company_analysis(company_name, drug_name)

        # ── Generate forecast ────────────────────────────────────────────────
        print(f"\n  Generating patent forecast...")
        try:
            forecast = await _generate_drug_forecast(
                drug_name=drug_name,
                company_name=company_name,
                patent_data=patent_df,
                innovator_patterns=innovator_patterns,
                company_analysis=company_analysis,
            )

            if forecast and "error" not in forecast:
                # Print forecast summary
                print(f"\n  Global Phase: {forecast.get('current_phase', '?')}")
                print(f"  Class: {forecast.get('drug_class', '?')}")
                print(f"  Forecasted patents: {len(forecast.get('forecast', []))}")
                for entry in forecast.get("forecast", [])[:5]:
                    print(f"    • [{entry.get('likelihood','?')}] {entry.get('patent_type','?')} "
                          f"[{entry.get('jurisdiction','')}] ({entry.get('filing_window','')}) "
                          f"~{entry.get('estimated_patents', '?')} patents")
                if len(forecast.get("forecast", [])) > 5:
                    print(f"    ... and {len(forecast['forecast']) - 5} more entries")

                # ── Score the forecast ───────────────────────────────────────
                print(f"\n  Scoring forecast...")
                scored = _score_forecast(drug_name, forecast)

                if scored:
                    all_scored.extend(scored)
                    succeeded += 1
                else:
                    print(f"  No scoreable entries")
                    failed.append((drug_name, "No scoreable entries"))
            else:
                err = forecast.get("error", "No data")
                print(f"  Forecast failed: {err}")
                failed.append((drug_name, err))

        except Exception as e:
            print(f"  [ERROR] {drug_name}: {e}")
            failed.append((drug_name, str(e)))

    # ── Write all scored results to BigQuery ─────────────────────────────────
    if all_scored:
        print(f"\n{'═'*80}")
        print(f"  WRITING SCORED RESULTS TO BIGQUERY")
        print(f"{'═'*80}")
        _write_to_bq(all_scored)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  PIPELINE COMPLETE — Forecast + Scoring")
    print(f"{'═'*80}")
    print(f"  Drugs:     {total}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Output:    {BQ_OUTPUT_TABLE}")

    if succeeded:
        print(f"\n  {'Drug Name':<30} {'Score':<8} {'Avg YTE':<12} {'Patents'}")
        print(f"  {'─'*65}")
        seen = set()
        for p in all_scored:
            drug = p["drug_name"]
            if drug in seen:
                continue
            seen.add(drug)
            print(f"  {drug:<30} {p['ip_dimension_1_score']:<8} "
                  f"{p['avg_years_to_entry_us_ep']:<12} "
                  f"{len([x for x in all_scored if x['drug_name'] == drug])}")

    if failed:
        print(f"\n  Failed:")
        for drug, err in failed:
            print(f"    - {drug}: {err}")

    print(f"{'═'*80}\n")

    return all_scored


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6 + Scoring: Generate and score patent forecasts. "
                    "Reads from BQ, writes scored output to BQ."
    )
    parser.add_argument("--drug",         default=None,  help="Single drug name")
    parser.add_argument("--limit",        type=int, default=None, help="Limit number of drugs")
    parser.add_argument("--dry-run",      action="store_true", help="List drugs only")
    parser.add_argument("--output-table", default=None,  help="Override output BQ table")
    args = parser.parse_args()

    if args.output_table:
        BQ_OUTPUT_TABLE = args.output_table

    # Get drug list
    if args.drug:
        drugs = [args.drug]
    else:
        # Load drugs from Step 2/3 BQ table
        try:
            data = _load_step23_from_bq()
            drugs = sorted(data.keys()) if data else []
        except Exception:
            drugs = []

        if not drugs:
            print(f"No drugs found in {BQ_FORECAST_S3_TABLE}. Use --drug to specify one.")
            sys.exit(1)

    if args.limit:
        drugs = drugs[:args.limit]
        print(f"\n[LIMIT] First {args.limit} drug(s) only")

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(drugs)} drug(s):")
        for i, d in enumerate(drugs, 1):
            print(f"  {i}. {d}")
        sys.exit(0)

    # Run
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

    start = datetime.now()
    _run(run_pipeline(drugs))
    print(f"Total time: {(datetime.now() - start).total_seconds():.1f}s")
