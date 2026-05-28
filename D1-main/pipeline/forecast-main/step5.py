"""
business_strategy_reviewer.py
─────────────────────────────
Review Business Strategy and Revenue Priorities.

Purpose:
    Understand whether the company has strong incentives to expand patent protection.

Data sources:
  1. SEC EDGAR API        — 10-K / 20-F annual filings (full text)
  2. Vertex AI Search     — investor presentations + annual reports
  3. Gemini web search    — supplemental intelligence (grounding)

Output:
  - BigQuery table  : PROJECT_ID.DATASET_ID.TABLE_ID
  - Optional Excel  : --export flag

Usage:
    python business_strategy_reviewer.py --company "Novo Nordisk" --therapy "GLP-1"
    python business_strategy_reviewer.py --company "Eli Lilly" --therapy "GLP-1" --drugs "Tirzepatide"
    python business_strategy_reviewer.py --company "Novo Nordisk" --therapy "GLP-1" \\
        --drugs "Semaglutide,Liraglutide" --export
    python business_strategy_reviewer.py          # batch mode — reads companies from BigQuery
"""

import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run

# ── Google Cloud ──────────────────────────────────────────────────────────────
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Tool, grounding
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these or set via environment variables
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID       = os.getenv("GCP_PROJECT_ID",      "cognito-prod-394707")
BQ_LOCATION      = os.getenv("BQ_LOCATION",          "asia-south1")
BQ_PROJECT_ID    = os.getenv("BQ_PROJECT_ID",        "cognito-prod-394707")
BQ_DATASET_ID    = os.getenv("BQ_DATASET_ID",        "cognito_prod_datamart")
BQ_TABLE_ID      = os.getenv("BQ_TABLE_ID",          "company_analysis_table")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

VERTEX_LOCATION  = os.getenv("VERTEX_LOCATION",      "us-central1")
GEMINI_MODEL     = "gemini-2.5-flash-preview-05-20"

_HTTP_HEADERS = {
    "User-Agent": "PatentPipeline/1.0 (patent.analysis@example.com)",
    "Accept":     "text/html,application/json",
}
_HTTP_DELAY = 0.5   # SEC EDGAR polite rate-limit

# ─────────────────────────────────────────────────────────────────────────────
# Credentials + client initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _build_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and Path(CREDENTIALS_PATH).exists():
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        print(f"[AUTH] Using service account: {CREDENTIALS_PATH}")
        return creds
    print("[AUTH] Using application default credentials (Cloud Run)")
    return None   # google-auth will fall back to ADC


_credentials = _build_credentials()

# Vertex AI SDK init
vertexai.init(
    project     = PROJECT_ID,
    location    = VERTEX_LOCATION,
    credentials = _credentials,
)

# BigQuery client
_bq_client = bigquery.Client(
    project     = BQ_PROJECT_ID,
    location    = BQ_LOCATION,
    credentials = _credentials,
)

# Gemini model (Vertex AI)
_gemini = GenerativeModel(GEMINI_MODEL)

print(f"[INIT] Gemini model : {GEMINI_MODEL}")
print(f"[INIT] BQ target    : {BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}")


# ─────────────────────────────────────────────────────────────────────────────
# BigQuery schema + table helpers
# ─────────────────────────────────────────────────────────────────────────────

_BQ_SCHEMA = [
    # ── metadata ─────────────────────────────────────────────────────────────
    bigquery.SchemaField("run_id",              "STRING",  description="UUID for this run"),
    bigquery.SchemaField("inserted_at",         "TIMESTAMP", description="Row insertion timestamp (UTC)"),
    bigquery.SchemaField("company",             "STRING"),
    bigquery.SchemaField("therapy_area",        "STRING"),
    bigquery.SchemaField("drugs",               "STRING",  description="Comma-separated drug list"),
    bigquery.SchemaField("analysis_date",       "DATE"),
    bigquery.SchemaField("sources_used",        "STRING",  description="Semicolon-separated source labels"),

    # ── revenue_dependency ───────────────────────────────────────────────────
    bigquery.SchemaField("rev_rating",          "STRING"),
    bigquery.SchemaField("rev_total_revenue",   "STRING"),
    bigquery.SchemaField("rev_therapy_revenue", "STRING"),
    bigquery.SchemaField("rev_percentage",      "STRING"),
    bigquery.SchemaField("rev_growth_rate",     "STRING"),
    bigquery.SchemaField("rev_drug_breakdown",  "STRING"),
    bigquery.SchemaField("rev_details",         "STRING"),

    # ── pipeline_depth ───────────────────────────────────────────────────────
    bigquery.SchemaField("pipe_rating",         "STRING"),
    bigquery.SchemaField("pipe_assets",         "STRING",  description="JSON array of asset strings"),
    bigquery.SchemaField("pipe_acquisitions",   "STRING"),
    bigquery.SchemaField("pipe_details",        "STRING"),

    # ── strategic_priority ───────────────────────────────────────────────────
    bigquery.SchemaField("strat_rating",        "STRING"),
    bigquery.SchemaField("strat_key_quotes",    "STRING",  description="JSON array of quote strings"),
    bigquery.SchemaField("strat_restructuring", "STRING"),
    bigquery.SchemaField("strat_details",       "STRING"),

    # ── indication_expansion ─────────────────────────────────────────────────
    bigquery.SchemaField("ind_rating",          "STRING"),
    bigquery.SchemaField("ind_planned",         "STRING",  description="JSON array of indication strings"),
    bigquery.SchemaField("ind_patent_impl",     "STRING"),
    bigquery.SchemaField("ind_details",         "STRING"),

    # ── lifecycle_management ─────────────────────────────────────────────────
    bigquery.SchemaField("lc_rating",           "STRING"),
    bigquery.SchemaField("lc_strategies",       "STRING",  description="JSON array of strategy strings"),
    bigquery.SchemaField("lc_new_formulations", "STRING"),
    bigquery.SchemaField("lc_new_devices",      "STRING"),
    bigquery.SchemaField("lc_new_dosing",       "STRING"),
    bigquery.SchemaField("lc_combinations",     "STRING"),
    bigquery.SchemaField("lc_details",          "STRING"),

    # ── competitive_pressure ─────────────────────────────────────────────────
    bigquery.SchemaField("comp_rating",         "STRING"),
    bigquery.SchemaField("comp_competitors",    "STRING",  description="JSON array of competitor strings"),
    bigquery.SchemaField("comp_patent_expiry",  "STRING"),
    bigquery.SchemaField("comp_biosimilar",     "STRING"),
    bigquery.SchemaField("comp_defense",        "STRING"),
    bigquery.SchemaField("comp_details",        "STRING"),

    # ── patent_expansion_likelihood ──────────────────────────────────────────
    bigquery.SchemaField("pat_rating",          "STRING"),
    bigquery.SchemaField("pat_formulation",     "STRING"),
    bigquery.SchemaField("pat_device",          "STRING"),
    bigquery.SchemaField("pat_dosing_regimen",  "STRING"),
    bigquery.SchemaField("pat_combination",     "STRING"),
    bigquery.SchemaField("pat_method_treatment","STRING"),
    bigquery.SchemaField("pat_manufacturing",   "STRING"),

    # ── overall ──────────────────────────────────────────────────────────────
    bigquery.SchemaField("overall_assessment",  "STRING"),
]


def _ensure_bq_table() -> str:
    """Create BQ table if it doesn't exist. Returns fully-qualified table ID."""
    fq = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
    try:
        _bq_client.get_table(fq)
        print(f"[BQ] Table exists: {fq}")
    except Exception:
        table = bigquery.Table(fq, schema=_BQ_SCHEMA)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="analysis_date",
        )
        _bq_client.create_table(table, exists_ok=True)
        print(f"[BQ] Table created: {fq}")
    return fq


def _assessment_to_bq_row(assessment: Dict) -> Dict:
    """Flatten the nested assessment dict into a single BQ row dict."""
    import uuid

    meta  = assessment.get("_metadata", {})
    rev   = assessment.get("revenue_dependency",          {})
    pipe  = assessment.get("pipeline_depth",              {})
    strat = assessment.get("strategic_priority",          {})
    ind   = assessment.get("indication_expansion",        {})
    lc    = assessment.get("lifecycle_management",        {})
    comp  = assessment.get("competitive_pressure",        {})
    pat   = assessment.get("patent_expansion_likelihood", {})

    def _lst(v) -> str:
        """Serialise a list to JSON string."""
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        return str(v) if v else ""

    def _s(v) -> str:
        if v is None:
            return ""
        return str(v)

    today = datetime.now(timezone.utc).date().isoformat()

    return {
        # metadata
        "run_id":              str(uuid.uuid4()),
        "inserted_at":         datetime.now(timezone.utc).isoformat(),
        "company":             _s(meta.get("company")),
        "therapy_area":        _s(meta.get("therapy_area")),
        "drugs":               ", ".join(meta.get("drugs", [])),
        "analysis_date":       meta.get("date", today),
        "sources_used":        "; ".join(meta.get("sources_used", [])),

        # revenue_dependency
        "rev_rating":          _s(rev.get("rating")),
        "rev_total_revenue":   _s(rev.get("total_company_revenue")),
        "rev_therapy_revenue": _s(rev.get("therapy_area_revenue")),
        "rev_percentage":      _s(rev.get("percentage")),
        "rev_growth_rate":     _s(rev.get("growth_rate")),
        "rev_drug_breakdown":  _s(rev.get("drug_breakdown")),
        "rev_details":         _s(rev.get("details")),

        # pipeline_depth
        "pipe_rating":         _s(pipe.get("rating")),
        "pipe_assets":         _lst(pipe.get("assets")),
        "pipe_acquisitions":   _s(pipe.get("recent_acquisitions")),
        "pipe_details":        _s(pipe.get("details")),

        # strategic_priority
        "strat_rating":        _s(strat.get("rating")),
        "strat_key_quotes":    _lst(strat.get("key_quotes")),
        "strat_restructuring": _s(strat.get("restructuring")),
        "strat_details":       _s(strat.get("details")),

        # indication_expansion
        "ind_rating":          _s(ind.get("rating")),
        "ind_planned":         _lst(ind.get("planned_indications")),
        "ind_patent_impl":     _s(ind.get("patent_implications")),
        "ind_details":         _s(ind.get("details")),

        # lifecycle_management
        "lc_rating":           _s(lc.get("rating")),
        "lc_strategies":       _lst(lc.get("strategies")),
        "lc_new_formulations": _s(lc.get("new_formulations")),
        "lc_new_devices":      _s(lc.get("new_devices")),
        "lc_new_dosing":       _s(lc.get("new_dosing")),
        "lc_combinations":     _s(lc.get("combinations")),
        "lc_details":          _s(lc.get("details")),

        # competitive_pressure
        "comp_rating":         _s(comp.get("rating")),
        "comp_competitors":    _lst(comp.get("competitors")),
        "comp_patent_expiry":  _s(comp.get("patent_expiry_dates")),
        "comp_biosimilar":     _s(comp.get("biosimilar_threats")),
        "comp_defense":        _s(comp.get("defense_strategy")),
        "comp_details":        _s(comp.get("details")),

        # patent_expansion_likelihood
        "pat_rating":          _s(pat.get("rating")),
        "pat_formulation":     _s(pat.get("formulation_patents")),
        "pat_device":          _s(pat.get("device_patents")),
        "pat_dosing_regimen":  _s(pat.get("dosing_regimen_patents")),
        "pat_combination":     _s(pat.get("combination_patents")),
        "pat_method_treatment":_s(pat.get("method_of_treatment_patents")),
        "pat_manufacturing":   _s(pat.get("manufacturing_patents")),

        # overall
        "overall_assessment":  _s(assessment.get("overall_assessment")),
    }


def _write_to_bq(assessment: Dict) -> bool:
    """Insert a single assessment row into BigQuery. Returns True on success."""
    if "error" in assessment:
        print("[BQ] Skipping — assessment has errors")
        return False
    try:
        fq  = _ensure_bq_table()
        row = _assessment_to_bq_row(assessment)
        errors = _bq_client.insert_rows_json(fq, [row])
        if errors:
            print(f"[BQ] Insert errors: {errors}")
            return False
        print(f"[BQ] Row inserted → {fq}")
        return True
    except Exception as e:
        print(f"[BQ] Write failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Known company → config
# ─────────────────────────────────────────────────────────────────────────────

_COMPANY_CONFIG = {
    "novo nordisk": {"sec_cik": "0000353278", "ticker": "NVO"},
    "eli lilly":    {"sec_cik": "0000059478", "ticker": "LLY"},
    "astrazeneca":  {"sec_cik": "0000806535", "ticker": "AZN"},
    "pfizer":       {"sec_cik": "0000078003", "ticker": "PFE"},
    "amgen":        {"sec_cik": "0000318154", "ticker": "AMGN"},
}


def _get_company_config(company: str) -> Dict:
    key = company.strip().lower()
    for known, cfg in _COMPANY_CONFIG.items():
        if known in key or key in known:
            return cfg
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _http_get(url: str, **kwargs) -> Optional[requests.Response]:
    try:
        time.sleep(_HTTP_DELAY)
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=15, **kwargs)
        if r.status_code == 200:
            return r
        print(f"[HTTP] {r.status_code} → {url}")
    except Exception as e:
        print(f"[HTTP] Error: {url} — {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — SEC EDGAR
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_cik(company: str) -> Optional[str]:
    cfg = _get_company_config(company)
    if cfg.get("sec_cik"):
        print(f"[SEC] Known CIK: {cfg['sec_cik']} for '{company}'")
        return cfg["sec_cik"]
    resp = _http_get("https://www.sec.gov/files/company_tickers.json")
    if resp:
        try:
            tickers = resp.json()
            lc = company.strip().lower()
            for _, entry in tickers.items():
                if lc in str(entry.get("title", "")).lower():
                    cik = str(entry["cik_str"]).zfill(10)
                    print(f"[SEC] Resolved CIK: {cik}")
                    return cik
        except Exception:
            pass
    return None


def _fetch_sec_10k(company: str) -> str:
    cik = _resolve_cik(company)
    if not cik:
        print(f"[SEC] Could not resolve CIK for '{company}'")
        return ""

    resp = _http_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not resp:
        return ""

    try:
        data        = resp.json()
        co_name     = data.get("name", company)
        recent      = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        accessions  = recent.get("accessionNumber", [])
        dates       = recent.get("filingDate", [])
        primary_doc = recent.get("primaryDocument", [])

        filing_url = filing_date = form_type = None
        for i, form in enumerate(forms):
            if form in ("10-K", "20-F", "10-K/A", "20-F/A"):
                acc = accessions[i].replace("-", "")
                filing_url  = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{acc}/{primary_doc[i]}"
                )
                filing_date = dates[i]
                form_type   = form
                print(f"[SEC] {form} filed {filing_date}: {filing_url}")
                break

        if not filing_url:
            print(f"[SEC] No 10-K/20-F found for '{company}'")
            return ""

        doc_resp = _http_get(filing_url)
        if not doc_resp:
            return ""

        text = BeautifulSoup(doc_resp.text, "html.parser").get_text(separator="\n", strip=True)
        if len(text) > 200_000:
            text = text[:200_000]
        print(f"[SEC] Extracted {len(text)} chars from {form_type}")
        return f"SOURCE: SEC {form_type} filed {filing_date}\nCompany: {co_name}\n\n{text}"

    except Exception as e:
        print(f"[SEC] Parse error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Vertex AI Search
# ─────────────────────────────────────────────────────────────────────────────

def _vertex_search(company: str, therapy_area: str) -> str:
    """
    Uses the Vertex AI Search grounding tool (via Gemini) to pull
    investor presentations and annual-report content.
    The grounding backend queries the public web corpus that is indexed
    by Google Search — no custom data store required.

    If you have a private Vertex AI Search data store containing your
    company documents, set VERTEX_SEARCH_ENGINE_ID and the code will
    use it instead.
    """
    engine_id = os.getenv("VERTEX_SEARCH_ENGINE_ID", "")

    queries = [
        f"{company} latest investor presentation earnings slides",
        f"{company} annual report 2024 2025 revenue pipeline strategy",
        f"{company} {therapy_area} product revenue pipeline patent strategy",
    ]

    results = []
    for q in queries:
        try:
            if engine_id:
                # ── Private data store ─────────────────────────────────────
                tool = Tool.from_retrieval(
                    grounding.Retrieval(
                        grounding.VertexAISearch(
                            datastore=engine_id,
                            project=PROJECT_ID,
                            location="global",
                        )
                    )
                )
            else:
                # ── Google Search grounding (public web) ───────────────────
                tool = Tool.from_google_search_retrieval(
                    grounding.GoogleSearchRetrieval()
                )

            response = _gemini.generate_content(
                f"Find and summarise the most relevant content about: {q}\n"
                f"Focus on revenue figures, pipeline assets, strategy statements, "
                f"and patent/IP information. Be specific with numbers and dates.",
                tools=[tool],
            )
            text = response.text or ""
            if text.strip():
                results.append(f"QUERY: {q}\n\n{text.strip()}")
                print(f"[VERTEX SEARCH] {len(text)} chars for: {q[:60]}")
        except Exception as e:
            print(f"[VERTEX SEARCH] Failed for '{q}': {e}")

    if not results:
        return ""

    combined = "\n\n---\n\n".join(results)
    return f"SOURCE: Vertex AI Search (investor presentations + annual reports)\n\n{combined}"


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — Gemini web search (supplemental grounding)
# ─────────────────────────────────────────────────────────────────────────────

def _search_supplemental(company: str, therapy_area: str, drugs: List[str]) -> str:
    drug_list       = ", ".join(drugs) if drugs else ""
    is_company_wide = therapy_area == "all therapeutic areas"

    if is_company_wide:
        focus = (
            f"1. {company} total annual revenue breakdown by therapeutic area with exact figures.\n"
            f"2. {company} full pipeline across all therapeutic areas.\n"
            f"3. CEO / management quotes about overall strategy (last 12 months).\n"
            f"4. All patent cliffs, generic/biosimilar threats across the portfolio.\n"
            f"5. Top 10 products by revenue with growth rates."
        )
    elif drug_list:
        focus = (
            f"1. {company} revenue from {drug_list} with exact figures and YoY growth.\n"
            f"2. Pipeline assets related to {drug_list}.\n"
            f"3. CEO quotes about {drug_list} strategy.\n"
            f"4. Patent expiry dates and generic/biosimilar threats for {drug_list}.\n"
            f"5. New indications and formulations being developed for {drug_list}."
        )
    else:
        focus = (
            f"1. {company} revenue from {therapy_area} with individual product figures.\n"
            f"2. Full pipeline in {therapy_area} (all phases).\n"
            f"3. CEO / management quotes about {therapy_area} strategy.\n"
            f"4. Patent expiry timeline for {company}'s {therapy_area} portfolio.\n"
            f"5. Competitive landscape in {therapy_area}."
        )

    prompt = (
        f"You are a pharmaceutical business analyst. Search for the most recent "
        f"information about {company}.\n\nFind:\n{focus}\n\n"
        f"Use only factual sourced information. Cite the source for each finding. "
        f"Every point must have substantive content."
    )

    try:
        tool     = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        response = _gemini.generate_content(prompt, tools=[tool])
        text     = (response.text or "").strip()
        print(f"[SEARCH] {len(text)} chars of supplemental intelligence")
        return f"SOURCE: Gemini web search (supplemental)\n\n{text}"
    except Exception as e:
        print(f"[SEARCH] Web search failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM assessment
# ─────────────────────────────────────────────────────────────────────────────

def _build_assessment_prompt(
    company: str,
    therapy_area: str,
    drugs: List[str],
    combined: str,
) -> str:
    drug_list       = ", ".join(drugs) if drugs else ""
    is_company_wide = therapy_area == "all therapeutic areas"

    if is_company_wide:
        scope_desc  = f"across {company}'s entire portfolio"
        rev_instr   = (
            f"Exact total revenue + breakdown by every therapeutic area. "
            f"Top 5 products by revenue. YoY growth per segment."
        )
        pipe_instr  = f"All pipeline assets grouped by therapy area."
        strat_instr = f"Overall strategic direction, priority therapy areas, pivots."
        ind_instr   = f"New indications across all therapy areas grouped by product."
        lc_instr    = f"Lifecycle strategies across all key products."
        comp_instr  = f"Competitive threats + patent cliffs across the entire portfolio."
        pat_instr   = f"Patent expansion across the entire company — by therapy area."
    elif drug_list:
        scope_desc  = f"focused on {drug_list} in {therapy_area}"
        rev_instr   = f"Exact revenue per drug: {drug_list}. % of total. YoY growth."
        pipe_instr  = f"All pipeline in {therapy_area} or related to {drug_list}."
        strat_instr = f"Management quotes specifically about {drug_list}."
        ind_instr   = f"Every new indication being pursued for {drug_list}."
        lc_instr    = f"All lifecycle strategies for {drug_list}."
        comp_instr  = f"Direct competitors to {drug_list}. Patent expiry dates."
        pat_instr   = f"Patent expansion specifically for {drug_list}."
    else:
        scope_desc  = f"focused on {company}'s {therapy_area} portfolio"
        rev_instr   = (
            f"Exact {therapy_area} segment revenue. Individual product figures. "
            f"% of total. YoY growth."
        )
        pipe_instr  = f"All pipeline in {therapy_area}."
        strat_instr = f"Management quotes about {therapy_area} as a strategic priority."
        ind_instr   = f"All new indications across {company}'s {therapy_area} portfolio."
        lc_instr    = f"Lifecycle strategies across all {therapy_area} products."
        comp_instr  = f"All competitors in {therapy_area}. Patent expiry timeline."
        pat_instr   = f"Patent expansion across the entire {therapy_area} portfolio."

    return f"""You are a senior pharmaceutical patent strategy analyst.

Company: {company}
Scope: {scope_desc}

Intelligence gathered from SEC filings, Vertex AI Search, and web search:

{combined[:90_000]}

Provide a detailed structured assessment {scope_desc}.

RULES:
- Every field must have substantive content. Never leave anything blank or write "N/A".
- Use exact numbers, dates, drug names and quotes from the source material.
- If exact figures are unavailable, provide a reasoned estimate clearly labelled as estimated.

1. REVENUE DEPENDENCY        : {rev_instr}   Rating: Critical >40% | High 20-40% | Moderate 10-20% | Low <10%
2. PIPELINE DEPTH            : {pipe_instr}  Rating: Deep 5+ | Moderate 3-4 | Shallow 1-2 | None
3. STRATEGIC PRIORITY        : {strat_instr} Rating: Core Franchise | Growth Driver | Maintained | De-prioritised
4. INDICATION EXPANSION      : {ind_instr}   Rating: Aggressive | Moderate | Minimal | None
5. LIFECYCLE MANAGEMENT      : {lc_instr}    Rating: Active | Planned | Minimal | None
6. COMPETITIVE PRESSURE      : {comp_instr}  Rating: Imminent <2yr | Near-term 2-5yr | Distant >5yr | None visible
7. PATENT EXPANSION LIKELIHOOD: {pat_instr}  Rating: Very High | High | Moderate | Low
   Assess each type: formulation | device | dosing regimen | combination | method of treatment | manufacturing
8. OVERALL ASSESSMENT: 5-7 sentence executive summary of commercial incentive for patent expansion.

Respond ONLY with valid JSON (no markdown fences):
{{
  "revenue_dependency": {{
    "rating": "<Critical|High|Moderate|Low>",
    "total_company_revenue": "<figure + currency + year>",
    "therapy_area_revenue": "<figure or segment breakdown>",
    "percentage": "<X% of total>",
    "growth_rate": "<YoY %>",
    "drug_breakdown": "<revenue per drug/segment>",
    "details": "<detailed analysis with specific numbers>"
  }},
  "pipeline_depth": {{
    "rating": "<Deep|Moderate|Shallow|None>",
    "assets": ["<name — mechanism — stage — indication — milestone>"],
    "recent_acquisitions": "<acquisitions>",
    "details": "<detailed analysis>"
  }},
  "strategic_priority": {{
    "rating": "<Core Franchise|Growth Driver|Maintained|De-prioritised>",
    "key_quotes": ["<exact quote — source — date>"],
    "restructuring": "<organisational changes>",
    "details": "<detailed analysis>"
  }},
  "indication_expansion": {{
    "rating": "<Aggressive|Moderate|Minimal|None>",
    "planned_indications": ["<indication — drug — stage — trial — timeline>"],
    "patent_implications": "<which indications could generate new patents>",
    "details": "<detailed analysis>"
  }},
  "lifecycle_management": {{
    "rating": "<Active|Planned|Minimal|None>",
    "strategies": ["<strategy type — product — stage — evidence>"],
    "new_formulations": "<formulation programmes>",
    "new_devices": "<device programmes>",
    "new_dosing": "<dosing programmes>",
    "combinations": "<combination programmes>",
    "details": "<detailed analysis>"
  }},
  "competitive_pressure": {{
    "rating": "<Imminent|Near-term|Distant|None visible>",
    "competitors": ["<product — company — status — market share>"],
    "patent_expiry_dates": "<known expiry dates>",
    "biosimilar_threats": "<biosimilar/generic threats>",
    "defense_strategy": "<company's competitive strategy>",
    "details": "<detailed analysis>"
  }},
  "patent_expansion_likelihood": {{
    "rating": "<Very High|High|Moderate|Low>",
    "formulation_patents": "<detailed assessment with evidence>",
    "device_patents": "<detailed assessment with evidence>",
    "dosing_regimen_patents": "<detailed assessment with evidence>",
    "combination_patents": "<detailed assessment with evidence>",
    "method_of_treatment_patents": "<detailed assessment with evidence>",
    "manufacturing_patents": "<detailed assessment with evidence>"
  }},
  "overall_assessment": "<5-7 sentence executive summary>"
}}
"""


def _parse_json_response(raw: str) -> Optional[Dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return None


def _na_ratio(d: Dict) -> float:
    _NA = {"n/a","na","","none","null","insufficient data","not available",
           "data not available","unknown","—","see overall assessment"}
    total = na = 0
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            for _, vv in v.items():
                total += 1
                if str(vv).strip().lower() in _NA:
                    na += 1
        elif isinstance(v, str):
            total += 1
            if v.strip().lower() in _NA:
                na += 1
    return na / max(total, 1)


def _assess_patent_incentive(
    company: str,
    therapy_area: str,
    drugs: List[str],
    combined: str,
) -> Dict:
    prompt = _build_assessment_prompt(company, therapy_area, drugs, combined)
    try:
        resp   = _gemini.generate_content(prompt)
        parsed = _parse_json_response(resp.text or "")
        if parsed:
            ratio = _na_ratio(parsed)
            print(f"[STRATEGY] JSON parsed — {ratio:.0%} empty fields")
            if ratio < 0.4:
                return parsed
            print("[STRATEGY] Too many empty fields — trying direct Gemini research...")
    except Exception as e:
        print(f"[STRATEGY] Assessment failed: {e}")

    return _direct_gemini_report(company, therapy_area, drugs)


def _direct_gemini_report(company: str, therapy_area: str, drugs: List[str]) -> Dict:
    """Last-resort: ask Gemini to research the company itself with web search."""
    drug_list = ", ".join(drugs) if drugs else ""
    scope = (
        f"the entire company {company}"
        if therapy_area == "all therapeutic areas"
        else (f"{company}'s {therapy_area} portfolio, specifically {drug_list}"
              if drug_list else f"{company}'s {therapy_area} portfolio")
    )

    prompt = (
        f"You are a senior pharmaceutical patent strategy analyst. "
        f"Research {company} thoroughly using web search.\n\n"
        f"Write a comprehensive business strategy and patent expansion assessment for {scope}.\n\n"
        f"Cover: revenue, pipeline, strategic direction, indication expansion, "
        f"lifecycle management, competitive landscape, patent/IP strategy.\n"
        f"Never write 'insufficient data'. Provide estimates if exact numbers are unavailable.\n\n"
        f"Respond ONLY with valid JSON (no markdown):\n"
        f'{{"revenue_dependency":{{"rating":"","total_company_revenue":"","therapy_area_revenue":"",'
        f'"percentage":"","growth_rate":"","drug_breakdown":"","details":""}},'
        f'"pipeline_depth":{{"rating":"","assets":[],"recent_acquisitions":"","details":""}},'
        f'"strategic_priority":{{"rating":"","key_quotes":[],"restructuring":"","details":""}},'
        f'"indication_expansion":{{"rating":"","planned_indications":[],"patent_implications":"","details":""}},'
        f'"lifecycle_management":{{"rating":"","strategies":[],"new_formulations":"","new_devices":"",'
        f'"new_dosing":"","combinations":"","details":""}},'
        f'"competitive_pressure":{{"rating":"","competitors":[],"patent_expiry_dates":"",'
        f'"biosimilar_threats":"","defense_strategy":"","details":""}},'
        f'"patent_expansion_likelihood":{{"rating":"","formulation_patents":"","device_patents":"",'
        f'"dosing_regimen_patents":"","combination_patents":"","method_of_treatment_patents":"",'
        f'"manufacturing_patents":""}},"overall_assessment":""}}'
    )
    try:
        tool     = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        response = _gemini.generate_content(prompt, tools=[tool])
        parsed   = _parse_json_response(response.text or "")
        if parsed:
            print("[STRATEGY] Direct Gemini report succeeded")
            return parsed
    except Exception as e:
        print(f"[STRATEGY] Direct Gemini report failed: {e}")

    return {
        "overall_assessment": f"Assessment generation failed for {company}.",
        "revenue_dependency":          {"rating": "—", "details": "Failed"},
        "pipeline_depth":              {"rating": "—", "details": "Failed"},
        "strategic_priority":          {"rating": "—", "details": "Failed"},
        "indication_expansion":        {"rating": "—", "details": "Failed"},
        "lifecycle_management":        {"rating": "—", "details": "Failed"},
        "competitive_pressure":        {"rating": "—", "details": "Failed"},
        "patent_expansion_likelihood": {"rating": "—", "details": "Failed"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def review_business_strategy(
    company: str,
    therapy_area: Optional[str] = None,
    drugs: Optional[List[str]] = None,
    write_bq: bool = True,
) -> Dict:
    drugs        = drugs or []
    therapy_area = therapy_area or "all therapeutic areas"

    print(f"\n{'═'*70}")
    print(f"  Business Strategy Review: {company}")
    print(f"  Scope: {therapy_area}")
    if drugs:
        print(f"  Drugs: {', '.join(drugs)}")
    print(f"{'═'*70}\n")

    all_sources: List[str] = []

    # Source 1 — SEC EDGAR
    print("[1/3] SEC EDGAR...")
    sec = _fetch_sec_10k(company)
    if sec:
        all_sources.append(sec)

    # Source 2 — Vertex AI Search
    print("[2/3] Vertex AI Search...")
    vas = _vertex_search(company, therapy_area)
    if vas:
        all_sources.append(vas)

    # Source 3 — Gemini web search
    print("[3/3] Gemini web search (supplemental)...")
    sup = _search_supplemental(company, therapy_area, drugs)
    if sup:
        all_sources.append(sup)

    if not all_sources:
        print("[STRATEGY] No intelligence gathered.")
        return {"error": "No intelligence gathered"}

    # Cap each source to avoid one noisy blob dominating the context
    _MAX_PER_SOURCE = 60_000
    capped = [s[:_MAX_PER_SOURCE] for s in all_sources]

    combined = "\n\n" + ("=" * 60 + "\n\n").join(capped)
    print(f"\n[STRATEGY] Combined: {len(combined)} chars from {len(all_sources)} source(s)")

    print("[STRATEGY] Running LLM assessment...")
    assessment = _assess_patent_incentive(company, therapy_area, drugs, combined)

    assessment["_metadata"] = {
        "company":      company,
        "therapy_area": therapy_area,
        "drugs":        drugs,
        "date":         datetime.now(timezone.utc).date().isoformat(),
        "sources_used": [s.split("\n")[0] for s in all_sources],
    }

    if write_bq:
        _write_to_bq(assessment)

    return assessment


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_assessment(assessment: Dict) -> None:
    if "error" in assessment:
        print(f"\n[ERROR] {assessment['error']}")
        return

    meta = assessment.get("_metadata", {})
    print(f"\n{'═'*70}")
    print(f"  BUSINESS STRATEGY ASSESSMENT — {meta.get('company','?')}")
    print(f"  Therapy Area : {meta.get('therapy_area','?')}")
    print(f"  Date         : {meta.get('date','?')}")
    print(f"{'═'*70}\n")

    _SECTIONS = [
        ("REVENUE DEPENDENCY",          "revenue_dependency"),
        ("PIPELINE DEPTH",              "pipeline_depth"),
        ("STRATEGIC PRIORITY",          "strategic_priority"),
        ("INDICATION EXPANSION",        "indication_expansion"),
        ("LIFECYCLE MANAGEMENT",        "lifecycle_management"),
        ("COMPETITIVE PRESSURE",        "competitive_pressure"),
        ("PATENT EXPANSION LIKELIHOOD", "patent_expansion_likelihood"),
    ]
    for title, key in _SECTIONS:
        data = assessment.get(key, {})
        if not data:
            continue
        print(f"  {title}: {data.get('rating','—')}")
        for k, v in data.items():
            if k == "rating":
                continue
            if isinstance(v, list):
                print(f"    {k}: {', '.join(str(x) for x in v)}")
            elif v:
                print(f"    {k}: {v}")
        print()

    overall = assessment.get("overall_assessment", "")
    if overall:
        print(f"  {'─'*66}")
        print(f"  OVERALL: {overall}")
        print(f"  {'─'*66}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional Excel export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_excel(assessment: Dict, output_path: str) -> Optional[str]:
    try:
        import pandas as pd
    except ImportError:
        print("[EXPORT] pandas not installed — skipping Excel export")
        return None

    if "error" in assessment:
        return None

    meta = assessment.get("_metadata", {})
    rows = [
        {"Section": "Company",      "Content": meta.get("company", "")},
        {"Section": "Therapy Area", "Content": meta.get("therapy_area", "")},
        {"Section": "Drugs",        "Content": ", ".join(meta.get("drugs", []))},
        {"Section": "Date",         "Content": meta.get("date", "")},
        {"Section": "Sources",      "Content": "; ".join(meta.get("sources_used", []))},
        {"Section": "", "Content": ""},
    ]

    _SECTIONS = [
        ("revenue_dependency",         "REVENUE DEPENDENCY"),
        ("pipeline_depth",             "PIPELINE DEPTH"),
        ("strategic_priority",         "STRATEGIC PRIORITY"),
        ("indication_expansion",       "INDICATION EXPANSION"),
        ("lifecycle_management",       "LIFECYCLE MANAGEMENT"),
        ("competitive_pressure",       "COMPETITIVE PRESSURE"),
        ("patent_expansion_likelihood","PATENT EXPANSION LIKELIHOOD"),
    ]
    for key, title in _SECTIONS:
        data = assessment.get(key, {})
        if not data:
            continue
        rows.append({"Section": title, "Content": f"Rating: {data.get('rating','—')}"})
        for k, v in data.items():
            if k == "rating":
                continue
            rows.append({"Section": f"  {k}", "Content": (
                ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
            )})
        rows.append({"Section": "", "Content": ""})

    overall = assessment.get("overall_assessment", "")
    if overall:
        rows.append({"Section": "OVERALL ASSESSMENT", "Content": overall})

    try:
        import openpyxl
        import io as _io

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cog import gcs_cache

        df = pd.DataFrame(rows)
        buf = _io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Business Strategy")
            ws = writer.sheets["Business Strategy"]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value or "")) for c in col), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(width + 4, 120)
        buf.seek(0)

        filename = Path(output_path).name
        uri = gcs_cache.write_bytes(
            "company_analysis",
            filename,
            buf.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        print(f"[EXPORT] Saved to GCS: {uri}")
        return uri
    except Exception as e:
        print(f"[EXPORT] Failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch — fetch companies from BigQuery
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_companies(therapy: str = "GLP-1") -> List[str]:
    src_table = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{os.getenv('BQ_SOURCE_TABLE', 'patent_pipeline')}"
    query = f"""
    SELECT DISTINCT Parent_Company_Name
    FROM `{src_table}`
    WHERE
      ( UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
        OR UPPER(cleaned_Target) LIKE '%GLP-1%'
        OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
      )
      OR
      ( data_source = 'IPD'
        AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
      )
    ORDER BY Parent_Company_Name
    """
    try:
        df        = _bq_client.query(query).to_dataframe()
        companies = (
            df["Parent_Company_Name"]
            .dropna().astype(str).str.strip()
            .loc[lambda s: (s != "") & (s.str.lower() != "nan")]
            .unique().tolist()
        )
        companies = sorted(companies)
        print(f"[BQ] Found {len(companies)} companies in source table")
        return companies
    except Exception as e:
        print(f"[BQ] Source query failed: {e}")
        return []


async def _run_batch(companies: List[str], therapy_area: str, export: bool):

    succeeded, failed = 0, []
    for i, company in enumerate(companies, 1):
        print(f"\n{'█'*70}\n  [{i}/{len(companies)}] {company}\n{'█'*70}")
        try:
            loop       = asyncio.get_event_loop()
            assessment = await loop.run_in_executor(
                None,
                lambda c=company: review_business_strategy(
                    company=c, therapy_area=therapy_area, write_bq=True
                ),
            )
            print_assessment(assessment)
            if export:
                safe = company.lower().replace(" ", "_").replace("/", "_")
                export_to_excel(assessment, f"{safe}.xlsx")
            succeeded += 1
        except Exception as e:
            print(f"[ERROR] {company}: {e}")
            failed.append((company, str(e)))

    print(f"\n{'═'*70}")
    print(f"  Done — {succeeded}/{len(companies)} succeeded, {len(failed)} failed")
    if failed:
        for c, e in failed:
            print(f"  ✗ {c}: {e}")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Review business strategy and assess patent expansion incentive."
    )
    parser.add_argument("--company",  default=None, help="Single company name")
    parser.add_argument("--therapy",  default=None, help="Therapy area (e.g. 'GLP-1')")
    parser.add_argument("--drugs",    default=None, help="Comma-separated drug names")
    parser.add_argument("--export",   action="store_true", help="Export to Excel")
    parser.add_argument("--output",   default=None, help="Excel output path")
    parser.add_argument("--limit",    type=int, default=None, help="Batch: max companies")
    parser.add_argument("--dry-run",  action="store_true", help="Batch: list companies only")
    parser.add_argument("--no-bq",    action="store_true", help="Skip BigQuery write")
    args = parser.parse_args()

    drugs = [d.strip() for d in args.drugs.split(",")] if args.drugs else []

    if args.company:
        # ── Single company ────────────────────────────────────────────────────
        assessment = review_business_strategy(
            company      = args.company,
            therapy_area = args.therapy,
            drugs        = drugs,
            write_bq     = not args.no_bq,
        )
        print_assessment(assessment)
        if args.export:
            safe = args.company.lower().replace(" ", "_")
            export_to_excel(assessment, args.output or f"{safe}.xlsx")

    else:
        # ── Batch ─────────────────────────────────────────────────────────────
        companies = _fetch_all_companies(args.therapy or "GLP-1")
        if not companies:
            print("No companies found. Use --company or check BQ_SOURCE_TABLE config.")
            sys.exit(1)
        if args.limit:
            companies = companies[: args.limit]
        if args.dry_run:
            print(f"\n[DRY RUN] {len(companies)} companies:")
            for i, c in enumerate(companies, 1):
                print(f"  {i}. {c}")
            sys.exit(0)

        asyncio.run(_run_batch(companies, args.therapy or "GLP-1", args.export))
