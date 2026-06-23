"""
generate_report.py
──────────────────
Reads scored data from BigQuery tables produced by the Patent Legal Robustness Scorer
and generates a professional detailed PDF report using Gemini 2.5 Flash
for narrative generation and python-docx for document creation.

Input tables (BigQuery):
  - patent_strength_table
  - patent_strength_country_score_table

Usage:
    pip install python-docx pandas google-cloud-bigquery google-genai db-dtypes google-cloud-storage python-dotenv --break-system-packages
    # Create a .env file with:
    #   GEMINI_API_KEY=your-key
    #   GCS_CREDENTIALS=/path/to/service-account.json
    python generate_report.py --output report.pdf
"""

import os
import re
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not needed on Cloud Run

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, ListFlowable, ListItem,
)

from google import genai as genai_client
from google.genai import types

# ── Reportlab color palette ───────────────────────────────────────────────────
_NAVY       = colors.HexColor("#1F3864")
_BLUE       = colors.HexColor("#2F5496")
_LIGHT_BLUE = colors.HexColor("#D6E4F0")
_LGREY      = colors.HexColor("#F2F2F2")
_WHITE      = colors.white
_RED        = colors.HexColor("#CC0000")
_GREEN      = colors.HexColor("#008000")
_GREY       = colors.HexColor("#666666")

def _rl_styles():
    b = getSampleStyleSheet()
    return {
        "title":  ParagraphStyle("T",  parent=b["Title"],    fontSize=18, leading=22, textColor=_NAVY, spaceAfter=2),
        "meta":   ParagraphStyle("M",  parent=b["Normal"],   fontSize=9, leading=11, textColor=_GREY, spaceAfter=6),
        "h1":     ParagraphStyle("H1", parent=b["Heading2"], fontSize=13, leading=16, textColor=_NAVY, spaceBefore=8, spaceAfter=4),
        "h2":     ParagraphStyle("H2", parent=b["Heading3"], fontSize=11, leading=14, textColor=_NAVY, spaceBefore=6, spaceAfter=2),
        "h3":     ParagraphStyle("H3", parent=b["Heading4"], fontSize=10, leading=12, textColor=_BLUE, spaceBefore=4, spaceAfter=2),
        "body":   ParagraphStyle("BD", parent=b["Normal"],   fontSize=9, leading=12, alignment=TA_JUSTIFY, spaceAfter=4),
        "bullet": ParagraphStyle("BU", parent=b["Normal"],   fontSize=9, leading=12, spaceAfter=2, leftIndent=18, bulletIndent=6),
        "legend": ParagraphStyle("LG", parent=b["Normal"],   fontSize=7, leading=9, textColor=_GREY, spaceAfter=1),
        "th":     ParagraphStyle("TH", parent=b["Normal"],   fontSize=8, leading=10, textColor=_WHITE, alignment=TA_CENTER),
        "td":     ParagraphStyle("TD", parent=b["Normal"],   fontSize=8, leading=10, alignment=TA_CENTER),
        "tdl":    ParagraphStyle("TL", parent=b["Normal"],   fontSize=8, leading=10, alignment=TA_LEFT),
        "footer": ParagraphStyle("FT", parent=b["Normal"],   fontSize=7, leading=9, textColor=_GREY, alignment=TA_CENTER),
    }

_SCORE_CLR = {
    1: "#008000", 2: "#4CAF50", 3: "#CC9900", 4: "#E65100", 5: "#CC0000",
}

# ── Config ────────────────────────────────────────────────────────────────────
# Loaded from environment / .env file
API_KEY        = os.environ.get("GEMINI_API_KEY", "")
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

MODEL = "gemini-2.5-flash"

BQ_PROJECT_ID = "cognito-prod-394707"
BQ_DATASET_ID = "cognito_prod_datamart"
BQ_LOCATION   = "asia-south1"

BQ_STRENGTH_TABLE      = "patent_strength_table"
BQ_COUNTRY_SCORE_TABLE = "patent_strength_country_score_table"

# ── GCS Configuration ─────────────────────────────────────────────────────────
GCS_BUCKET    = "cognito-gcs"
GCS_BASE_PATH = "Cognito_new/reports"
GCS_FILE_NAME = "Patent_Strength_Analysis.pdf"

SCORE_LABEL = {
    1: "Very Robust",
    2: "Robust",
    3: "Moderate",
    4: "Vulnerable",
    5: "Highly Vulnerable",
}

REPORT_TITLE = "Patent Strength Scoring Analysis"


# ── Gemini helper ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str, retries: int = 5, backoff: float = 3.0) -> str:
    """Call Gemini 2.5 Flash with retry + exponential backoff for 429s."""
    import time as _time
    client = genai_client.Client(api_key=API_KEY)
    config = types.GenerateContentConfig(temperature=0.3)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower()
            if attempt < retries - 1 and is_rate_limit:
                wait = backoff * (2 ** attempt)
                print(f"  ⚠ Gemini 429 rate-limit, retrying in {wait:.0f}s (attempt {attempt+1}/{retries})…")
                _time.sleep(wait)
            elif attempt < retries - 1:
                wait = backoff * (attempt + 1)
                print(f"  ⚠ Gemini error ({e}), retrying in {wait:.0f}s…")
                _time.sleep(wait)
            else:
                print(f"  ✗ Gemini failed after {retries} attempts: {e}")
                return ""


def _extract_json(text: str):
    """Extract JSON from a Gemini response that may contain markdown fences."""
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None  # Use ADC (Application Default Credentials)


def _bq_client():
    credentials = _get_credentials()
    return bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )


def load_from_bigquery() -> dict:
    """Read scored data from BigQuery tables produced by the scorer."""
    client = _bq_client()
    data = {}

    # ── patent_strength_table ─────────────────────────────────────────────────
    print(f"Loading {BQ_STRENGTH_TABLE} from BigQuery...")
    try:
        q = f"SELECT DISTINCT * FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_STRENGTH_TABLE}`"
        df_final = client.query(q).to_dataframe()
        df_final = df_final.rename(columns={
            "drug_name":           "Drug Name",
            "patent_number":       "Patent Number",
            "patent_type":         "Patent Type",
            "jurisdiction":        "Jurisdiction",
            "country_weight":      "Country Weight",
            "core_inventive_step": "Core Inventive Step",
            "sf1_score":           "SF1 Score",
            "sf2_score":           "SF2 Score",
            "sf3_score":           "SF3 Score",
            "sf4_score":           "SF4 Score",
            "weighted_score":      "Weighted Final Score",
            "key_vulnerabilities": "Key Vulnerabilities",
            "key_strengths":       "Key Strengths",
            "filing_date":         "Filing Date",
            "grant_date":          "Grant Date",
            "sf1_label":           "SF1 Label",
            "sf1_score_reason":    "SF1 Score Reason",
            "sf1_reasoning":       "SF1 Reasoning",
            "sf1_key_finding_1":   "SF1 Key Finding 1",
            "sf1_key_finding_2":   "SF1 Key Finding 2",
            "sf1_key_finding_3":   "SF1 Key Finding 3",
            "sf1_chunks_used":     "SF1 Chunks Used",
            "sf2_label":           "SF2 Label",
            "sf2_score_reason":    "SF2 Score Reason",
            "sf2_reasoning":       "SF2 Reasoning",
            "sf2_key_finding_1":   "SF2 Key Finding 1",
            "sf2_key_finding_2":   "SF2 Key Finding 2",
            "sf2_key_finding_3":   "SF2 Key Finding 3",
            "sf2_chunks_used":     "SF2 Chunks Used",
            "sf3_label":           "SF3 Label",
            "sf3_score_reason":    "SF3 Score Reason",
            "sf3_reasoning":       "SF3 Reasoning",
            "sf3_key_finding_1":   "SF3 Key Finding 1",
            "sf3_key_finding_2":   "SF3 Key Finding 2",
            "sf3_key_finding_3":   "SF3 Key Finding 3",
            "sf3_chunks_used":     "SF3 Chunks Used",
            "sf4_label":           "SF4 Label",
            "sf4_score_reason":    "SF4 Score Reason",
            "sf4_reasoning":       "SF4 Reasoning",
            "sf4_key_finding_1":   "SF4 Key Finding 1",
            "sf4_key_finding_2":   "SF4 Key Finding 2",
            "sf4_key_finding_3":   "SF4 Key Finding 3",
            "sf4_chunks_used":     "SF4 Chunks Used",
            "created_at":          "Created At",
            "updated_at":          "Updated At",
        })
        data["final"] = df_final
        print(f"  Loaded {len(df_final)} rows.")
    except Exception as e:
        print(f"  WARNING: Could not load {BQ_STRENGTH_TABLE}: {e}")
        data["final"] = pd.DataFrame()

    # ── patent_strength_country_score_table ───────────────────────────────────
    print(f"Loading {BQ_COUNTRY_SCORE_TABLE} from BigQuery...")
    try:
        q = f"SELECT DISTINCT * FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_COUNTRY_SCORE_TABLE}`"
        df_country = client.query(q).to_dataframe()
        df_country = df_country.rename(columns={
            "drug_name":              "Drug Name",
            "jurisdiction":           "Jurisdiction",
            "country_name":           "Country Name",
            "country_weight":         "Country Weight",
            "patent_count":           "# Patents",
            "avg_weighted_score":     "Avg Weighted Score",
            "country_weighted_score": "Country Weighted Score",
            "final_patent_score":     "Final Patent Score (Drug Total)",
        })
        data["country_scores"] = df_country
        print(f"  Loaded {len(df_country)} rows.")
    except Exception as e:
        print(f"  WARNING: Could not load {BQ_COUNTRY_SCORE_TABLE}: {e}")
        data["country_scores"] = pd.DataFrame()

    return data


def compute_statistics(df: pd.DataFrame) -> dict:  # noqa: C901
    """Compute summary statistics from the Final Output sheet."""
    stats = {
        "total_patents": len(df),
        "drugs": [],
        "avg_weighted": None,
        "score_distribution": {},
        "most_vulnerable": [],
        "most_robust": [],
        "per_drug_stats": {},
        "highest_score_patent": None,
        "highest_score_per_jurisdiction": {},
    }
    if df.empty:
        return stats

    stats["drugs"] = [str(d) for d in df["Drug Name"].dropna().unique().tolist()]

    ws_col = "Weighted Final Score"
    if ws_col in df.columns:
        numeric = pd.to_numeric(df[ws_col], errors="coerce").dropna()
        if len(numeric):
            stats["avg_weighted"] = float(round(numeric.mean(), 2))
            for s in range(1, 6):
                stats["score_distribution"][s] = int((numeric.round() == s).sum())
            top_vuln = df.loc[numeric.nlargest(3).index]
            stats["most_vulnerable"] = top_vuln[["Drug Name", "Patent Number", ws_col]].to_dict("records")
            top_rob = df.loc[numeric.nsmallest(3).index]
            stats["most_robust"] = top_rob[["Drug Name", "Patent Number", ws_col]].to_dict("records")

            highest_idx = numeric.idxmax()
            highest_row = df.loc[highest_idx]
            stats["highest_score_patent"] = {
                "drug": str(highest_row.get("Drug Name", "N/A")),
                "patent_number": str(highest_row.get("Patent Number", "N/A")),
                "patent_type": str(highest_row.get("Patent Type", "N/A")),
                "jurisdiction": str(highest_row.get("Jurisdiction", "N/A")),
                "weighted_score": float(numeric[highest_idx]),
                "sf1": highest_row.get("SF1 Score", "N/A"),
                "sf2": highest_row.get("SF2 Score", "N/A"),
                "sf3": highest_row.get("SF3 Score", "N/A"),
                "sf4": highest_row.get("SF4 Score", "N/A"),
                "core_step": str(highest_row.get("Core Inventive Step", "N/A")),
                "vulnerabilities": str(highest_row.get("Key Vulnerabilities", "N/A")),
                "strengths": str(highest_row.get("Key Strengths", "N/A")),
            }

            if "Jurisdiction" in df.columns:
                per_jurisdiction = {}
                for jur in df["Jurisdiction"].dropna().unique():
                    jur_df = df[df["Jurisdiction"] == jur]
                    jur_numeric = pd.to_numeric(jur_df[ws_col], errors="coerce")
                    jur_valid = jur_numeric.dropna()
                    if len(jur_valid):
                        jur_highest_idx = jur_valid.idxmax()
                        jur_row = df.loc[jur_highest_idx]
                        per_jurisdiction[str(jur)] = {
                            "drug": str(jur_row.get("Drug Name", "N/A")),
                            "patent_number": str(jur_row.get("Patent Number", "N/A")),
                            "patent_type": str(jur_row.get("Patent Type", "N/A")),
                            "jurisdiction": str(jur),
                            "weighted_score": float(jur_valid[jur_highest_idx]),
                            "sf1": jur_row.get("SF1 Score", "N/A"),
                            "sf2": jur_row.get("SF2 Score", "N/A"),
                            "sf3": jur_row.get("SF3 Score", "N/A"),
                            "sf4": jur_row.get("SF4 Score", "N/A"),
                            "core_step": str(jur_row.get("Core Inventive Step", "N/A")),
                            "vulnerabilities": str(jur_row.get("Key Vulnerabilities", "N/A")),
                            "strengths": str(jur_row.get("Key Strengths", "N/A")),
                        }
                stats["highest_score_per_jurisdiction"] = per_jurisdiction

        for drug in stats["drugs"]:
            drug_df = df[df["Drug Name"] == drug]
            drug_numeric = pd.to_numeric(drug_df[ws_col], errors="coerce").dropna()
            if len(drug_numeric):
                stats["per_drug_stats"][str(drug)] = {
                    "count": int(len(drug_df)),
                    "avg_score": float(round(drug_numeric.mean(), 2)),
                    "min_score": float(round(drug_numeric.min(), 2)),
                    "max_score": float(round(drug_numeric.max(), 2)),
                }

    for sf_col in ["SF1 Score", "SF2 Score", "SF3 Score", "SF4 Score"]:
        if sf_col in df.columns:
            sf_numeric = pd.to_numeric(df[sf_col], errors="coerce").dropna()
            if len(sf_numeric):
                stats[f"avg_{sf_col.lower().replace(' ', '_')}"] = float(round(sf_numeric.mean(), 2))

    return stats


# ── Country score statistics ──────────────────────────────────────────────────

def compute_country_statistics(df_country: pd.DataFrame) -> dict:
    """Derive summary data from patent_strength_country_score_table."""
    result = {
        "by_drug": {},
        "final_scores": {},
        "all_jurisdictions": [],
    }
    if df_country.empty:
        return result

    result["all_jurisdictions"] = sorted(df_country["Jurisdiction"].dropna().unique().tolist())

    for drug, grp in df_country.groupby("Drug Name"):
        jur_map = {}
        final_score = None
        for _, row in grp.iterrows():
            jur = str(row.get("Jurisdiction", ""))
            jur_map[jur] = {
                "country_name":           str(row.get("Country Name", "")),
                "country_weight":         row.get("Country Weight"),
                "patent_count":           row.get("# Patents"),
                "avg_weighted_score":     row.get("Avg Weighted Score"),
                "country_weighted_score": row.get("Country Weighted Score"),
                "final_patent_score":     row.get("Final Patent Score (Drug Total)"),
            }
            if final_score is None:
                final_score = row.get("Final Patent Score (Drug Total)")
        result["by_drug"][str(drug)] = jur_map
        result["final_scores"][str(drug)] = final_score

    return result


# ── LLM narrative generation ──────────────────────────────────────────────────

def generate_executive_summary(stats: dict, df_final: pd.DataFrame, country_stats: dict = None) -> dict:
    """Use Gemini to produce a detailed executive summary and findings narrative."""
    patent_rows = []
    for _, r in df_final.iterrows():
        patent_rows.append({
            "drug": str(r.get("Drug Name", "")),
            "patent": str(r.get("Patent Number", "")),
            "type": str(r.get("Patent Type", "")),
            "jurisdiction": str(r.get("Jurisdiction", "N/A")),
            "sf1": r.get("SF1 Score", "N/A"),
            "sf2": r.get("SF2 Score", "N/A"),
            "sf3": r.get("SF3 Score", "N/A"),
            "sf4": r.get("SF4 Score", "N/A"),
            "weighted": r.get("Weighted Final Score", "N/A"),
            "core_step": str(r.get("Core Inventive Step", ""))[:300],
            "vulnerabilities": str(r.get("Key Vulnerabilities", ""))[:300],
            "strengths": str(r.get("Key Strengths", ""))[:300],
        })

    per_drug_json   = json.dumps(stats.get('per_drug_stats') or {})
    score_dist_json = json.dumps(stats.get('score_distribution') or {})

    hsp = stats.get("highest_score_patent") or {}
    hsp_summary = (
        f"Highest weighted score patent: {hsp.get('patent_number', 'N/A')} "
        f"(Drug: {hsp.get('drug', 'N/A')}, Jurisdiction: {hsp.get('jurisdiction', 'N/A')}, "
        f"Score: {hsp.get('weighted_score', 'N/A')}). "
        f"Vulnerabilities: {hsp.get('vulnerabilities', 'N/A')[:300]}"
        if hsp else "N/A"
    )

    hspj = stats.get("highest_score_per_jurisdiction") or {}
    hspj_summary = "; ".join(
        f"{jur}: {v.get('patent_number', 'N/A')} (Score: {v.get('weighted_score', 'N/A')})"
        for jur, v in hspj.items()
    ) if hspj else "N/A"

    country_context = "N/A"
    if country_stats and country_stats.get("by_drug"):
        lines = []
        for drug, jur_map in country_stats["by_drug"].items():
            final = country_stats["final_scores"].get(drug, "N/A")
            lines.append(f"Drug: {drug}  |  Final Patent Score: {final}")
            for jur, jdata in sorted(jur_map.items(), key=lambda x: -(x[1].get("country_weight") or 0)):
                lines.append(
                    f"  {jur} ({jdata['country_name']}): weight={jdata['country_weight']}, "
                    f"patents={jdata['patent_count']}, avg_weighted={jdata['avg_weighted_score']}, "
                    f"country_weighted={jdata['country_weighted_score']}"
                )
        country_context = "\n".join(lines)

    prompt = f"""You are a senior pharmaceutical patent analyst writing a detailed report.
Based on the data below, produce a thorough analytical report. Be specific — reference
patent numbers, drug names, jurisdictions, and actual scores throughout your analysis.

PORTFOLIO STATISTICS:
- Total blocking patents analysed: {stats['total_patents']}
- Drugs covered: {', '.join(stats['drugs'])}
- Average weighted robustness score: {stats['avg_weighted']} (1 = Very Robust, 5 = Highly Vulnerable)
- Score distribution: {score_dist_json}
- Per-drug averages: {per_drug_json}
- Sub-factor averages: SF1 (Novelty)={stats.get('avg_sf1_score', 'N/A')}, SF2 (Obvious-to-Combine)={stats.get('avg_sf2_score', 'N/A')}, SF3 (Prosecution History)={stats.get('avg_sf3_score', 'N/A')}, SF4 (Secondary Considerations)={stats.get('avg_sf4_score', 'N/A')}
- {hsp_summary}
- Highest score patent per jurisdiction: {hspj_summary}

COUNTRY-WEIGHTED SCORES (from patent_strength_country_score_table):
{country_context}

PATENT-LEVEL DATA (JSON):
{json.dumps(patent_rows, indent=1)}

Respond ONLY with a valid JSON object (no markdown fences):
{{
  "executive_summary": "<5-8 sentences providing a thorough overview of the portfolio's legal robustness, mentioning specific drugs and overall risk posture>",
  "key_findings": ["<detailed finding 1 with patent numbers>", "<detailed finding 2>", "<detailed finding 3>", "<detailed finding 4>", "<detailed finding 5>", "<detailed finding 6>"],
  "risk_highlights": "<4-6 sentences detailing the highest-risk patents, their specific vulnerabilities, and why they are exposed to challenge. Reference patent numbers and scores.>",
  "strength_highlights": "<4-6 sentences detailing the most robust patents, their defensive strengths, and what makes them resilient. Reference patent numbers and scores.>",
  "sf_analysis": "<4-6 sentences analysing sub-factor trends across the portfolio. Which sub-factor is weakest/strongest overall? Are there patterns by drug or patent type?>",
  "highest_score_narrative": "<4-6 sentences providing an in-depth analysis of the patent with the highest weighted score: why it scores so high, its core inventive weaknesses, the specific legal risks it faces, and strategic implications for challengers or defenders.>",
  "country_score_narrative": "<3-5 sentences analysing the country-weighted final patent scores per drug: which jurisdictions drive the highest risk, how country weights affect the overall score, and strategic geographic implications>",
  "per_drug_narratives": {{
    "<drug_name>": "<3-5 sentence analysis of this drug's patent portfolio robustness, key risks, and protective strengths>"
  }}
}}
"""
    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result
    return {
        "executive_summary": "Analysis complete. See tables below for details.",
        "key_findings": ["See detailed tables."],
        "risk_highlights": "Refer to score tables.",
        "strength_highlights": "Refer to score tables.",
        "sf_analysis": "See sub-factor scores in the table.",
        "highest_score_narrative": "See score tables for details on the highest-scoring patent.",
        "country_score_narrative": "See country-weighted score table for details.",
        "per_drug_narratives": {},
    }


def apply_rationale_column(df: pd.DataFrame, per_drug_narratives: dict) -> pd.DataFrame:
    """
    Stamp each row of df with the Drug-Level Analysis narrative for its drug.

    The narrative written under Drug-Level Analysis for a given drug is inserted
    into a new 'rationale' column for every row belonging to that drug.
    """
    df = df.copy()
    df["rationale"] = ""
    for drug_name, narrative_text in per_drug_narratives.items():
        mask = df["Drug Name"] == drug_name
        df.loc[mask, "rationale"] = narrative_text
    return df


def ensure_rationale_column(client: bigquery.Client) -> None:
    """
    Add a STRING 'rationale' column to patent_strength_table if it doesn't
    already exist. BigQuery ALTER TABLE ADD COLUMN is idempotent-safe when
    the column is absent, but raises an error if it already exists, so we
    check the schema first.
    """
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_STRENGTH_TABLE}"
    table = client.get_table(table_ref)
    existing_cols = {field.name for field in table.schema}

    if "rationale" in existing_cols:
        print("  'rationale' column already exists in BigQuery table — skipping ALTER.")
        return

    print("  Adding 'rationale' column to BigQuery table...")
    alter_sql = f"ALTER TABLE `{table_ref}` ADD COLUMN rationale STRING"
    client.query(alter_sql).result()
    print("  Column added successfully.")


def write_rationale_to_bigquery(per_drug_narratives: dict) -> None:
    """
    Persist the drug-level rationale text back to patent_strength_table in
    BigQuery.  One parameterised UPDATE is issued per drug so only the rows
    for that drug are touched.

    Steps:
      1. Ensure the 'rationale' column exists (ALTER TABLE if needed).
      2. Run a parameterised UPDATE … WHERE drug_name = @drug_name for each drug.
    """
    if not per_drug_narratives:
        print("  No per-drug narratives to write — skipping BigQuery update.")
        return

    client    = _bq_client()
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_STRENGTH_TABLE}"

    # Step 1 — guarantee the column exists
    ensure_rationale_column(client)

    # Step 2 — update each drug's rows
    print(f"  Writing rationale to BigQuery for {len(per_drug_narratives)} drug(s)...")
    for drug_name, narrative_text in per_drug_narratives.items():
        sql = f"""
            UPDATE `{table_ref}`
            SET rationale = @rationale
            WHERE drug_name = @drug_name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("rationale",  "STRING", narrative_text),
                bigquery.ScalarQueryParameter("drug_name",  "STRING", drug_name),
            ]
        )
        try:
            client.query(sql, job_config=job_config).result()
            print(f"    ✅ rationale written for drug: {drug_name}")
        except Exception as e:
            print(f"    [ERROR] Failed to update rationale for drug '{drug_name}': {e}")
            raise

    print("  BigQuery rationale update complete.")


def ensure_timestamp_columns(client: bigquery.Client) -> None:
    """
    Add TIMESTAMP columns 'created_at' and 'updated_at' to patent_strength_table
    if they don't already exist.

    - created_at: set only on the first run (when the column value is NULL).
    - updated_at: overwritten on every subsequent run.
    """
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_STRENGTH_TABLE}"
    table = client.get_table(table_ref)
    existing_cols = {field.name for field in table.schema}

    for col in ("created_at", "updated_at"):
        if col in existing_cols:
            print(f"  '{col}' column already exists in BigQuery table — skipping ALTER.")
        else:
            print(f"  Adding '{col}' column to BigQuery table...")
            alter_sql = f"ALTER TABLE `{table_ref}` ADD COLUMN {col} TIMESTAMP"
            client.query(alter_sql).result()
            print(f"  '{col}' column added successfully.")


def write_timestamps_to_bigquery(drug_names: list, ts: datetime = None) -> None:
    """
    Write created_at and updated_at timestamps to patent_strength_table.

    - created_at: written only when the existing value is NULL (i.e. first run).
                  Existing values are preserved on subsequent runs.
    - updated_at: always overwritten with the current UTC time on every run.

    Steps:
      1. Ensure both timestamp columns exist (ALTER TABLE if needed).
      2. Run parameterised UPDATEs per drug.
    """
    if not drug_names:
        print("  No drug names provided — skipping timestamp update.")
        return

    if ts is None:
        ts = datetime.now(tz=timezone.utc)

    client    = _bq_client()
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_STRENGTH_TABLE}"

    ensure_timestamp_columns(client)

    print(f"  Writing timestamps ({ts.isoformat()}) to BigQuery for {len(drug_names)} drug(s)...")
    for drug_name in drug_names:
        # created_at — only set when currently NULL (preserves the original first-run value)
        sql_created = f"""
            UPDATE `{table_ref}`
            SET created_at = @ts
            WHERE drug_name = @drug_name
              AND created_at IS NULL
        """
        # updated_at — always written to reflect the current run time
        sql_updated = f"""
            UPDATE `{table_ref}`
            SET updated_at = @ts
            WHERE drug_name = @drug_name
        """
        job_cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ts",        "TIMESTAMP", ts),
                bigquery.ScalarQueryParameter("drug_name", "STRING",    drug_name),
            ]
        )
        try:
            client.query(sql_created, job_config=job_cfg).result()
            client.query(sql_updated, job_config=job_cfg).result()
            print(f"    ✅ timestamps written for drug: {drug_name}")
        except Exception as e:
            print(f"    [ERROR] Failed to update timestamps for drug '{drug_name}': {e}")
            raise

    print("  BigQuery timestamp update complete.")


# ── Document builder ──────────────────────────────────────────────────────────


# ── Reportlab rendering helpers ───────────────────────────────────────────────

_SCORE_CLR = {
    1: "#008000", 2: "#4CAF50", 3: "#CC9900", 4: "#E65100", 5: "#CC0000",
}

def _hsp_table(story, hsp: dict, st: dict, W: float):
    """Render a Highest-Score-Patent info box as a reportlab Table."""
    try:
        score_label = SCORE_LABEL.get(int(round(float(hsp.get("weighted_score", 0)))), "N/A")
    except (ValueError, TypeError):
        score_label = "N/A"
    items = [
        ("Patent Number",                  hsp.get("patent_number", "N/A")),
        ("Drug Name",                       hsp.get("drug", "N/A")),
        ("Patent Type",                     hsp.get("patent_type", "N/A")),
        ("Jurisdiction",                    hsp.get("jurisdiction", "N/A")),
        ("Weighted Final Score",            f"{hsp.get('weighted_score', 'N/A')} / 5.0 — {score_label}"),
        ("SF1 (Novelty)",                   str(hsp.get("sf1", "N/A"))),
        ("SF2 (Obvious-to-Combine)",        str(hsp.get("sf2", "N/A"))),
        ("SF3 (Prosecution History)",       str(hsp.get("sf3", "N/A"))),
        ("SF4 (Secondary Considerations)",  str(hsp.get("sf4", "N/A"))),
    ]
    rows = [[Paragraph(f"<b>{lbl}</b>", st["tdl"]), Paragraph(val, st["td"])] for lbl, val in items]
    t = Table(rows, colWidths=[W * 0.38, W * 0.62])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), _LGREY),
        ("BOX",         (0, 0), (-1, -1), 0.5, _BLUE),
        ("INNERGRID",   (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def _rl_data_table(headers, data_rows, col_widths, st, score_col_indices=None):
    """Build a reportlab Table with navy header row and optional score coloring."""
    score_col_indices = score_col_indices or []
    tbl_rows = [[Paragraph(f"<b>{h}</b>", st["th"]) for h in headers]]
    for dr in data_rows:
        row = []
        for ci, val in enumerate(dr):
            s = str(val) if pd.notna(val) and str(val) not in ("nan", "None") else "N/A"
            if ci in score_col_indices:
                try:
                    sc = int(round(float(val)))
                    clr = _SCORE_CLR.get(sc, "#333333")
                    row.append(Paragraph(f'<font color="{clr}"><b>{s}</b></font>', st["td"]))
                    continue
                except (ValueError, TypeError):
                    pass
            row.append(Paragraph(s, st["td"]))
        tbl_rows.append(row)
    t = Table(tbl_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.5, _BLUE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LGREY]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def build_report(data: dict, output_path: str):
    """Build the PDF report directly using reportlab."""
    from reportlab.lib.pagesizes import letter as _letter
    df_final = data["final"]
    if df_final.empty:
        print("ERROR: patent_strength_table is empty or missing.")
        return

    df_country    = data.get("country_scores", pd.DataFrame())
    stats         = compute_statistics(df_final)
    country_stats = compute_country_statistics(df_country)

    print("Generating narrative with Gemini 2.5 Flash...")
    narrative = generate_executive_summary(stats, df_final, country_stats)

    per_drug_narratives = narrative.get("per_drug_narratives", {})
    df_final = apply_rationale_column(df_final, per_drug_narratives)
    data["final"] = df_final

    print("Writing rationale back to BigQuery...")
    write_rationale_to_bigquery(per_drug_narratives)
    print("Writing timestamps back to BigQuery...")
    drug_names_list = df_final["Drug Name"].dropna().unique().tolist() if not df_final.empty else []
    write_timestamps_to_bigquery(drug_names_list)

    st  = _rl_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=_letter,
        topMargin=18*mm, bottomMargin=14*mm, leftMargin=20*mm, rightMargin=20*mm,
        title=REPORT_TITLE, author="ADK Pipeline",
    )
    W     = _letter[0] - 40*mm
    story = []

    # ── Title ──
    story.append(Paragraph(REPORT_TITLE, st["title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}&nbsp;&nbsp;•&nbsp;&nbsp;"
        f"{stats['total_patents']} Blocking Patents&nbsp;&nbsp;•&nbsp;&nbsp;"
        f"{len(stats['drugs'])} Drug(s)", st["meta"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_NAVY, spaceAfter=8))

    # ── Executive Summary ──
    story.append(Paragraph("Executive Summary", st["h1"]))
    story.append(Paragraph(narrative.get("executive_summary", ""), st["body"]))
    story.append(Spacer(1, 6))

    # ── Portfolio Overview ──
    story.append(Paragraph("Portfolio Overview", st["h1"]))
    ov_items = [
        ("Total Patents Analysed", str(stats["total_patents"])),
        ("Drugs Covered",          ", ".join(stats["drugs"])),
        ("Average Weighted Score", f"{stats['avg_weighted']} / 5.0" if stats["avg_weighted"] else "N/A"),
    ]
    for key, label in [("avg_sf1_score","Avg SF1 (Novelty)"),("avg_sf2_score","Avg SF2 (Obvious-to-Combine)"),
                       ("avg_sf3_score","Avg SF3 (Prosecution History)"),("avg_sf4_score","Avg SF4 (Secondary Considerations)")]:
        val = stats.get(key)
        if val is not None:
            ov_items.append((label, f"{val} / 5.0"))
    dist_parts = [f"{SCORE_LABEL[s]}: {stats['score_distribution'].get(s,0)}" for s in range(1,6) if stats["score_distribution"].get(s,0) > 0]
    if dist_parts:
        ov_items.append(("Score Distribution", "; ".join(dist_parts)))
    ov_rows = [[Paragraph(f"<b>{lbl}</b>", st["tdl"]), Paragraph(val, st["td"])] for lbl, val in ov_items]
    ov_t = Table(ov_rows, colWidths=[W*0.38, W*0.62])
    ov_t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#E8EDF3")),("BOX",(0,0),(-1,-1),0.5,_BLUE),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.lightgrey),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(ov_t)
    story.append(Spacer(1, 8))

    # ── Key Findings ──
    story.append(Paragraph("Key Findings", st["h1"]))
    for finding in narrative.get("key_findings", []):
        story.append(Paragraph(f"&bull; {finding}", st["bullet"]))
    story.append(Spacer(1, 6))

    # ── Patent Score Summary Table ──
    story.append(Paragraph("Patent Score Summary", st["h1"]))
    tbl_cols = ["Drug Name","Patent Number","SF1 Score","SF2 Score","SF3 Score","SF4 Score","Weighted Final Score"]
    tbl_disp = ["Drug","Patent No.","SF1","SF2","SF3","SF4","Final"]
    cw = [W*0.19,W*0.19,W*0.1,W*0.1,W*0.1,W*0.1,W*0.12]
    df_table = df_final.copy()
    if "Weighted Final Score" in df_table.columns:
        df_table["_ws"] = pd.to_numeric(df_table["Weighted Final Score"], errors="coerce")
        df_table = df_table.dropna(subset=["_ws"]).drop(columns=["_ws"])
    dr = [[row.get(c,"") for c in tbl_cols] for _,row in df_table.iterrows()]
    story.append(_rl_data_table(tbl_disp, dr, cw, st, score_col_indices=list(range(2,7))))
    story.append(Spacer(1, 8))

    # ── Highest Weighted Score Patent ──
    hsp = stats.get("highest_score_patent")
    if hsp:
        story.append(Paragraph('<font color="#CC0000">Highest Weighted Score Patent</font>', st["h1"]))
        _hsp_table(story, hsp, st, W)
        if hsp.get("vulnerabilities") and hsp["vulnerabilities"] != "N/A":
            story.append(Paragraph(f'<font color="#CC0000"><b>Key Vulnerabilities:</b></font> {hsp["vulnerabilities"]}', st["body"]))
        if hsp.get("strengths") and hsp["strengths"] != "N/A":
            story.append(Paragraph(f'<font color="#008000"><b>Key Strengths:</b></font> {hsp["strengths"]}', st["body"]))
        if hsp.get("core_step") and hsp["core_step"] != "N/A":
            story.append(Paragraph(f'<b>Core Inventive Step:</b> {hsp["core_step"]}', st["body"]))
        hn = narrative.get("highest_score_narrative", "")
        if hn:
            story.append(Paragraph("<b>In-Depth Analysis</b>", st["h2"]))
            story.append(Paragraph(hn, st["body"]))
        story.append(Spacer(1, 8))

    # ── Highest Score per Jurisdiction ──
    hspj = stats.get("highest_score_per_jurisdiction") or {}
    if hspj:
        story.append(Paragraph('<font color="#CC0000">Highest Weighted Score Patent by Jurisdiction</font>', st["h1"]))
        story.append(Paragraph("The table below identifies the most legally vulnerable patent within each jurisdiction.", st["body"]))
        for jur, jur_hsp in hspj.items():
            story.append(Paragraph(f"Jurisdiction: {jur}", st["h3"]))
            _hsp_table(story, jur_hsp, st, W)

    # ── Country-Weighted Scores ──
    if country_stats and country_stats.get("by_drug"):
        story.append(PageBreak())
        story.append(Paragraph("Country-Weighted Patent Scores", st["h1"]))
        story.append(Paragraph("Jurisdiction-level weighted scores from patent_strength_country_score_table.", st["body"]))
        ct_h = ["Jurisdiction","Country","Weight","# Patents","Avg Weighted Score","Country Weighted Score"]
        ct_cw = [W*0.12,W*0.22,W*0.1,W*0.1,W*0.2,W*0.22]
        for drug, jur_map in country_stats["by_drug"].items():
            story.append(Paragraph(drug, st["h3"]))
            fs = country_stats["final_scores"].get(drug)
            if fs is not None:
                story.append(Paragraph(f'<b>Final Patent Score (Drug Total): {round(float(fs),4)}</b>', st["body"]))
            ct_rows = []
            for jur, jd in sorted(jur_map.items(), key=lambda x:-(x[1].get("country_weight") or 0)):
                ct_rows.append([jur, str(jd.get("country_name","")), str(jd.get("country_weight","")),
                    str(jd.get("patent_count","")), str(jd.get("avg_weighted_score","N/A")),
                    str(round(float(jd["country_weighted_score"]),4)) if jd.get("country_weighted_score") is not None else "N/A"])
            story.append(_rl_data_table(ct_h, ct_rows, ct_cw, st, score_col_indices=[4]))
            story.append(Spacer(1, 6))
        csn = narrative.get("country_score_narrative", "")
        if csn:
            story.append(Paragraph("<b>Geographic Risk Analysis</b>", st["h2"]))
            story.append(Paragraph(csn, st["body"]))
        story.append(Spacer(1, 8))

    # ── Sub-Factor Analysis ──
    sf_text = narrative.get("sf_analysis", "")
    if sf_text:
        story.append(Paragraph("Sub-Factor Analysis", st["h1"]))
        story.append(Paragraph(sf_text, st["body"]))

    # ── Risk & Strength ──
    story.append(Paragraph("Risk &amp; Strength Analysis", st["h1"]))
    story.append(Paragraph('<font color="#CC0000"><b>Highest Risk Patents</b></font>', st["h2"]))
    story.append(Paragraph(narrative.get("risk_highlights", "N/A"), st["body"]))
    story.append(Paragraph('<font color="#008000"><b>Most Robust Patents</b></font>', st["h2"]))
    story.append(Paragraph(narrative.get("strength_highlights", "N/A"), st["body"]))

    # ── Per-Drug Breakdown ──
    per_drug = narrative.get("per_drug_narratives", {})
    if per_drug:
        story.append(PageBreak())
        story.append(Paragraph("Drug-Level Analysis", st["h1"]))
        for dn, dn_nar in per_drug.items():
            story.append(Paragraph(dn, st["h3"]))
            ds = stats.get("per_drug_stats",{}).get(dn,{})
            if ds:
                story.append(Paragraph(f'<i><font color="#666666">Patents: {ds.get("count","N/A")} | Avg: {ds.get("avg_score","N/A")} | Range: {ds.get("min_score","N/A")}–{ds.get("max_score","N/A")}</font></i>', st["body"]))
            story.append(Paragraph(dn_nar, st["body"]))
            story.append(Spacer(1, 4))

    # ── Sub-Factor Framework Table ──
    story.append(Paragraph("Sub-Factor Scoring Framework", st["h1"]))
    sf_data = [("1","Novelty &amp; Non-Obviousness","Closeness of claimed invention to prior art.","40%"),
               ("2","Obvious-to-Combine Risk","Likelihood of combining known elements with reasonable expectation of success.","30%"),
               ("3","Prosecution History Vulnerability","Extent of claim narrowing during prosecution.","20%"),
               ("4","Secondary Considerations","Objective evidence supporting non-obviousness.","10%")]
    story.append(_rl_data_table(["SF #","Name","Description","Weight"], sf_data, [W*0.07,W*0.23,W*0.55,W*0.1], st))
    story.append(Spacer(1, 8))

    legend = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(f'<font color="#666666"><b>Score Legend:</b> {legend}</font>', st["legend"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>This report was auto-generated from Patent Legal Robustness Scorer output using Gemini 2.5 Flash.</i>", st["footer"]))

    doc.build(story)
    print(f"\n✅ PDF report saved → {output_path}")

def upload_to_gcs(local_path: str, drug_names: list) -> list:
    """
    Upload the generated .pdf to GCS under each drug's folder.

    Destination path per drug:
        gs://cognito-gcs/Cognito_new/reports/{drug_name}/Patent_Strength_Analysis.pdf

    Uses service account file or ADC (Cloud Run).
    Requires: pip install google-cloud-storage
    """
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required.\n"
            "Run: pip install google-cloud-storage"
        )

    credentials = _get_credentials()
    client   = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket   = client.bucket(GCS_BUCKET)
    gcs_uris = []

    for drug_name in drug_names:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(drug_name))
        blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_FILE_NAME}"
        gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"

        print(f"  Uploading to GCS: {gcs_uri}")
        try:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(
                local_path,
                content_type="application/pdf",
            )
            print(f"  Upload successful: {gcs_uri}")
            gcs_uris.append(gcs_uri)
        except Exception as e:
            print(f"  [ERROR] GCS upload failed for drug '{drug_name}': {e}")
            raise

    return gcs_uris


# ── Main ──────────────────────────────────────────────────────────────────────



def main():
    parser = argparse.ArgumentParser(
        description="Generate a detailed PDF report from BigQuery patent scorer tables"
    )
    parser.add_argument("--output", "-o", default="patent_robustness_report.pdf",
                        help="Output .pdf path (default: patent_robustness_report.pdf)")
    args = parser.parse_args()

    if not API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=your-key"
        )
    data = load_from_bigquery()
    build_report(data, args.output)

    # ── Upload to GCS ─────────────────────────────────────────────────────────
    df_final   = data.get("final", pd.DataFrame())
    drug_names = df_final["Drug Name"].dropna().unique().tolist() if not df_final.empty else []

    if drug_names:
        print(f"\nUploading report to GCS for {len(drug_names)} drug(s)...")
        gcs_uris = upload_to_gcs(args.output, drug_names)

        print(f"\n{'='*60}")
        print("  GCS Upload Summary:")
        for uri in gcs_uris:
            print(f"    {uri}")
        print("=" * 60)
    else:
        print("\n[WARN] No drug names found — skipping GCS upload.")


if __name__ == "__main__":
    main()
