#!/usr/bin/env python3
"""
litigation_report_generator.py
───────────────────────────────
Generates a detailed, neatly formatted 2-page pharmaceutical litigation
analysis PDF by reading directly from BigQuery `litigation_analysis_table`.

Page 1: Executive Summary + Innovator/Company Analysis (with web search)
Page 2: Per-Drug Litigation Analysis + Key Risks & Strategic Outlook

Usage:
    # All drugs in the table
    python litigation_report_generator.py --output ./out/report.pdf

    # Specific drugs only
    python litigation_report_generator.py --drugs semaglutide tirzepatide --output ./out/report.pdf

    # Latest run only (most recent loaded_at per drug)
    python litigation_report_generator.py --latest-only --output ./out/report.pdf
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment / .env — Gemini calls will fail")
    client = None
else:
    client   = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# ── BigQuery ───────────────────────────────────────────────────────────────────
BQ_PROJECT_ID     = os.getenv("BQ_PROJECT_ID",     os.getenv("PROJECT_ID", "cognito-prod-394707"))
BQ_DATASET        = os.getenv("BQ_DATASET_ID",     "cognito_prod_datamart")
LITIGATION_TABLE  = f"{BQ_PROJECT_ID}.{BQ_DATASET}.litigation_analysis_table"


# ══════════════════════════════════════════════════════════════════════════════
# BigQuery — fetch litigation data
# ══════════════════════════════════════════════════════════════════════════════

def _get_bq_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account as _sa
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    credentials = None
    if cred_path and Path(cred_path).exists():
        credentials = _sa.Credentials.from_service_account_file(cred_path)
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)


def load_from_bigquery(drugs: list | None = None, latest_only: bool = False) -> pd.DataFrame:
    """
    Fetch rows from litigation_analysis_table and return a DataFrame with
    columns that match what the reference report generator expects:
        Drug Name, Innovator, Case Number, Case Type, Challenger,
        Court, Outcome, Rationale, patent_number, status, filing_date,
        brand_names, summary, analysis_date, total_cases, unique_challengers
    """
    from google.cloud import bigquery

    bq = _get_bq_client()

    # Build drug filter
    drug_filter = ""
    params = []
    if drugs:
        drug_filter = "AND LOWER(drug_name) IN UNNEST(@drugs)"
        params.append(
            bigquery.ArrayQueryParameter("drugs", "STRING", [d.lower().strip() for d in drugs])
        )

    # When latest_only, keep only the most recent loaded_at per drug
    if latest_only:
        inner = f"""
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY LOWER(drug_name)
                       ORDER BY loaded_at DESC
                   ) AS _rn
            FROM `{LITIGATION_TABLE}`
            WHERE 1=1 {drug_filter}
        """
        sql = f"SELECT * EXCEPT(_rn) FROM ({inner}) WHERE _rn = 1"
    else:
        sql = f"SELECT DISTINCT * FROM `{LITIGATION_TABLE}` WHERE 1=1 {drug_filter} ORDER BY drug_name, loaded_at DESC"

    job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
    print(f"  [BQ] Querying {LITIGATION_TABLE}...")
    try:
        df = bq.query(sql, job_config=job_config).to_dataframe()
    except Exception as e:
        raise RuntimeError(f"BigQuery query failed: {e}")

    if df.empty:
        raise RuntimeError(f"No rows returned from {LITIGATION_TABLE} for the requested drugs. "
                           "Run litigation_analysis.py first.")

    print(f"  [BQ] {len(df)} row(s) fetched for {df['drug_name'].nunique()} drug(s)")

    # ── Rename / map columns to match reference report expectations ───────────
    df = df.rename(columns={
        "drug_name":    "Drug Name",
        "innovator":    "Innovator",
        "case_number":  "Case Number",
        "case_type":    "Case Type",
        "challenger":   "Challenger",
        "court":        "Court",
        "outcome":      "Outcome",          # will be passed through simplify_outcome
        "status":       "Status",
        "filing_date":  "Filing Date",
        "patent_number":"Patent Number",
        "brand_names":  "Brand Names",
        "summary":      "Summary",
    })

    # Rationale: use the per-run summary as a proxy if no dedicated rationale column
    if "Rationale" not in df.columns:
        df["Rationale"] = df.get("Summary", "")

    # Fill common missing columns so downstream code doesn't crash
    for col in ["Case Number", "Case Type", "Challenger", "Court", "Outcome", "Rationale", "Innovator"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["Outcome"] = df["Outcome"].where(df["Outcome"] != "", df.get("Status", "Pending"))
    df["Outcome"] = df["Outcome"].apply(simplify_outcome)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Outcome normalisation (identical to reference)
# ══════════════════════════════════════════════════════════════════════════════

OUTCOME_MAP = {
    "innovator win":       "Won",
    "win":                 "Won",
    "won":                 "Won",
    "prevailed":           "Won",
    "challenger win":      "Lost",
    "lost":                "Lost",
    "invalidated":         "Lost",
    "settled":             "Settled",
    "settlement":          "Settled",
    "pending":             "Pending",
    "ongoing":             "Pending",
    "active":              "Pending",
    "dismissed":           "Dismissed",
    "voluntary dismissal": "Dismissed",
    "resolved":            "Resolved",
    "consent judgment":    "Resolved",
    "license":             "Resolved",
    "licensed":            "Resolved",
}


def simplify_outcome(raw):
    if pd.isna(raw) or str(raw).strip() == "":
        return "Pending"
    key = str(raw).strip().lower()
    if key in OUTCOME_MAP:
        return OUTCOME_MAP[key]
    for pattern, label in OUTCOME_MAP.items():
        if pattern in key:
            return label
    first_word = key.split()[0].capitalize() if key else "Pending"
    return first_word


# ══════════════════════════════════════════════════════════════════════════════
# Web research helpers (identical to reference)
# ══════════════════════════════════════════════════════════════════════════════

def research_innovator(company: str, drugs: list) -> str:
    prompt = (
        f"Research pharmaceutical company '{company}' and provide:\n"
        f"1. Patent filing history — total patent portfolio size, filings per year trend, "
        f"key therapeutic areas, lifecycle management and evergreening patterns\n"
        f"2. Litigation track record — win rate in Hatch-Waxman/ANDA cases, notable wins and losses, "
        f"preferred courts, litigation duration patterns\n"
        f"3. Recent patent activity for: {', '.join(drugs)}\n"
        f"4. Industry reputation — how their IP strategy compares to peers like Pfizer, Roche, Novartis\n\n"
        f"Be specific with numbers, dates, case names. 250 words max. Plain text, no markdown."
    )
    try:
        response = client.models.generate_content(
            model=MODEL_ID, contents=prompt,
            config=types.GenerateContentConfig(tools=[google_search_tool]),
        )
        text = (response.text or "").strip()
        print(f"  [WEB] {company}: {len(text)} chars")
        return text
    except Exception as e:
        print(f"  [WEB] Failed for {company}: {e}")
        return ""


def _trim(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    t = " ".join(words[:limit])
    for p in ".!?":
        i = t.rfind(p)
        if i > len(t) * 0.7:
            return t[: i + 1]
    return t + "."


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Executive Summary + Innovator Analysis
# ══════════════════════════════════════════════════════════════════════════════

def generate_page1(df: pd.DataFrame, innovator_research: dict, all_drug_data: dict) -> str:
    stats = json.dumps({
        "total_cases": len(df),
        "outcomes":    df["Outcome"].value_counts().to_dict(),
        "case_types":  df["Case Type"].value_counts().to_dict(),
    }, indent=2)

    case_lines = [
        f"{r.get('Drug Name','')} | {r.get('Case Number','')} | "
        f"{r.get('Challenger','')} | {r.get('Outcome','')}"
        for _, r in df.iterrows()
    ]

    innovator_blocks = []
    for co, research in innovator_research.items():
        drugs = [d for d, data in all_drug_data.items() if data["innovator"] == co]
        innovator_blocks.append(
            f"COMPANY: {co} | DRUGS: {', '.join(drugs)}\n"
            f"WEB RESEARCH:\n{research[:600]}"
        )

    prompt = f"""You are a senior pharmaceutical patent litigation analyst writing page 1 of a
detailed board-level report. This must be substantive and insightful — NOT a summary of bullet points.
Write in flowing, professional prose. Plain text only — no markdown, no bullets, no asterisks.
Use ALL CAPS for section headers only.

PORTFOLIO DATA:
{stats}

CASES:
{chr(10).join(case_lines)}

INNOVATOR INTELLIGENCE:
{chr(10).join(innovator_blocks)}

═══════════════════════════════════════════════
WRITE EXACTLY THESE SECTIONS — fill the full page:
═══════════════════════════════════════════════

EXECUTIVE SUMMARY
Two substantial paragraphs (~200 words total).
Paragraph 1: Portfolio landscape — total cases analysed,
outcome breakdown with exact percentages (e.g. "innovator prevailed in 65% of cases"),
the most active challengers and their success rates, and which case types dominate.
Paragraph 2: Strategic significance — what this litigation record reveals about the
strength of the innovator's patent estate, whether the trend favours innovators or
challengers, and the single most important strategic implication for decision-makers.

INNOVATOR ANALYSIS
For EACH innovator company, write a dedicated subsection with the COMPANY NAME in ALL CAPS.
Each company gets 2 well-developed paragraphs (~200 words per company):

Paragraph 1 — FILING HISTORY & IP STRATEGY: Total patents filed (from web research),
filing trends over recent years, key jurisdictions (US/EP/PCT), lifecycle management
approach — do they file continuation patents, divisionals, new formulations near expiry?
How does their filing pattern for the drugs in this portfolio compare to industry norms?
Cite specific numbers and years from the web research.

Paragraph 2 — LITIGATION TRACK RECORD & POSITIONING: Win/loss record from the case data
(cite specific case numbers), preferred litigation venues, how they compare to industry
peers in IP aggressiveness, their reputation for settlement vs trial, and any notable
strategic moves (e.g. authorized generics, licensing deals, IPR defences).
Reference specific cases from the data.

═══════════════════════════════════════════════
RULES:
- Write ~550 words total. This fills one A4 page at 9.5pt font.
- Every paragraph must contain specific numbers, case references, or dates.
- Do NOT write generic statements like "the company has a strong patent portfolio" without evidence.
- Start directly with EXECUTIVE SUMMARY. No preamble, no introduction line.
"""
    try:
        return _trim(client.models.generate_content(model=MODEL_ID, contents=prompt).text.strip(), 600)
    except Exception as e:
        return f"[Page 1 failed: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Drug Analysis + Strategic Outlook
# ══════════════════════════════════════════════════════════════════════════════

def generate_page2(df: pd.DataFrame, all_drug_data: dict) -> str:
    drug_blocks = []
    for drug, data in all_drug_data.items():
        cdf = data["cases_df"]
        cols = [c for c in ["Case Number", "Case Type", "Challenger", "Court", "Outcome", "Rationale"] if c in cdf.columns]
        cases = cdf[cols].to_string(index=False)
        drug_blocks.append(
            f"DRUG: {drug} | INNOVATOR: {data['innovator']} | CASES: {len(cdf)}\n{cases}"
        )

    num      = len(all_drug_data)
    per_drug = max(100, 380 // max(num, 1))

    prompt = f"""You are a senior pharmaceutical patent litigation analyst writing page 2 of a
detailed board-level report. This must be substantive — rich with case-specific analysis.
Write in flowing, professional prose. Plain text only — no markdown, no bullets, no asterisks.
Use ALL CAPS for section headers only.

DRUG-LEVEL DATA:
{chr(10).join(drug_blocks)}

═══════════════════════════════════════════════
WRITE EXACTLY THESE SECTIONS — fill the full page:
═══════════════════════════════════════════════

DRUG-LEVEL ANALYSIS
For each drug, write a subsection with the DRUG NAME in ALL CAPS as header.
Each drug: 1-2 detailed paragraphs (~{per_drug} words). Cover:
- Case outcomes in detail: which specific cases (cite case numbers) were won, lost,
  or settled, and the key reasoning behind each outcome
- Challenger landscape: who challenged, their litigation strategy (ANDA, IPR, declaratory
  judgment), and whether they succeeded
- Patent strength assessment: based on the outcomes, how defensible is this drug's
  patent estate? Were any patents invalidated? On what grounds?
- Court patterns: any notable jurisdictional choices or venue strategies

KEY RISKS AND STRATEGIC OUTLOOK
Two paragraphs (~150 words total).
Paragraph 1: Identify the most vulnerable drugs and the specific risks — which patents
are weakest, which challengers are most likely to succeed, and any upcoming expirations
or pending cases that could change the landscape. Be specific about timelines.
Paragraph 2: Strategic recommendations — what the innovator should do (strengthen
specific patents, settle particular cases, file continuations, pursue authorized generics)
and what generic challengers should target. End with a forward-looking assessment.

═══════════════════════════════════════════════
RULES:
- Write ~550 words total. This fills one A4 page at 9.5pt font.
- Every paragraph must cite specific case numbers and outcomes.
- Do NOT write generic statements. Every claim must be grounded in the data.
- Start directly with the first drug header. No preamble.
"""
    try:
        return _trim(client.models.generate_content(model=MODEL_ID, contents=prompt).text.strip(), 600)
    except Exception as e:
        return f"[Page 2 failed: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# PDF builder (identical styling to reference)
# ══════════════════════════════════════════════════════════════════════════════

def _render(text: str, story: list, s_h2, s_body) -> None:
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        is_hdr = (
            re.match(r"^[0-9]+\.\s+[A-Z\s&]+$", line)
            or (line.isupper() and 4 < len(line) < 80 and not line.startswith("["))
        )
        if is_hdr:
            story.append(Spacer(1, 0.06 * cm))
            story.append(Paragraph(line, s_h2))
        else:
            story.append(Paragraph(line, s_body))


def build_pdf(df: pd.DataFrame, p1: str, p2: str, output_pdf: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_pdf), pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.3 * cm,
        title="Pharmaceutical Litigation Analysis Report",
    )
    styles = getSampleStyleSheet()
    W = A4[0] - 3.6 * cm

    s_title = ParagraphStyle("CT", parent=styles["Title"], fontSize=18,
        textColor=colors.HexColor("#1a237e"), spaceAfter=3, alignment=TA_CENTER)
    s_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=11.5,
        textColor=colors.HexColor("#1a237e"), spaceBefore=6, spaceAfter=3)
    s_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10,
        textColor=colors.HexColor("#283593"), spaceBefore=5, spaceAfter=2)
    s_body = ParagraphStyle("BD", parent=styles["Normal"], fontSize=9.5, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=4)
    s_meta = ParagraphStyle("MT", parent=styles["Normal"], fontSize=7.5,
        textColor=colors.HexColor("#78909c"), alignment=TA_CENTER)
    s_ft = ParagraphStyle("FT", parent=styles["Normal"], fontSize=6.5,
        textColor=colors.HexColor("#9e9e9e"), alignment=TA_CENTER)

    story = []

    drug_name      = df["Drug Name"].dropna().unique()
    innovator_name = df["Innovator"].dropna().unique()
    drug_label      = drug_name[0]      if len(drug_name) == 1      else ", ".join(drug_name)
    innovator_label = innovator_name[0] if len(innovator_name) == 1 else ", ".join(innovator_name)

    # ── PAGE 1 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("PHARMACEUTICAL LITIGATION ANALYSIS", s_title))
    story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph(
        f"Source: BigQuery · {LITIGATION_TABLE.split('.')[-1]}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Drug: {drug_label}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Innovator: {innovator_label}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Cases: {len(df)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Challengers: {df['Challenger'].nunique()}", s_meta,
    ))
    story.append(Spacer(1, 0.12 * cm))

    # Outcome bar
    oc = df["Outcome"].value_counts()
    if not oc.empty:
        items = list(oc.items())
        hdrs  = [str(o) for o, _ in items]
        vals  = [f"{c} ({c/len(df)*100:.0f}%)" for _, c in items]
        n     = len(items)
        t = Table([hdrs, vals], colWidths=[W / max(n, 1)] * n)
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#37474f")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("BACKGROUND",   (0, 1), (-1, 1), colors.HexColor("#eceff1")),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 7.5),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.12 * cm))

    _render(p1, story, s_h2, s_body)
    story.append(PageBreak())

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story.append(Paragraph("DRUG-LEVEL ANALYSIS &amp; STRATEGIC OUTLOOK", s_h1))
    story.append(HRFlowable(width=W, thickness=1.5, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.08 * cm))
    _render(p2, story, s_h2, s_body)

    story.append(Spacer(1, 0.12 * cm))
    story.append(HRFlowable(width=W, thickness=0.3, color=colors.HexColor("#cfd8dc")))
    story.append(Paragraph(
        f"Generated by Gemini 2.0 Flash with Google Search"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Source: {LITIGATION_TABLE}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;AI-generated — review by qualified professionals recommended.",
        s_ft,
    ))

    doc.build(story)


# ── GCS upload (same destination as reports.py) ───────────────────────────────
GCS_BUCKET    = os.getenv("GCS_BUCKET",      "cognito-prod")
GCS_BASE_PATH = "Cognito_new/reports"
GCS_SUBFOLDER = "IP"
GCS_FILENAME  = "Litigation_Analysis.pdf"

CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")


def _get_gcs_client():
    from google.cloud import storage
    if CREDENTIALS_PATH and Path(CREDENTIALS_PATH).exists():
        return storage.Client.from_service_account_json(CREDENTIALS_PATH)
    return storage.Client(project=BQ_PROJECT_ID)


def _upload_to_gcs(local_path: str, drug_name: str) -> str:
    """Upload PDF to gs://{GCS_BUCKET}/Cognito_new/reports/{drug_name}/IP/Litigation_Analysis.pdf"""
    from google.cloud import storage
    import re as _re

    safe_name = _re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}/{GCS_FILENAME}"
    gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_name)

    # Archive existing version before overwriting
    try:
        existing = bucket.blob(blob_name)
        if existing.exists():
            ts_str = (existing.updated or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
            archive_name = (
                f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}"
                f"/archive/{ts_str}_{GCS_FILENAME}"
            )
            bucket.copy_blob(existing, bucket, archive_name)
            print(f"    📦 archived prior version → gs://{GCS_BUCKET}/{archive_name}")
    except Exception as e:
        print(f"    [WARN] archive step failed: {e}")

    blob.upload_from_filename(local_path, content_type="application/pdf")
    print(f"    ✅ Uploaded → {gcs_uri}")
    return gcs_uri


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate litigation PDF from BigQuery litigation_analysis_table")
    parser.add_argument("--drugs",       nargs="+", default=None,
        help="Filter to specific drugs. Example: --drugs semaglutide tirzepatide")
    parser.add_argument("--latest-only", action="store_true",
        help="Use only the most recent analysis run per drug (deduplicates reruns)")
    parser.add_argument("--output",      default=None,
        help="Output PDF path (default: ./outputs/litigation_report_<timestamp>.pdf)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHARMACEUTICAL LITIGATION REPORT GENERATOR")
    print("=" * 60)
    print(f"  Source  : {LITIGATION_TABLE}")
    if args.drugs:
        print(f"  Drugs   : {', '.join(args.drugs)}")
    if args.latest_only:
        print("  Mode    : latest run per drug only")
    print()

    print("[1/6] Fetching litigation data from BigQuery...")
    df = load_from_bigquery(drugs=args.drugs, latest_only=args.latest_only)

    print("[2/6] Building per-drug data structures...")
    drugs = df["Drug Name"].dropna().unique()
    all_drug_data: dict = {}
    for i, d in enumerate(drugs, 1):
        rows = df[df["Drug Name"] == d]
        inn  = rows["Innovator"].iloc[0] if not rows.empty else "Unknown"
        all_drug_data[d] = {"cases_df": rows, "innovator": inn}
        print(f"  ({i}/{len(drugs)}) {d} — {inn} — {len(rows)} case(s)")

    # Generate one report per drug
    for drug_name, data in all_drug_data.items():
        drug_df = data["cases_df"]
        inn     = data["innovator"]

        print(f"\n{'─'*60}")
        print(f"  Generating report for: {drug_name}")
        print(f"{'─'*60}")

        drug_data_single = {drug_name: data}

        print(f"[3/6] Researching innovator ({inn}) via Google Search...")
        inv_research = {inn: research_innovator(inn, [drug_name])}

        print("[4/6] Generating page 1 (Executive Summary + Innovator Analysis)...")
        p1 = generate_page1(drug_df, inv_research, drug_data_single)

        print("[5/6] Generating page 2 (Drug Analysis + Strategic Outlook)...")
        p2 = generate_page2(drug_df, drug_data_single)

        # Local save
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        if args.output:
            out = Path(args.output).resolve()
        else:
            out = output_dir / safe / "Litigation_Analysis.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)

        print("[6/6] Building PDF...")
        build_pdf(drug_df, p1, p2, out)
        print(f"  Report saved locally: {out}")

        # Upload to GCS (same path as other reports)
        try:
            _upload_to_gcs(str(out), drug_name)
        except Exception as e:
            print(f"  [WARN] GCS upload failed for {drug_name}: {e}")

    print("\n" + "=" * 60)
    print(f"  Done — {len(all_drug_data)} report(s) generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
