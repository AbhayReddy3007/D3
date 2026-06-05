"""
forecast_report.py
──────────────────
Generates polished PDF forecast reports for drugs.

Reads data from BigQuery:
  - forecasted_loe  (scored forecasts from Step 6 pipeline)
  - Master_LOE              (combined LOE + forecast table)

Usage:
    python forecast_report.py                           # all drugs
    python forecast_report.py --drug Semaglutide        # single drug
    python forecast_report.py --limit 5                 # first 5
    python forecast_report.py --output-dir ./reports    # custom output dir
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None
# google.generativeai is deprecated. Use google.genai (new SDK) only.
from google import genai
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    PageTemplate, Frame, HRFlowable, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── ENV SETUP ─────────────────────────────────────────────────────────────── #
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
_genai_client = genai.Client(api_key=api_key) if api_key else None

# ── BQ CONFIG ─────────────────────────────────────────────────────────────── #
BQ_PROJECT_ID      = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID      = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")

FORECAST_SCORED_TABLE = os.getenv("BQ_FORECAST_SCORED_TABLE",
    f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.forecasted_loe")
MASTER_LOE_TABLE      = os.getenv("BQ_MASTER_LOE_TABLE",
    f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.Master_LOE")

OUTPUT_DIR = "forecast_reports"


# ── BQ CLIENT ─────────────────────────────────────────────────────────────── #

_bq_client = None

def _get_bq_client():
    global _bq_client
    if _bq_client is not None:
        return _bq_client

    from google.cloud import bigquery
    from google.oauth2 import service_account

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path and os.path.exists(cred_path):
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        _bq_client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
    else:
        _bq_client = bigquery.Client(project=BQ_PROJECT_ID)

    return _bq_client


# ── LOAD DATA FROM BQ ────────────────────────────────────────────────────── #

def _load_forecast_scored(drug_name: str = None) -> pd.DataFrame:
    """Load deduplicated forecast data from forecasted_loe using ROW_NUMBER."""
    client = _get_bq_client()
    from google.cloud import bigquery

    drug_filter = ""
    params = []
    if drug_name:
        drug_filter = "WHERE drug_name = @drug"
        params = [bigquery.ScalarQueryParameter("drug", "STRING", drug_name)]

    query = f"""
    SELECT * EXCEPT(rn) FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY drug_name, patent_number, jurisdiction, step1_claim_category
                ORDER BY scored_at DESC
            ) AS rn
        FROM `{FORECAST_SCORED_TABLE}`
        {drug_filter}
    ) WHERE rn = 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
    df = client.query(query, job_config=job_config).to_dataframe()
    return df


def _load_master_loe(drug_name: str = None) -> pd.DataFrame:
    """Load Master_LOE data from BQ, optionally filtered by drug."""
    client = _get_bq_client()
    if drug_name:
        query = f"""
        SELECT DISTINCT * FROM `{MASTER_LOE_TABLE}`
        WHERE LOWER(Drug_Name) = LOWER(@drug)
        """
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("drug", "STRING", drug_name)
            ]
        )
        df = client.query(query, job_config=job_config).to_dataframe()
    else:
        query = f"SELECT DISTINCT * FROM `{MASTER_LOE_TABLE}`"
        df = client.query(query).to_dataframe()
    return df


def _list_drugs() -> list:
    """List all distinct drugs in the forecasted_loe table."""
    client = _get_bq_client()
    query = f"""
    SELECT DISTINCT drug_name
    FROM `{FORECAST_SCORED_TABLE}`
    WHERE drug_name IS NOT NULL AND TRIM(drug_name) != ''
    ORDER BY drug_name
    """
    df = client.query(query).to_dataframe()
    return df["drug_name"].tolist()


# ── BUILD DATA SUMMARY FOR PROMPT ────────────────────────────────────────── #

def _build_data_summary(drug_name: str, forecast_df: pd.DataFrame,
                         master_df: pd.DataFrame) -> str:
    """Build a structured text summary of the drug's data for the AI prompt."""

    lines = []

    # ── Forecast scored data ──
    if not forecast_df.empty:
        lines.append(f"=== FORECAST DATA FOR {drug_name.upper()} ===")
        lines.append(f"Total forecasted patent entries: {len(forecast_df)}")

        # Key metrics
        if "ip_dimension_1_score" in forecast_df.columns:
            scores = forecast_df["ip_dimension_1_score"].dropna().unique()
            if len(scores):
                lines.append(f"IP Dimension 1 Score: {scores[0]}")

        if "avg_years_to_entry_us_ep" in forecast_df.columns:
            avg_yte = forecast_df["avg_years_to_entry_us_ep"].dropna().unique()
            if len(avg_yte):
                lines.append(f"Avg Years to Entry (US & EP): {avg_yte[0]}")

        if "avg_years_to_entry" in forecast_df.columns:
            avg_all = forecast_df["avg_years_to_entry"].dropna().unique()
            if len(avg_all):
                lines.append(f"Avg Years to Entry (all jurisdictions): {avg_all[0]}")

        if "global_phase" in forecast_df.columns:
            phase = forecast_df["global_phase"].dropna().unique()
            if len(phase):
                lines.append(f"Global Phase: {phase[0]}")

        if "company" in forecast_df.columns:
            company = forecast_df["company"].dropna().unique()
            if len(company):
                lines.append(f"Company: {company[0]}")

        if "drug_class" in forecast_df.columns:
            cls = forecast_df["drug_class"].dropna().unique()
            if len(cls):
                lines.append(f"Drug Class: {cls[0]}")

        # Summary fields
        for col in ["overall_forecast", "portfolio_gaps", "risk_assessment",
                     "existing_patent_summary"]:
            if col in forecast_df.columns:
                vals = forecast_df[col].dropna().astype(str).unique()
                vals = [v for v in vals if v.strip() and v.lower() not in ("nan", "none", "")]
                if vals:
                    lines.append(f"\n{col.replace('_', ' ').title()}:")
                    lines.append(f"  {vals[0][:1500]}")

        # Per-entry details
        lines.append(f"\n--- Forecast Entries (top 40) ---")
        display_cols = [c for c in [
            "patent_number", "step1_claim_category", "jurisdiction",
            "likelihood", "filing_window", "phase_in_jurisdiction",
            "controlling_patent_expiry_year", "years_to_entry",
            "rationale", "strategic_purpose", "no_of_forecasted_patents"
        ] if c in forecast_df.columns]

        for _, row in forecast_df.head(40).iterrows():
            parts = [f"{c}: {row[c]}" for c in display_cols if pd.notna(row.get(c))]
            lines.append("  " + " | ".join(parts))

    # ── Master LOE context (existing patents) ──
    if not master_df.empty:
        loe_rows = master_df[master_df.get("Source_File", pd.Series(dtype=str)).str.lower() == "loe"] \
            if "Source_File" in master_df.columns else master_df

        if not loe_rows.empty:
            lines.append(f"\n=== EXISTING LOE PATENTS FOR {drug_name.upper()} ===")
            lines.append(f"Total existing patents: {len(loe_rows)}")

            if "Jurisdiction" in loe_rows.columns:
                juris = loe_rows["Jurisdiction"].value_counts().to_dict()
                lines.append(f"Jurisdictions: {juris}")

            if "Step_1_Claim_Category" in loe_rows.columns:
                cats = loe_rows["Step_1_Claim_Category"].value_counts().to_dict()
                lines.append(f"Patent categories: {cats}")

            if "Tag" in loe_rows.columns:
                tags = loe_rows["Tag"].value_counts().to_dict()
                lines.append(f"Tags: {tags}")

            if "Years_to_Entry" in loe_rows.columns:
                yte_vals = pd.to_numeric(loe_rows["Years_to_Entry"], errors="coerce").dropna()
                if not yte_vals.empty:
                    lines.append(f"Existing YTE range: {yte_vals.min()} – {yte_vals.max()}")

            # Show a few existing patent rows
            lines.append(f"\n--- Existing Patent Details (first 20) ---")
            show_cols = [c for c in [
                "Patent_Number", "Step_1_Claim_Category", "Jurisdiction",
                "Tag", "Blocking_Category", "Filing_Date",
                "Controlling_Patent_Expiry_Year", "Years_to_Entry"
            ] if c in loe_rows.columns]

            for _, row in loe_rows.head(20).iterrows():
                parts = [f"{c}: {row[c]}" for c in show_cols if pd.notna(row.get(c))]
                lines.append("  " + " | ".join(parts))

    return "\n".join(lines)


# ── GENERATE REPORT TEXT ──────────────────────────────────────────────────── #

def _generate_report_text(drug_name: str, data_summary: str) -> str:
    """Use Gemini to generate the report text."""

    prompt = f"""
You are a senior pharma patent strategist producing a concise consulting report.

DRUG: {drug_name}
DATE: {datetime.now().strftime('%Y-%m-%d')}

STRICT RULES:
- Maximum 700–800 words (STRICT)
- Must NOT exceed 2 pages
- No repetition of table data
- Be concise, analytical, and structured
- Write in direct analytical voice — state findings and insights directly
- Do NOT use phrases like "this report analyses", "this report examines", or "in this report"
- Do NOT use markdown bold markers like ** around headings or text
- Section headings should be plain text only
- Include specific numbers from the data (scores, years to entry, patent counts)
- Compare forecasted patents against existing LOE patents where data is available

MANDATORY SECTIONS (use these exact heading titles, plain text):
1. Executive Summary
2. Patent Strategy Insights
3. Filing Window Analysis
4. Portfolio Gaps and Opportunities
5. Risk Assessment

DATA:
{data_summary}
"""

    if _genai_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY not set — forecast_report cannot generate report text."
        )
    response = _genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    report_text = response.text or ""

    # Strip any residual ** markers from the AI output
    report_text = re.sub(r'\*\*(.+?)\*\*', r'\1', report_text)
    report_text = re.sub(r'\*(.+?)\*', r'\1', report_text)

    # Hard cap for 2-page safety
    report_text = report_text[:4500]

    return report_text


# ── COLOUR PALETTE ────────────────────────────────────────────────────────── #
NAVY       = colors.HexColor("#0D2B55")
STEEL      = colors.HexColor("#1A5276")
ACCENT     = colors.HexColor("#2E86C1")
LIGHT_BLUE = colors.HexColor("#D6EAF8")
GOLD       = colors.HexColor("#D4AC0D")
WHITE      = colors.white
DARK_GREY  = colors.HexColor("#2C3E50")
MID_GREY   = colors.HexColor("#566573")


# ── CUSTOM STYLES ─────────────────────────────────────────────────────────── #
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    name="ReportTitle",
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=26,
    alignment=TA_CENTER,
    textColor=WHITE,
    spaceAfter=4,
)

subtitle_style = ParagraphStyle(
    name="ReportSubtitle",
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    alignment=TA_CENTER,
    textColor=LIGHT_BLUE,
    spaceAfter=0,
)

section_heading_style = ParagraphStyle(
    name="SectionHeading",
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=15,
    textColor=NAVY,
    spaceBefore=14,
    spaceAfter=4,
    leftIndent=0,
)

body_style = ParagraphStyle(
    name="BodyText",
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=DARK_GREY,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
    leftIndent=2,
)

# ── SECTION HEADING KEYWORDS ──────────────────────────────────────────────── #
SECTION_KEYWORDS = [
    "executive summary",
    "patent strategy insights",
    "filing window analysis",
    "portfolio gaps and opportunities",
    "portfolio gap",
    "risk assessment",
]

def is_section_heading(line: str) -> bool:
    low = line.lower().strip()
    low_clean = re.sub(r'^\d+[\.\)]\s*', '', low)
    return any(low_clean == kw or low_clean.startswith(kw) for kw in SECTION_KEYWORDS)


# ── BUILD PDF ─────────────────────────────────────────────────────────────── #

def _build_pdf(drug_name: str, report_text: str, output_path: str):
    """Build the styled PDF from report text."""

    HEADER_HEIGHT = 52

    def draw_page(canvas, doc):
        canvas.saveState()
        w, h = A4

        # Top colour band
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - HEADER_HEIGHT, w, HEADER_HEIGHT, fill=1, stroke=0)

        # Gold accent line below header
        canvas.setFillColor(GOLD)
        canvas.rect(0, h - HEADER_HEIGHT - 3, w, 3, fill=1, stroke=0)

        # Drug name title in header
        canvas.setFont("Helvetica-Bold", 16)
        canvas.setFillColor(WHITE)
        canvas.drawCentredString(
            w / 2, h - HEADER_HEIGHT + 22,
            f"{drug_name} — Forecast Report"
        )

        # Subtitle line
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(LIGHT_BLUE)
        canvas.drawCentredString(
            w / 2, h - HEADER_HEIGHT + 8,
            "Patent Landscape & Strategic Analysis"
        )

        # Thin bottom border
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(0.8)
        canvas.line(18 * mm, 15 * mm, w - 18 * mm, 15 * mm)

        # Page number
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MID_GREY)
        canvas.drawCentredString(w / 2, 9 * mm, f"Page {doc.page}")

        canvas.restoreState()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height - HEADER_HEIGHT - 6,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=4,
        bottomPadding=0,
    )

    template = PageTemplate(id="report_page", frames=[frame], onPage=draw_page)
    doc.addPageTemplates([template])

    # ── Build content ──
    content = []
    content.append(Spacer(1, 6))

    lines = report_text.split("\n")
    current_section_items = []

    def flush_section(items):
        if items:
            content.append(KeepTogether(items[:3]))
            for item in items[3:]:
                content.append(item)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        display = re.sub(r'^\d+[\.\)]\s*', '', stripped)

        if is_section_heading(stripped):
            flush_section(current_section_items)
            current_section_items = []

            current_section_items.append(
                HRFlowable(
                    width="100%",
                    thickness=1.5,
                    color=ACCENT,
                    spaceAfter=4,
                    spaceBefore=8,
                )
            )
            current_section_items.append(
                Paragraph(display.title(), section_heading_style)
            )
        else:
            current_section_items.append(Paragraph(display, body_style))

    flush_section(current_section_items)

    doc.build(content)


# ── GENERATE REPORT FOR ONE DRUG ──────────────────────────────────────────── #

def generate_report(drug_name: str, output_dir: str = None):
    """Full pipeline: load BQ data → AI report → PDF for one drug."""

    output_dir = output_dir or OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n  [{drug_name}] Loading data from BigQuery...")

    # Load forecast scored data
    forecast_df = _load_forecast_scored(drug_name)
    if forecast_df.empty:
        # Try fuzzy match
        all_drugs = _list_drugs()
        drug_lower = drug_name.lower().replace(" ", "").replace("-", "")
        for d in all_drugs:
            if drug_lower in d.lower().replace(" ", "").replace("-", ""):
                forecast_df = _load_forecast_scored(d)
                if not forecast_df.empty:
                    print(f"  [{drug_name}] Matched to '{d}' in BQ")
                    drug_name = d
                    break

    if forecast_df.empty:
        print(f"  [{drug_name}] No forecast data found — skipping")
        return None

    print(f"  [{drug_name}] Forecast: {len(forecast_df)} rows")

    # Load Master_LOE data for context
    master_df = _load_master_loe(drug_name)
    print(f"  [{drug_name}] Master_LOE: {len(master_df)} rows")

    # Build data summary
    data_summary = _build_data_summary(drug_name, forecast_df, master_df)
    print(f"  [{drug_name}] Data summary: {len(data_summary)} chars")

    # Generate report text via Gemini
    print(f"  [{drug_name}] Generating report text...")
    report_text = _generate_report_text(drug_name, data_summary)
    print(f"  [{drug_name}] Report text: {len(report_text)} chars")

    # Build PDF
    safe_name = drug_name.replace(" ", "_").replace("/", "_")
    output_path = str(Path(output_dir) / f"{safe_name}_Forecast_Report.pdf")

    print(f"  [{drug_name}] Building PDF...")
    _build_pdf(drug_name, report_text, output_path)
    print(f"  [{drug_name}] ✅ Saved: {output_path}")

    return output_path


# ─────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────

def run_batch(drug_names: list = None, output_dir: str = None, limit: int = None):
    """Generate reports for multiple drugs."""

    output_dir = output_dir or OUTPUT_DIR

    if not drug_names:
        print("[REPORT] Loading drug list from BQ...")
        drug_names = _list_drugs()

    if not drug_names:
        print(f"[REPORT] No drugs found in {FORECAST_SCORED_TABLE}")
        return

    if limit:
        drug_names = drug_names[:limit]

    total = len(drug_names)
    print(f"\n{'═'*70}")
    print(f"  FORECAST REPORT GENERATION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Drugs: {total}")
    print(f"  Output: {output_dir}/")
    print(f"{'═'*70}")

    succeeded = []
    failed = []

    for i, drug in enumerate(drug_names, 1):
        print(f"\n[{i}/{total}] {drug}")
        try:
            path = generate_report(drug, output_dir)
            if path:
                succeeded.append((drug, path))
            else:
                failed.append((drug, "No data"))
        except Exception as e:
            print(f"  [{drug}] ❌ Error: {e}")
            failed.append((drug, str(e)))

    # Summary
    print(f"\n{'═'*70}")
    print(f"  REPORT GENERATION COMPLETE")
    print(f"{'═'*70}")
    print(f"  Total:     {total}")
    print(f"  Generated: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")

    if succeeded:
        print(f"\n  Generated reports:")
        for drug, path in succeeded:
            print(f"    ✅ {drug}: {path}")

    if failed:
        print(f"\n  Failed:")
        for drug, err in failed:
            print(f"    ❌ {drug}: {err}")

    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PDF forecast reports from BigQuery data."
    )
    parser.add_argument("--drug",       default=None, help="Single drug name")
    parser.add_argument("--limit",      type=int, default=None, help="Limit number of drugs")
    parser.add_argument("--output-dir", default=None, help="Output directory for PDFs")
    parser.add_argument("--dry-run",    action="store_true", help="List drugs only")
    args = parser.parse_args()

    if args.dry_run:
        drugs = _list_drugs() if not args.drug else [args.drug]
        if args.limit:
            drugs = drugs[:args.limit]
        print(f"\n[DRY RUN] {len(drugs)} drug(s):")
        for i, d in enumerate(drugs, 1):
            print(f"  {i}. {d}")
        sys.exit(0)

    if args.drug:
        generate_report(args.drug, args.output_dir)
    else:
        run_batch(output_dir=args.output_dir, limit=args.limit)
