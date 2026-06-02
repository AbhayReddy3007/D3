"""
PTE_Analysis.py
─────────────────────────────────────────────────────────────────────────────
Reads data from Google BigQuery tables:
  - shortlisted_secondary_patents_table  →  replaces "Shortlisted" sheet
  - arbitrage_summary_table              →  replaces "Arbitrage Summary" sheet

Generates a concise PDF report covering Regulatory Exclusivity & PTE analysis,
then uploads each per-drug PDF to Google Cloud Storage under:
  gs://cognito-gcs/Cognito_new/reports/{drug_name}/PTE_Analysis.pdf

API key read from .env  →  GEMINI_API_KEY
Model: gemini-2.5-flash
"""

import os
import sys
import re
import time
import subprocess
import tempfile
from datetime import date

import pandas as pd
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run

# google.generativeai is deprecated. Use google.genai (new SDK) only.
from google import genai
from google.cloud import bigquery
from google.oauth2 import service_account

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY


# ═══════════════════════════════════════════════════════════════════════════
#  BIGQUERY CONFIG
# ═══════════════════════════════════════════════════════════════════════════
BQ_PROJECT_ID  = "cognito-prod-394707"
BQ_DATASET_ID  = "cognito_prod_datamart"
BQ_TABLE_ID    = "Master_LOE"          # not used directly; kept for reference
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
BQ_LOCATION    = "asia-south1"

# Table names
SHORTLISTED_TABLE = "shortlisted_secondary_patents_table"
ARBITRAGE_TABLE   = "arbitrage_summary_table"

# ── snake_case  →  Title Case column mappings ─────────────────────────────
SHORTLISTED_COL_MAP = {
    "drug_name":                    "Drug Name",
    "jurisdiction":                 "Jurisdiction",
    "patent_number":                "Patent Number",
    "step_1_claim_category":        "Step 1 Claim Category",
    "adjusted_expiry_with_pte":     "Adjusted Expiry (with PTE)",
    "expiry_gap_years":             "Expiry Gap (Years)",
    "pte_status":                   "PTE Status",
    "pte_months_granted":           "PTE Months (Granted)",
}

ARBITRAGE_COL_MAP = {
    "drug_name":                    "Drug Name",
    "jurisdiction":                 "Jurisdiction",
    "dimension_iv_score":           "Dimension IV Score",
    "dimension_iv_rating":          "Dimension IV Rating",
    "product_loe_year":             "Product LOE (Year)",
    "gap_vs_us_years":              "Gap vs US (Years)",
    "gap_vs_longest_loe_years":     "Gap vs Longest LOE (Years)",
    "key_protection_gap":           "Key Protection Gap",
    "arbitrage_score":              "Arbitrage Score",
    "arbitrage_signal":             "Arbitrage Signal",
}


# ═══════════════════════════════════════════════════════════════════════════
#  BIGQUERY LOADER
# ═══════════════════════════════════════════════════════════════════════════
def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None  # Use ADC (Application Default Credentials)

def _get_bq_client() -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    credentials = _get_credentials()
    return bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )


def _load_table(client: bigquery.Client, table_name: str) -> pd.DataFrame:
    """Load an entire BQ table into a DataFrame (all columns as strings)."""
    full_ref = f"`{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}`"
    query    = f"SELECT DISTINCT * FROM {full_ref}"
    print(f"  [BQ] Querying {full_ref} …")
    df = client.query(query).to_dataframe().astype(str)
    # Replace literal "None" / "nan" strings with proper NaN
    df.replace({"None": pd.NA, "nan": pd.NA, "<NA>": pd.NA}, inplace=True)
    print(f"       → {len(df)} rows, columns: {list(df.columns)}")
    return df


def _rename_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Rename snake_case BQ columns to Title Case display names.
    Only renames columns that exist; unknown columns are left as-is.
    """
    existing_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing_map)

    # Warn about expected columns that were missing in the table
    missing = [v for k, v in col_map.items() if k not in existing_map]
    if missing:
        print(f"  ⚠  The following expected columns were NOT found in BQ: {missing}")
    return df


def load_data_from_bigquery() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (shortlisted_df, arbitrage_df) with Title Case columns,
    ready for the report generation functions.
    """
    client = _get_bq_client()

    # Shortlisted
    sl_raw = _load_table(client, SHORTLISTED_TABLE)
    sl_df  = _rename_columns(sl_raw, SHORTLISTED_COL_MAP)
    sl_df.columns = sl_df.columns.str.strip()

    # Arbitrage Summary
    arb_raw = _load_table(client, ARBITRAGE_TABLE)
    arb_df  = _rename_columns(arb_raw, ARBITRAGE_COL_MAP)
    arb_df.columns = arb_df.columns.str.strip()

    print(f"\n  ✓ Shortlisted  : {len(sl_df)} rows | "
          f"{sl_df['Drug Name'].nunique() if 'Drug Name' in sl_df.columns else '?'} drug(s)")
    print(f"  ✓ Arbitrage    : {len(arb_df)} rows")

    return sl_df, arb_df


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI SETUP
# ═══════════════════════════════════════════════════════════════════════════
load_dotenv(override=True)
GEMINI_MODEL = "gemini-2.5-flash"
_genai_client = None


def _get_genai_client():
    """Lazily create the Gemini client so importing this module doesn't
    crash when GEMINI_API_KEY is unset (the friendly check in main() then
    handles it gracefully)."""
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _call_gemini(prompt: str, retries: int = 3, backoff: float = 2.5) -> str | None:
    for attempt in range(retries):
        try:
            resp = _get_genai_client().models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
            )
            return (resp.text or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"  ⚠ Gemini error ({e}), retry in {wait}s …")
                time.sleep(wait)
            else:
                print(f"  ✗ Gemini failed: {e}")
                return None


# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm

NAVY   = colors.HexColor("#1F3864")
BLUE   = colors.HexColor("#2F5597")
LBLUE  = colors.HexColor("#DCE6F1")
GREEN  = colors.HexColor("#2ECC71")
AMBER  = colors.HexColor("#F39C12")
RED    = colors.HexColor("#E74C3C")
LGREY  = colors.HexColor("#F5F5F5")
WHITE  = colors.white

SCORE_COLOUR = {5: GREEN, 4: colors.HexColor("#27AE60"),
                3: AMBER,  2: colors.HexColor("#E67E22"), 1: RED}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def safe_year(val) -> str:
    dt = pd.to_datetime(val, errors="coerce")
    return str(dt.year) if pd.notna(dt) else "N/A"


def score_colour(score) -> colors.Color:
    try:
        return SCORE_COLOUR.get(int(score), colors.HexColor("#BDC3C7"))
    except (ValueError, TypeError):
        return colors.HexColor("#BDC3C7")


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI NARRATIVE (per-drug)
# ═══════════════════════════════════════════════════════════════════════════
PTE_EXCL_RULES = {
    "US": "5-yr NCE / 3-yr supplemental exclusivity (Hatch-Waxman); PTE up to 5 yrs (max 14 yrs from approval)",
    "EU": "8+2+1 yr data/market exclusivity; SPC up to 5 yrs extra",
    "CN": "6-yr data exclusivity (innovative drugs); PTE up to 5 yrs (effective patent term capped at 14 yrs from approval)",
    "IN": "No PTE; no data exclusivity under current law",
    "BR": "No PTE; pipeline patents allowed (Art. 230 now struck down); 5-yr data exclusivity for agrochemicals",
    "AU": "5-yr data exclusivity; PTE up to 5 yrs (15-yr effective cap from first TGA approval)",
    "RU": "6-yr data exclusivity; PTE up to 5 yrs",
}


def build_gemini_narrative(drug: str, df: pd.DataFrame, arb_df: pd.DataFrame) -> str:
    """Ask Gemini for a 200-word strategic narrative on exclusivity & PTE for a single drug."""

    jurs      = df["Jurisdiction"].unique().tolist() if "Jurisdiction" in df.columns else []
    pte_count = (df["PTE Status"] == "Granted").sum() if "PTE Status" in df.columns else 0

    # Summarise expiry gaps
    if "Expiry Gap (Years)" in df.columns:
        gaps    = pd.to_numeric(df["Expiry Gap (Years)"], errors="coerce").dropna()
        avg_gap = round(gaps.mean(), 1) if not gaps.empty else "N/A"
        min_gap = int(gaps.min()) if not gaps.empty else "N/A"
        max_gap = int(gaps.max()) if not gaps.empty else "N/A"
    else:
        avg_gap = min_gap = max_gap = "N/A"

    # Arbitrage dimension IV
    dim4_summary = ""
    if not arb_df.empty and "Dimension IV Rating" in arb_df.columns:
        ratings = arb_df[["Drug Name", "Dimension IV Rating"]].drop_duplicates("Drug Name").to_dict("records")
        dim4_summary = "; ".join(f"{r['Drug Name']}→{r['Dimension IV Rating']}" for r in ratings[:6])

    rules_text = "\n".join(f"  {k}: {v}" for k, v in PTE_EXCL_RULES.items() if k in jurs)

    prompt = f"""You are a pharmaceutical patent & regulatory exclusivity strategist.
Below is a summary of a multi-jurisdictional patent analysis for the drug: {drug}.
Write a concise, insight-driven narrative (strictly 180-220 words) covering:
1. Overall patent expiry landscape and exclusivity risk.
2. PTE status and its impact on effective protection.
3. Geographic arbitrage opportunities and weakest jurisdictions.
4. Key strategic recommendations (2-3 bullet points).

Data summary:
- Drug analysed: {drug}
- Jurisdictions: {', '.join(jurs)}
- PTE Granted count: {pte_count}
- Expiry Gap (Years) — avg: {avg_gap}, min: {min_gap}, max: {max_gap}
- Dimension IV ratings: {dim4_summary or 'N/A'}

Regulatory exclusivity rules in scope:
{rules_text}

Rules for your response:
- Plain prose; no markdown headers, no bullet symbols except for recommendations section.
- Recommendations section: start with "Recommendations:" then use "•" bullets.
- Do NOT include any JSON, code, or special formatting.
- Exactly 180-220 words.
"""
    raw = _call_gemini(prompt)
    return raw if raw else "Narrative unavailable — Gemini API error."


# ═══════════════════════════════════════════════════════════════════════════
#  STYLES
# ═══════════════════════════════════════════════════════════════════════════
def build_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        parent = kw.pop("parent", "Normal")
        s = ParagraphStyle(name, parent=base[parent], **kw)
        return s

    return {
        "title":    S("RptTitle",  parent="Title",  fontSize=16, textColor=NAVY,
                      spaceAfter=2, leading=20, alignment=TA_LEFT),
        "subtitle": S("RptSub",    fontSize=9,  textColor=BLUE, spaceAfter=6, leading=11),
        "h1":       S("RptH1",     fontSize=10, textColor=WHITE, leading=14,
                      backColor=NAVY, borderPadding=(3, 6, 3, 6)),
        "h2":       S("RptH2",     fontSize=9,  textColor=NAVY, leading=12,
                      fontName="Helvetica-Bold", spaceAfter=3, spaceBefore=4),
        "body":     S("RptBody",   fontSize=8,  leading=11, spaceAfter=3, alignment=TA_JUSTIFY),
        "small":    S("RptSmall",  fontSize=7,  leading=9,  textColor=colors.grey),
        "cell":     S("RptCell",   fontSize=7.5, leading=10, alignment=TA_CENTER),
        "cell_l":   S("RptCellL",  fontSize=7.5, leading=10, alignment=TA_LEFT),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
def section_header(text: str, st: dict):
    return [
        Spacer(1, 0.15 * cm),
        Table(
            [[Paragraph(text, st["h1"])]],
            colWidths=[PAGE_W - 2 * MARGIN],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ])
        ),
        Spacer(1, 0.1 * cm),
    ]


def pte_table(df: pd.DataFrame, st: dict) -> list:
    """PTE summary table: Drug | Jurisdiction | PTE Status | Months | Adjusted Expiry."""
    needed = ["Drug Name", "Jurisdiction", "PTE Status", "PTE Months (Granted)", "Adjusted Expiry (with PTE)"]
    cols   = [c for c in needed if c in df.columns]
    sub    = df[cols].copy()

    # Pretty-print expiry year
    if "Adjusted Expiry (with PTE)" in sub.columns:
        sub["Adjusted Expiry (with PTE)"] = sub["Adjusted Expiry (with PTE)"].apply(safe_year)

    headers = {
        "Drug Name": "Drug", "Jurisdiction": "Jur.",
        "PTE Status": "PTE Status", "PTE Months (Granted)": "PTE (mo.)",
        "Adjusted Expiry (with PTE)": "Adj. Expiry",
    }
    display_cols = [headers.get(c, c) for c in cols]
    col_w = [3.8*cm, 1.5*cm, 2.4*cm, 2*cm, 2.5*cm][:len(cols)]
    remaining = PAGE_W - 2*MARGIN - sum(col_w)
    if remaining > 0 and col_w:
        col_w[-1] += remaining

    rows = [display_cols]
    for _, row in sub.iterrows():
        rows.append([str(row.get(c, "")) for c in cols])

    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGREY, WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
    ])

    # Colour-code PTE status column (index 2 if present)
    if "PTE Status" in cols:
        si = cols.index("PTE Status")
        for i, row in enumerate(rows[1:], start=1):
            val = row[si]
            if val == "Granted":
                ts.add("BACKGROUND", (si, i), (si, i), GREEN)
                ts.add("TEXTCOLOR",  (si, i), (si, i), WHITE)
            elif val in ("Pending",):
                ts.add("BACKGROUND", (si, i), (si, i), AMBER)
                ts.add("TEXTCOLOR",  (si, i), (si, i), WHITE)
            elif val in ("Not applicable",):
                ts.add("TEXTCOLOR",  (si, i), (si, i), colors.grey)

    return [Table(rows, colWidths=col_w, style=ts)]


def arbitrage_summary_table(arb_df: pd.DataFrame, st: dict) -> list:
    """Compact arbitrage table: Drug | Jurisdiction | LOE Year | Score | Signal."""
    needed = ["Drug Name", "Jurisdiction", "Product LOE (Year)", "Arbitrage Score", "Arbitrage Signal"]
    cols   = [c for c in needed if c in arb_df.columns]
    sub    = arb_df[cols].copy()

    headers = {
        "Drug Name": "Drug", "Jurisdiction": "Jur.",
        "Product LOE (Year)": "LOE Year", "Arbitrage Score": "Score",
        "Arbitrage Signal": "Signal",
    }
    display_cols = [headers.get(c, c) for c in cols]
    col_w = [3.5*cm, 1.5*cm, 2*cm, 1.5*cm, 3.5*cm][:len(cols)]
    remaining = PAGE_W - 2*MARGIN - sum(col_w)
    if remaining > 0 and col_w:
        col_w[-1] += remaining

    rows = [display_cols]
    for _, row in sub.iterrows():
        rows.append([str(row.get(c, "")) for c in cols])

    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGREY, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])

    # Colour score column
    if "Arbitrage Score" in cols:
        si = cols.index("Arbitrage Score")
        for i, row in enumerate(rows[1:], start=1):
            c = score_colour(row[si])
            ts.add("BACKGROUND", (si, i), (si, i), c)
            ts.add("TEXTCOLOR",  (si, i), (si, i), WHITE)
            ts.add("FONTNAME",   (si, i), (si, i), "Helvetica-Bold")

    return [Table(rows, colWidths=col_w, style=ts)]


def dimension4_table(arb_df: pd.DataFrame, st: dict) -> list:
    """Dimension IV summary per drug."""
    if "Dimension IV Score" not in arb_df.columns:
        return []

    dim4 = (
        arb_df[["Drug Name", "Dimension IV Score", "Dimension IV Rating"]]
        .drop_duplicates("Drug Name")
        .reset_index(drop=True)
    )

    col_w = [4.5*cm, 2.5*cm, PAGE_W - 2*MARGIN - 7*cm]
    rows = [["Drug", "Dim. IV Score", "Rating"]]
    for _, r in dim4.iterrows():
        rows.append([str(r["Drug Name"]), str(r["Dimension IV Score"]), str(r["Dimension IV Rating"])])

    RATING_COLOURS = {
        "Very Strong Geographic Arbitrage Opportunity": colors.HexColor("#2ECC71"),
        "Strong Geographic Arbitrage Opportunity":      colors.HexColor("#27AE60"),
        "Moderate Geographic Arbitrage Opportunity":    AMBER,
        "Limited Geographic Arbitrage Opportunity":     colors.HexColor("#E67E22"),
        "FAIL — No viable geographic arbitrage":        RED,
    }

    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGREY, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])

    for i, row in enumerate(rows[1:], start=1):
        rating_c = RATING_COLOURS.get(row[2], colors.HexColor("#BDC3C7"))
        ts.add("BACKGROUND", (2, i), (2, i), rating_c)
        ts.add("TEXTCOLOR",  (2, i), (2, i), WHITE)
        ts.add("FONTNAME",   (2, i), (2, i), "Helvetica-Bold")

    return [Table(rows, colWidths=col_w, style=ts)]


def kpi_bar(df: pd.DataFrame, st: dict) -> list:
    """Small KPI row at the top."""
    drugs = df["Drug Name"].nunique() if "Drug Name" in df.columns else 0
    jurs  = df["Jurisdiction"].nunique() if "Jurisdiction" in df.columns else 0

    pte_granted = 0
    if "PTE Status" in df.columns:
        pte_granted = (df["PTE Status"] == "Granted").sum()

    if "Expiry Gap (Years)" in df.columns:
        gaps = pd.to_numeric(df["Expiry Gap (Years)"], errors="coerce").dropna()
        avg_gap = f"{gaps.mean():.1f} yrs" if not gaps.empty else "N/A"
    else:
        avg_gap = "N/A"

    kpis = [
        ("Drugs", str(drugs)),
        ("Jurisdictions", str(jurs)),
        ("PTE Granted", str(pte_granted)),
        ("Avg Expiry Gap", avg_gap),
    ]

    cells = []
    for label, val in kpis:
        cells.append(
            Table(
                [[Paragraph(val, ParagraphStyle("kv", fontSize=14, fontName="Helvetica-Bold",
                                                 textColor=NAVY, alignment=TA_CENTER))],
                 [Paragraph(label, ParagraphStyle("kl", fontSize=7, textColor=colors.grey,
                                                   alignment=TA_CENTER))]],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), LGREY),
                    ("BOX",        (0, 0), (-1, -1), 0.5, BLUE),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]),
                colWidths=[(PAGE_W - 2 * MARGIN) / len(kpis) - 0.2 * cm],
            )
        )

    return [
        Table([cells],
              colWidths=[(PAGE_W - 2 * MARGIN) / len(kpis)] * len(kpis),
              style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 2)])),
        Spacer(1, 0.2 * cm),
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(PAGE_W - MARGIN, 0.7 * cm,
                           f"Page {doc.page}  |  IPD Patent Analysis — Regulatory Exclusivity & PTE Report  |  {date.today()}")
    canvas.drawString(MARGIN, 0.7 * cm, "CONFIDENTIAL")
    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD SINGLE-DRUG REPORT
# ═══════════════════════════════════════════════════════════════════════════
def _build_drug_report(drug: str, drug_sl: pd.DataFrame, drug_arb: pd.DataFrame,
                       output_path: str):
    """Build and save a per-drug PTE PDF report to output_path."""
    st = build_styles()

    print("  Fetching Gemini narrative …")
    narrative = build_gemini_narrative(drug, drug_sl, drug_arb)
    time.sleep(1)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    story = []

    # ── HEADER ────────────────────────────────────────────────────────────
    report_date = date.today().strftime("%d %B %Y")
    story.append(Paragraph(f"{drug} — Regulatory Exclusivity &amp; PTE Analysis", st["title"]))
    story.append(Paragraph(
        f"IPD Patent Analysis Tool — Automated Report &nbsp;|&nbsp; Generated {report_date}",
        st["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=6))

    # ── KPI BAR ───────────────────────────────────────────────────────────
    story += kpi_bar(drug_sl, st)

    # ── SECTION 1: STRATEGIC NARRATIVE ───────────────────────────────────
    story += section_header("1. Strategic Overview (AI-Generated)", st)
    story.append(Paragraph(narrative.replace("\n", " "), st["body"]))

    # ── SECTION 2: PTE ANALYSIS ───────────────────────────────────────────
    story += section_header("2. Patent Term Extension (PTE) Analysis", st)

    # Mini-legend
    legend_items = [
        ("Granted", GREEN), ("Pending", AMBER),
        ("Not filed / Not found", RED), ("Not applicable", colors.grey),
    ]
    legend_cells = []
    for lbl, col in legend_items:
        legend_cells.append(
            Table([[Paragraph(f"<font color='white'><b>{lbl}</b></font>",
                              ParagraphStyle("leg", fontSize=6.5, alignment=TA_CENTER))]],
                  style=TableStyle([("BACKGROUND", (0,0),(-1,-1), col),
                                    ("TOPPADDING", (0,0),(-1,-1), 1),
                                    ("BOTTOMPADDING",(0,0),(-1,-1), 1)]),
                  colWidths=[3*cm])
        )
    story.append(
        Table([legend_cells],
              colWidths=[3.1*cm]*len(legend_cells),
              style=TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                                ("RIGHTPADDING",(0,0),(-1,-1),2)]))
    )
    story.append(Spacer(1, 0.15*cm))
    story += pte_table(drug_sl, st)

    # PTE rules note
    jurs_in_data = drug_sl["Jurisdiction"].unique().tolist() if "Jurisdiction" in drug_sl.columns else []
    rules_lines = [f"<b>{k}</b>: {v}" for k, v in PTE_EXCL_RULES.items() if k in jurs_in_data]
    if rules_lines:
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph("<b>Exclusivity rules in scope:</b>", st["h2"]))
        for rl in rules_lines:
            story.append(Paragraph(f"&bull; {rl}", st["small"]))

    # ── SECTION 3: GEOGRAPHIC ARBITRAGE ───────────────────────────────────
    if not drug_arb.empty:
        story += section_header("3. Geographic Arbitrage &amp; Dimension IV Scores", st)

        story.append(Paragraph("<b>Arbitrage Score key:</b> 5 = Immediate opportunity &nbsp;|&nbsp; "
                               "4 = Strong &nbsp;|&nbsp; 3 = Meaningful &nbsp;|&nbsp; "
                               "2 = Limited &nbsp;|&nbsp; 1 = None &nbsp;|&nbsp; N/A = Reference (US)",
                               st["small"]))
        story.append(Spacer(1, 0.1*cm))
        story += arbitrage_summary_table(drug_arb, st)

        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("<b>Dimension IV — Overall Geographic Arbitrage Rating</b>", st["h2"]))
        story += dimension4_table(drug_arb, st)

    # ── FOOTER NOTE ───────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        "This report is auto-generated. Regulatory exclusivity periods and PTE grants are "
        "sourced from AI inference and may require verification against official registers "
        "(USPTO, CNIPA, IP Australia, Rospatent, CDSCO, ANVISA). Not legal advice.",
        st["small"]
    ))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"  ✓ Report saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  GCS UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
def upload_to_gcs(local_pdf: str, drug_name: str,
                  bucket_name: str = "cognito-gcs") -> str:
    """Upload PDF to gs://{bucket_name}/Cognito_new/reports/{drug_name}/PTE_Analysis.pdf
    Returns the full GCS URI."""
    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError(
            "google-cloud-storage not installed.\n"
            "Run: pip install google-cloud-storage --break-system-packages"
        )

    safe_drug = drug_name.replace("/", "_").replace(" ", "_")
    blob_name = f"Cognito_new/reports/{safe_drug}/PTE_Analysis.pdf"

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)

    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_filename(local_pdf, content_type="application/pdf")

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"      ✓ Uploaded → {gcs_uri}")
    return gcs_uri


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  REGULATORY EXCLUSIVITY & PTE REPORT GENERATOR")
    print("  Data source: Google BigQuery")
    print("  Output     : gs://cognito-gcs/Cognito_new/reports/{drug_name}/PTE_Analysis.pdf")
    print("=" * 60)

    if not os.getenv("GEMINI_API_KEY"):
        print("  ✗ GEMINI_API_KEY not set — aborting.")
        sys.exit(1)

    # Load data from BigQuery
    print("\n  Loading data from BigQuery…")
    shortlisted, arb_df = load_data_from_bigquery()

    if shortlisted.empty:
        print("  ✗ Shortlisted table is empty — aborting.")
        sys.exit(1)

    if "Drug Name" not in shortlisted.columns:
        print("  ✗ 'drug_name' column not found in shortlisted table — check mapping.")
        sys.exit(1)

    # Generate one report per drug
    drugs = sorted(shortlisted["Drug Name"].dropna().unique())
    print(f"\n  Generating {len(drugs)} report(s)…\n")

    uploaded_uris = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, drug in enumerate(drugs, start=1):
            print(f"  [{idx}/{len(drugs)}] {drug}")

            drug_sl  = shortlisted[shortlisted["Drug Name"] == drug].copy()
            drug_arb = (
                arb_df[arb_df["Drug Name"] == drug].copy()
                if not arb_df.empty and "Drug Name" in arb_df.columns
                else pd.DataFrame()
            )

            # Build PDF in a per-drug subfolder to avoid filename collisions
            safe_name = re.sub(r'[^\w\s-]', '', drug).strip().replace(' ', '_')
            drug_tmp  = os.path.join(tmpdir, safe_name)
            os.makedirs(drug_tmp, exist_ok=True)
            pdf_path  = os.path.join(drug_tmp, "PTE_Analysis.pdf")

            try:
                _build_drug_report(drug, drug_sl, drug_arb, pdf_path)
                print(f"      ✓ PDF built")
            except Exception as exc:
                print(f"      ✗ PDF build failed: {exc}")
                continue

            print(f"      Uploading to GCS …", end=" ", flush=True)
            try:
                uri = upload_to_gcs(pdf_path, drug)
                uploaded_uris.append(uri)
            except Exception as exc:
                print(f"FAILED\n      ✗ {exc}")

    print(f"\n{'='*60}")
    print(f"  ✓ Done — {len(uploaded_uris)}/{len(drugs)} PDF(s) uploaded")
    for uri in uploaded_uris:
        print(f"    {uri}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
