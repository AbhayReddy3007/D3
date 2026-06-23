"""
IPD Patent Analysis — Per-Drug Word Report Generator (Gemini-powered)
=====================================================================
Reads data from Google BigQuery tables:
  - shortlisted_secondary_patents_table  →  replaces "Shortlisted" sheet
  - arbitrage_summary_table              →  replaces "Arbitrage Summary" sheet

All BQ columns are snake_case and are mapped back to the original
Title Case names used throughout the report generation logic.

API key read from .env  →  GEMINI_API_KEY
Model: gemini-2.5-flash
"""


import os
import sys
import io
import re
import time
import tempfile

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date, datetime, timezone
from google.cloud import bigquery
from google.oauth2 import service_account
from google import genai
from google.genai import types

from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image,
)
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
    "rationale":                    "Rationale",
    "created_at":                   "Created At",
    "updated_at":                   "Updated At",
}



# ═══════════════════════════════════════════════════════════════════════════
#  BIGQUERY SCHEMA MIGRATION  —  create columns if absent
# ═══════════════════════════════════════════════════════════════════════════
def _ensure_arbitrage_columns(client: bigquery.Client) -> None:
    """
    Add 'rationale' (STRING), 'created_at' (TIMESTAMP), and 'updated_at'
    (TIMESTAMP) columns to arbitrage_summary_table if they do not already
    exist, then populate them appropriately.

    - created_at  : set once (first run) and never overwritten.
    - updated_at  : set to CURRENT_TIMESTAMP() on every run.
    """
    full_table = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{ARBITRAGE_TABLE}"

    # Fetch current schema
    table = client.get_table(full_table)
    existing = {f.name for f in table.schema}

    ddl_statements = []

    if "rationale" not in existing:
        ddl_statements.append(
            f"ALTER TABLE `{full_table}` ADD COLUMN IF NOT EXISTS rationale STRING"
        )

    if "created_at" not in existing:
        ddl_statements.append(
            f"ALTER TABLE `{full_table}` ADD COLUMN IF NOT EXISTS created_at TIMESTAMP"
        )

    if "updated_at" not in existing:
        ddl_statements.append(
            f"ALTER TABLE `{full_table}` ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"
        )

    for stmt in ddl_statements:
        print(f"  [BQ DDL] {stmt}")
        client.query(stmt).result()
        print(f"           ✓ Done")

    # Populate rationale where NULL
    update_rationale = f"""
        UPDATE `{full_table}`
        SET rationale = CONCAT(
            'Arbitrage Score is ', CAST(arbitrage_score AS STRING),
            ' because of ', CAST(arbitrage_signal AS STRING)
        )
        WHERE rationale IS NULL
    """
    print("  [BQ] Populating rationale column …")
    client.query(update_rationale).result()
    print("       ✓ rationale populated")

    # Populate created_at only where NULL (i.e. first time this row is seen)
    update_created_at = f"""
        UPDATE `{full_table}`
        SET created_at = CURRENT_TIMESTAMP()
        WHERE created_at IS NULL
    """
    print("  [BQ] Populating created_at column (first-run rows only) …")
    client.query(update_created_at).result()
    print("       ✓ created_at populated")

    # Always refresh updated_at to reflect the current run
    update_updated_at = f"""
        UPDATE `{full_table}`
        SET updated_at = CURRENT_TIMESTAMP()
        WHERE TRUE
    """
    print("  [BQ] Refreshing updated_at column (all rows) …")
    client.query(update_updated_at).result()
    print("       ✓ updated_at refreshed")


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

    # Ensure rationale + created_at + updated_at columns exist in BQ
    print("  Ensuring arbitrage table schema …")
    _ensure_arbitrage_columns(client)

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


def _call_gemini(prompt: str, retries: int = 3, backoff: float = 2.5) -> str:
    for attempt in range(retries):
        try:
            resp = _get_genai_client().models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
            )
            return (resp.text or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"    ⚠ Gemini error ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                print(f"    ✗ Gemini failed after {retries} attempts: {e}")
                return "Narrative unavailable."


# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
BRAND_BLUE     = RGBColor(0x2F, 0x55, 0x97)
ACCENT_BLUE    = RGBColor(0x44, 0x72, 0xC4)
LIGHT_BLUE_HEX = "DCE6F1"
HEADER_HEX     = "2F5597"
WHITE_HEX      = "FFFFFF"

SCORE_HEX = {5: "2ECC71", 4: "27AE60", 3: "F39C12", 2: "E67E22", 1: "E74C3C"}
SIGNAL_HEX = {
    "Immediate opportunity": "2ECC71",
    "Strong arbitrage":      "27AE60",
    "Meaningful arbitrage":  "F39C12",
    "Limited arbitrage":     "E67E22",
    "No arbitrage":          "E74C3C",
    "Reference market":      "3498DB",
}
RATING_HEX = {
    "Very Strong Geographic Arbitrage Opportunity": "2ECC71",
    "Strong Geographic Arbitrage Opportunity":      "27AE60",
    "Moderate Geographic Arbitrage Opportunity":    "F39C12",
    "Limited Geographic Arbitrage Opportunity":     "E67E22",
    "FAIL — No viable geographic arbitrage":        "E74C3C",
}
JUR_FULL = {
    "CN": "China", "IN": "India", "BR": "Brazil",
    "AU": "Australia", "RU": "Russia", "US": "United States",
    "JP": "Japan", "KR": "South Korea", "TW": "Taiwan",
    "CA": "Canada", "MX": "Mexico",
}


# ═══════════════════════════════════════════════════════════════════════════
#  XML / STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
BORDER_OPTS = {"val": "single", "sz": 4, "color": "CCCCCC"}



# ── Reportlab color palette ──
_NAVY       = rl_colors.HexColor("#2F5597")
_BLUE       = rl_colors.HexColor("#2F5496")
_LIGHT_BLUE = rl_colors.HexColor("#D6E4F0")
_LGREY      = rl_colors.HexColor("#F2F2F2")
_WHITE      = rl_colors.white

def _rl_styles():
    b = getSampleStyleSheet()
    return {
        "title":  ParagraphStyle("T",  parent=b["Title"],    fontSize=18, leading=22, textColor=_NAVY, spaceAfter=2),
        "meta":   ParagraphStyle("M",  parent=b["Normal"],   fontSize=9, leading=11, textColor=rl_colors.HexColor("#666666"), spaceAfter=6),
        "h1":     ParagraphStyle("H1", parent=b["Heading2"], fontSize=12, leading=15, textColor=_NAVY, spaceBefore=8, spaceAfter=4),
        "h2":     ParagraphStyle("H2", parent=b["Heading3"], fontSize=11, leading=14, textColor=_NAVY, spaceBefore=6, spaceAfter=2),
        "body":   ParagraphStyle("BD", parent=b["Normal"],   fontSize=9, leading=12, alignment=TA_JUSTIFY, spaceAfter=4),
        "th":     ParagraphStyle("TH", parent=b["Normal"],   fontSize=8, leading=10, textColor=_WHITE, alignment=TA_CENTER),
        "td":     ParagraphStyle("TD", parent=b["Normal"],   fontSize=8, leading=10, alignment=TA_CENTER),
        "tdl":    ParagraphStyle("TL", parent=b["Normal"],   fontSize=8, leading=10, alignment=TA_LEFT),
        "footer": ParagraphStyle("FT", parent=b["Normal"],   fontSize=7, leading=9, textColor=rl_colors.HexColor("#999999"), alignment=TA_CENTER),
    }


def _esc(text):
    """Escape text for reportlab Paragraph XML."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rl_table(headers, data_rows, col_widths, st, signal_col=-1, score_col=-1):
    """Generic reportlab table with navy header row."""
    tbl_rows = [[Paragraph(f"<b>{h}</b>", st["th"]) for h in headers]]
    for ri, dr in enumerate(data_rows):
        row = []
        for ci, val in enumerate(dr):
            s = _esc(str(val)) if val is not None and str(val) not in ("nan","None","") else "—"
            if ci == signal_col:
                hex_c = SIGNAL_HEX.get(s, "BDC3C7")
                row.append(Paragraph(f'<font color="#{hex_c}"><b>{s}</b></font>', st["td"]))
            elif ci == score_col:
                try:
                    si = int(round(float(val)))
                    hex_c = SCORE_HEX.get(si, "BDC3C7")
                    row.append(Paragraph(f'<font color="#{hex_c}"><b>{s}</b></font>', st["td"]))
                except (ValueError, TypeError):
                    row.append(Paragraph(s, st["td"]))
            else:
                row.append(Paragraph(s, st["td"]))
        tbl_rows.append(row)
    t = Table(tbl_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.5, _BLUE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, rl_colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LGREY]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def _fig_to_image(fig, width=5.0*inch):
    """Convert matplotlib figure to reportlab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    pil_img = PILImage.open(buf)
    w_px, h_px = pil_img.size
    aspect = h_px / w_px
    return Image(buf, width=width, height=width * aspect)


def _render_narrative_rl(narrative: str, story, st):
    """Parse LLM narrative into reportlab Paragraphs."""
    pattern = "(" + "|".join(re.escape(h) for h in NARRATIVE_HEADERS) + ")"
    parts   = re.split(pattern, narrative)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "STRATEGIC RECOMMENDATIONS" in part.upper() or "RECOMMENDATION" in part.upper():
            continue
        if part in NARRATIVE_HEADERS:
            story.append(Paragraph(f"<b>{_esc(part.title())}</b>", st["h2"]))
        else:
            story.append(Paragraph(_esc(part), st["body"]))


NARRATIVE_HEADERS = [
    "PATENT LANDSCAPE OVERVIEW",
    "GEOGRAPHIC ARBITRAGE ANALYSIS",
    "KEY PROTECTION GAPS",
]


def _build_drug_narrative_prompt(drug, drug_sl, drug_arb) -> str:
    exp_lines = []
    for _, r in drug_sl.iterrows():
        jur    = JUR_FULL.get(str(r.get("Jurisdiction", "")), str(r.get("Jurisdiction", "")))
        cat    = r.get("Step 1 Claim Category", "")
        expiry = str(r.get("Adjusted Expiry (with PTE)", "N/A"))
        gap    = str(r.get("Expiry Gap (Years)", "N/A"))
        pte_st = r.get("PTE Status", "")
        pte_m  = r.get("PTE Months (Granted)", "0")
        pat_no = r.get("Patent Number", "N/A")
        exp_lines.append(
            f"  {jur} | Patent: {pat_no} | {cat} | Expiry: {expiry} | Gap: {gap} yrs"
            f" | PTE: {pte_st} ({pte_m} months)"
        )

    arb_lines = []
    dim4_score  = "N/A"
    dim4_rating = "N/A"
    if not drug_arb.empty:
        dim4_score  = drug_arb["Dimension IV Score"].iloc[0]
        dim4_rating = drug_arb["Dimension IV Rating"].iloc[0]
        for _, r in drug_arb.iterrows():
            jur         = JUR_FULL.get(str(r.get("Jurisdiction", "")), str(r.get("Jurisdiction", "")))
            loe         = r.get("Product LOE (Year)", "N/A")
            gap_us      = r.get("Gap vs US (Years)", "N/A")
            gap_longest = r.get("Gap vs Longest LOE (Years)", "N/A")
            signal      = r.get("Arbitrage Signal", "N/A")
            gap_kp      = r.get("Key Protection Gap", "N/A")
            arb_lines.append(
                f"  {jur} | LOE: {loe} | Gap vs US: {gap_us} yrs"
                f" | Gap vs Longest LOE: {gap_longest} yrs"
                f" | Signal: {signal} | Protection gap: {gap_kp}"
            )

    jurs_in_data = set()
    for _, r in drug_sl.iterrows():
        jurs_in_data.add(str(r.get("Jurisdiction", "")))
    if not drug_arb.empty:
        for _, r in drug_arb.iterrows():
            jurs_in_data.add(str(r.get("Jurisdiction", "")))

    all_target_jurs   = {"CN", "IN", "BR", "AU", "RU", "US", "CA", "JP", "MX", "TW", "KR"}
    jurs_present      = jurs_in_data & all_target_jurs
    jurs_full_names   = ", ".join(sorted(JUR_FULL.get(j, j) for j in all_target_jurs))
    jurs_analysed_names = ", ".join(sorted(JUR_FULL.get(j, j) for j in jurs_present))

    return f"""
You are a senior pharmaceutical patent analyst writing a drug-level patent
assessment for an internal IP strategy report.

DRUG: {drug}

PATENT EXPIRY DATA (one row per jurisdiction):
{chr(10).join(exp_lines) if exp_lines else '  No data.'}

GEOGRAPHIC ARBITRAGE DATA (one row per jurisdiction):
{chr(10).join(arb_lines) if arb_lines else '  No data.'}

OVERALL DIMENSION IV SCORE: {dim4_score} — {dim4_rating}

TARGET JURISDICTIONS CONSIDERED: {jurs_full_names}
JURISDICTIONS WITH BLOCKING PATENTS ANALYSED: {jurs_analysed_names}

TASK:
Write a detailed, analytical drug-level narrative (220-280 words) structured
into exactly these three sections. Use the exact plain-text section headers shown
below on their own line (no markdown, no asterisks, no bullets):

PATENT LANDSCAPE OVERVIEW
Write 7-8 sentences summarising the breadth and strength of patent protection
for this drug across all jurisdictions, including the range of expiry years
and whether PTE extensions have been granted.
Do NOT give analysis of US. It is only for reference. Analyse other jurisdictions
based on the US information. Mention the Patent Numbers.
Mention that patents from {jurs_full_names} were considered, but only the blocking
ones were analysed. Jurisdictions not present in the analysis data lacked blocking
patents — do NOT say "no data is available", instead say they were considered but
not included as they lacked blocking patents.
Highlight any jurisdictions with particularly strong or weak patent protection,
and note the overall Dimension IV Score and Rating for the drug.

GEOGRAPHIC ARBITRAGE ANALYSIS
Write 10-12 sentences analysing where the strongest and weakest arbitrage
opportunities exist. Reference specific jurisdictions, LOE years, gap vs US,
gap vs longest LOE, and arbitrage signal ratings. Explain the strategic
significance of the "Gap vs Longest LOE" metric — it measures how many years
earlier a jurisdiction loses exclusivity compared to the jurisdiction with the
longest protection.

KEY PROTECTION GAPS
Write 7-8 sentences identifying any missing claim categories or jurisdictions
with weak or absent protection, and explain what this means for a potential
generic or biosimilar entrant.

Rules:
- Do NOT include any recommendations or strategic advice sections.
- Formal, precise tone. No filler phrases.
- Do NOT give Analysis for US patents. Compare other Patents with the US information.
- Use full jurisdiction names, not abbreviations.
- Do NOT invent any numbers or dates — use only the data provided above.
- Plain text only. No markdown, no bold, no asterisks, no bullet points.
""".strip()


def _chart_drug_loe(drug: str, drug_arb: pd.DataFrame):
    df = drug_arb[drug_arb["Jurisdiction"] != "US"].copy()
    df["LOE"] = pd.to_numeric(df["Product LOE (Year)"], errors="coerce")
    df = df.dropna(subset=["LOE"]).sort_values("LOE")
    if df.empty:
        return None

    today  = date.today().year
    labels = [JUR_FULL.get(j, j) for j in df["Jurisdiction"]]
    chart_colors = ["#" + SIGNAL_HEX.get(s, "BDC3C7") for s in df["Arbitrage Signal"]]

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.55 * len(df) + 1)))
    bars = ax.barh(labels, df["LOE"] - today, left=today,
                   color=chart_colors, edgecolor="white", linewidth=0.8)
    ax.axvline(x=today, color="#E74C3C", linestyle="--", linewidth=1.5, label="Today")
    ax.set_xlabel("Year", fontsize=9)
    ax.set_title(f"{drug} — Loss of Exclusivity by Jurisdiction",
                 fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, loe in zip(bars, df["LOE"]):
        ax.text(bar.get_x() + bar.get_width() - 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(int(loe)), va="center", ha="right",
                fontsize=8, fontweight="bold", color="white")

    patches = [mpatches.Patch(color="#" + v, label=k) for k, v in SIGNAL_HEX.items()]
    ax.legend(handles=patches, fontsize=7, loc="lower right", framealpha=0.8, ncol=1)
    fig.tight_layout()
    return fig


def _build_drug_report(drug, drug_sl, drug_arb, output_dir):
    """Build per-drug PDF report directly using reportlab."""
    st  = _rl_styles()
    pdf_path = os.path.join(output_dir, "Loe_Report(Secondary_Market).pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        topMargin=18*mm, bottomMargin=14*mm, leftMargin=20*mm, rightMargin=20*mm,
        title=f"{drug} — Secondary Market Analysis", author="ADK Pipeline",
    )
    W     = letter[0] - 40*mm
    story = []

    # ── Cover / Title ──
    story.append(Spacer(1, 40))
    story.append(Paragraph(drug, st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_NAVY, spaceAfter=6))
    story.append(Paragraph("Shortlisted Secondary Patents &amp; Geographic Arbitrage Analysis", st["body"]))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y')}", st["meta"]))
    story.append(Spacer(1, 20))

    # ── Dimension IV badge ──
    if not drug_arb.empty:
        dim4_score  = drug_arb["Dimension IV Score"].iloc[0]
        dim4_rating = drug_arb["Dimension IV Rating"].iloc[0]
        rating_hex  = RATING_HEX.get(str(dim4_rating), "BDC3C7")
        badge_rows = [[
            Paragraph("<b>Dimension IV Score</b>", st["th"]),
            Paragraph(f"<b>{dim4_score}  —  {dim4_rating}</b>", st["th"]),
        ]]
        badge = Table(badge_rows, colWidths=[W*0.3, W*0.7])
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(0,0), _NAVY),
            ("BACKGROUND", (1,0),(1,0), rl_colors.HexColor(f"#{rating_hex}")),
            ("TEXTCOLOR",  (0,0),(-1,0), _WHITE),
            ("ALIGN",      (0,0),(-1,0), "CENTER"),
            ("VALIGN",     (0,0),(-1,0), "MIDDLE"),
            ("TOPPADDING", (0,0),(-1,0), 6),
            ("BOTTOMPADDING",(0,0),(-1,0), 6),
            ("BOX",        (0,0),(-1,-1), 0.5, _BLUE),
        ]))
        story.append(badge)
        story.append(Spacer(1, 10))

    # ── Shortlisted Patents Table ──
    if not drug_sl.empty:
        story.append(Paragraph("Shortlisted Secondary Patents", st["h1"]))
        sl_cols = ["Patent Number", "Jurisdiction", "Step 1 Claim Category",
                   "Adjusted Expiry (with PTE)", "Expiry Gap (Years)"]
        sl_cols = [c for c in sl_cols if c in drug_sl.columns]
        sl_data = [[str(row.get(c,"")) for c in sl_cols] for _,row in drug_sl.iterrows()]
        cw = [W/len(sl_cols)] * len(sl_cols)
        story.append(_rl_table(sl_cols, sl_data, cw, st))
        story.append(Spacer(1, 10))

    # ── Geographic Arbitrage Map ──
    if not drug_arb.empty:
        story.append(Paragraph("Geographic Arbitrage Map", st["h1"]))
        arb_cols = ["Jurisdiction", "Product LOE (Year)", "Gap vs US (Years)",
                    "Gap vs Longest LOE (Years)", "Key Protection Gap",
                    "Arbitrage Score", "Arbitrage Signal"]
        arb_cols = [c for c in arb_cols if c in drug_arb.columns]
        sig_idx   = arb_cols.index("Arbitrage Signal") if "Arbitrage Signal" in arb_cols else -1
        score_idx = arb_cols.index("Arbitrage Score")  if "Arbitrage Score"  in arb_cols else -1
        arb_data  = [[str(row.get(c,"")) for c in arb_cols] for _,row in drug_arb.iterrows()]
        arb_cw    = [W*0.12, W*0.13, W*0.12, W*0.13, W*0.24, W*0.11, W*0.15][:len(arb_cols)]
        story.append(_rl_table(arb_cols, arb_data, arb_cw, st, signal_col=sig_idx, score_col=score_idx))
        story.append(Spacer(1, 10))

        # Chart
        fig = _chart_drug_loe(drug, drug_arb)
        if fig:
            story.append(_fig_to_image(fig, width=4.8*inch))
            story.append(Spacer(1, 8))

    # ── LLM Narrative ──
    story.append(Paragraph("Analysis", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_NAVY, spaceAfter=6))
    prompt    = _build_drug_narrative_prompt(drug, drug_sl, drug_arb)
    narrative = _call_gemini(prompt)
    time.sleep(1)
    _render_narrative_rl(narrative, story, st)

    # ── Patent Expiry Summary ──
    story.append(PageBreak())
    story.append(Paragraph("Patent Expiry Summary", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_NAVY, spaceAfter=6))
    exp_cols = ["Patent Number", "Jurisdiction", "Step 1 Claim Category",
                "Adjusted Expiry (with PTE)", "Expiry Gap (Years)",
                "PTE Months (Granted)"]
    exp_cols = [c for c in exp_cols if c in drug_sl.columns]
    exp_df   = drug_sl[exp_cols].copy()
    if "Adjusted Expiry (with PTE)" in exp_df.columns:
        exp_df["Adjusted Expiry (with PTE)"] = pd.to_datetime(
            exp_df["Adjusted Expiry (with PTE)"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("Not found")
    exp_data = [[str(row.get(c,"")) for c in exp_cols] for _,row in exp_df.iterrows()]
    exp_cw   = [W/len(exp_cols)] * len(exp_cols)
    story.append(_rl_table(exp_cols, exp_data, exp_cw, st))

    # ── Footer ──
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>Auto-generated using Gemini 2.5 Flash.</i>", st["footer"]))

    doc.build(story)
    print(f"      ✓ PDF built → {pdf_path}")
    return pdf_path

def upload_to_gcs(local_pdf: str, drug_name: str,
                  bucket_name: str = "cognito-gcs") -> str:
    """Upload PDF to gs://{bucket_name}/Cognito_new/reports/{drug_name}/Loe_Report(Secondary_Market).pdf
    Returns the full GCS URI."""
    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError(
            "google-cloud-storage not installed.\n"
            "Run: pip install google-cloud-storage --break-system-packages"
        )

    safe_drug = drug_name.replace("/", "_").replace(" ", "_")
    blob_name = f"Cognito_new/reports/{safe_drug}/Loe_Report(Secondary_Market).pdf"

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)

    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_filename(local_pdf, content_type="application/pdf")

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"      ✓ Uploaded → {gcs_uri}")
    return gcs_uri


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD SINGLE-DRUG REPORT
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  IPD PER-DRUG REPORT GENERATOR  —  Gemini 2.5 Flash")
    print("  Data source: Google BigQuery")
    print("  Output     : gs://cognito-gcs/Cognito_new/reports/{drug_name}/Loe_Report(Secondary_Market).pdf")
    print("=" * 60)

    if not os.getenv("GEMINI_API_KEY"):
        print("  ✗ GEMINI_API_KEY not set — aborting.")
        sys.exit(1)

    print("\n  Loading data from BigQuery…")
    shortlisted, arb_df = load_data_from_bigquery()

    if shortlisted.empty:
        print("  ✗ Shortlisted table is empty — aborting.")
        sys.exit(1)

    if "Drug Name" not in shortlisted.columns:
        print("  ✗ 'drug_name' column not found — check mapping.")
        sys.exit(1)

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

            drug_tmp = os.path.join(tmpdir, re.sub(r'[^\w\s-]', '', drug).strip().replace(' ', '_'))
            os.makedirs(drug_tmp, exist_ok=True)

            pdf_path = _build_drug_report(drug, drug_sl, drug_arb, drug_tmp)

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
