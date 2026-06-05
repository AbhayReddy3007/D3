"""
blocking_report_ai.py
──────────────────────
AI-powered 2-page blocking analysis report.

Feeds patent data from the ADK pipeline cache to Gemini 2.0 Flash,
which writes a concise analyst-quality report. The output is rendered as a
strictly 2-page PDF.

Three sections:
  1. Overall Analysis  — totals, category breakdown, headline takeaway
  2. Blocking Patents  — key patents highlighted with reasoning
  3. Non-Blocking Patents — patterns, why they don't block

Usage:
    python blocking_report_ai.py semaglutide
    python blocking_report_ai.py --excel patent_exports/semaglutide_20260409.xlsx
    python blocking_report_ai.py semaglutide -o reports/custom_name.pdf
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import pandas as pd
from google import genai
from google.genai import types

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


# ─────────────────────────────────────────────
# Auth (same pattern as rest of pipeline)
# ─────────────────────────────────────────────

load_dotenv(override=True)

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_TABLE_ID   = "Master_LOE"
BQ_LOCATION   = os.getenv("BQ_LOCATION", "asia-south1")

GCS_BUCKET    = os.getenv("GCS_BUCKET", "cognito-prod")
GCS_BASE_PATH = "Cognito_new/reports"
GCS_SUBFOLDER = "IP"
GCS_FILENAME  = "Blocking_Analysis.pdf"

CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")


def _get_credentials():
    """Use service account file if available, else ADC (Cloud Run)."""
    from google.oauth2 import service_account as _sa
    if CREDENTIALS_PATH and Path(CREDENTIALS_PATH).exists():
        return _sa.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None


# ─────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────

_gemini_client = None

def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ─────────────────────────────────────────────
# Colours & Styles
# ─────────────────────────────────────────────

_DARK_BLUE  = colors.HexColor("#1B2A4A")
_MED_BLUE   = colors.HexColor("#2C5F8A")
_LIGHT_BLUE = colors.HexColor("#E8F0FE")
_RED        = colors.HexColor("#B3261E")
_GREEN      = colors.HexColor("#1A7A3A")
_GREY       = colors.HexColor("#5F6368")
_LIGHT_GREY = colors.HexColor("#F1F3F4")
_WHITE      = colors.white


def _build_styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("T", parent=base["Title"], fontSize=20, leading=24,
                                 textColor=_DARK_BLUE, spaceAfter=2)
    s["subtitle"] = ParagraphStyle("ST", parent=base["Heading2"], fontSize=13,
                                    leading=16, textColor=_MED_BLUE, spaceAfter=4)
    s["meta"] = ParagraphStyle("M", parent=base["Normal"], fontSize=8.5, leading=11,
                                textColor=_GREY, spaceAfter=8)
    s["heading"] = ParagraphStyle("H", parent=base["Heading2"], fontSize=12,
                                   leading=15, textColor=_DARK_BLUE,
                                   spaceBefore=10, spaceAfter=4)
    s["body"] = ParagraphStyle("B", parent=base["Normal"], fontSize=9, leading=12.5,
                                textColor=colors.black, alignment=TA_JUSTIFY, spaceAfter=4)
    s["bullet"] = ParagraphStyle("BL", parent=base["Normal"], fontSize=9, leading=12.5,
                                  leftIndent=14, bulletIndent=3, spaceAfter=2)
    s["th"] = ParagraphStyle("TH", parent=base["Normal"], fontSize=8, leading=10,
                              textColor=_WHITE, alignment=TA_CENTER)
    s["td"] = ParagraphStyle("TD", parent=base["Normal"], fontSize=8, leading=10,
                              alignment=TA_CENTER)
    s["td_left"] = ParagraphStyle("TDL", parent=base["Normal"], fontSize=8, leading=10,
                                   alignment=TA_LEFT)
    s["footer"] = ParagraphStyle("F", parent=base["Normal"], fontSize=7, leading=9,
                                  textColor=_GREY, alignment=TA_CENTER)
    return s


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def _g(p, *keys, default="N/A"):
    for k in keys:
        v = p.get(k)
        if v is not None and str(v).strip().lower() not in ("", "nan", "none", "n/a"):
            return v
    return default


def _load_from_analysis_cache(drug_name: str) -> List[Dict]:
    cache_dir = Path(os.getenv("ANALYSIS_CACHE_DIR", Path(__file__).parent / "analysis_cache"))
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    drug_dir = cache_dir / safe
    if not drug_dir.exists():
        return []
    patents = []
    for f in sorted(drug_dir.glob("*.json")):
        try:
            patents.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return patents


def _load_from_results_cache(drug_name: str) -> List[Dict]:
    cache_dir = Path(os.getenv("RESULTS_CACHE_DIR", Path(__file__).parent / "results_cache"))
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    path = cache_dir / f"{safe}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("patents", [])
    except Exception:
        return []


def _load_from_bigquery(drug_name: str = None) -> List[Dict]:
    """Load patent data from Google BigQuery Master_LOE table."""
    from google.cloud import bigquery

    credentials = _get_credentials()
    client = bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )

    table_ref = f"`{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`"
    if drug_name:
        safe_drug = drug_name.replace("'", "\\'")
        query = f"SELECT DISTINCT * FROM {table_ref} WHERE Drug_Name = '{safe_drug}'"
    else:
        query = f"SELECT DISTINCT * FROM {table_ref}"

    print(f"[BQ] Running query: {query}")
    df = client.query(query).to_dataframe()
    print(f"[BQ] Loaded {len(df)} distinct rows from BigQuery.")

    # Normalise column names: BQ uses underscores, code expects spaces
    df.columns = [c.strip().replace("_", " ") for c in df.columns]
    return df.to_dict(orient="records")


def _load_from_excel(excel_path: str) -> List[Dict]:
    print(f"[DEBUG] Opening Excel: {excel_path}")
    xl = pd.ExcelFile(excel_path)
    df = pd.read_excel(excel_path, sheet_name="Combined")
    print(f"[EXCEL] {len(df)} rows loaded from sheet 'Combined'")
    return df.to_dict(orient="records")


def _filter_patents(patents: List[Dict]) -> List[Dict]:
    """Return all patents without any jurisdiction filtering."""
    return patents


def load_patents(drug_name: str = None, excel_path: str = None) -> tuple:
    """
    Load patents from BigQuery (primary source).
    Falls back to local JSON caches, then Excel if explicitly provided.
    """
    date = datetime.now().strftime("%Y-%m-%d")

    # ── Primary: BigQuery ──
    try:
        patents = _load_from_bigquery(drug_name)
        if patents:
            name = str(patents[0].get("Drug Name", drug_name or "Unknown"))
            print(f"[BQ] Loaded {len(patents)} patents for '{name}'")
            return patents, name, date
        else:
            print("[BQ] Query returned no rows — falling back to cache/Excel")
    except Exception as e:
        print(f"[BQ] BigQuery load failed: {e} — falling back to cache/Excel")

    # ── Fallback 1: local JSON caches ──
    if drug_name:
        for loader in (_load_from_results_cache, _load_from_analysis_cache):
            patents = loader(drug_name)
            if patents:
                return patents, drug_name, date

    # ── Fallback 2: Excel (explicit path only) ──
    if excel_path:
        try:
            patents = _load_from_excel(excel_path)
        except Exception as e:
            print(f"[DEBUG] Excel load failed: {e}")
            patents = []
        if patents:
            name = str(patents[0].get("Drug Name", drug_name or "Unknown"))
            stem = Path(excel_path).stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
                try:
                    date = datetime.strptime(parts[1], "%Y%m%d").strftime("%Y-%m-%d")
                except ValueError:
                    pass
            return patents, name, date

    return [], drug_name or "Unknown", date


# ─────────────────────────────────────────────
# Build structured data summary for Gemini
# ─────────────────────────────────────────────

def _build_patent_summary(patents: List[Dict]) -> str:
    blocking     = [p for p in patents if _g(p, "tag", "Tag") == "BLOCKING"]
    non_blocking = [p for p in patents if _g(p, "tag", "Tag") == "NON-BLOCKING"]

    lines = []
    lines.append(f"TOTAL PATENTS: {len(patents)}")
    lines.append(f"BLOCKING: {len(blocking)}")
    lines.append(f"NON-BLOCKING: {len(non_blocking)}")

    cat_counts = defaultdict(lambda: {"blocking": 0, "non_blocking": 0})
    for p in patents:
        cat = _g(p, "claim_category", "Step 1 Claim Category", default="Unknown")
        tag = _g(p, "tag", "Tag")
        if tag == "BLOCKING":
            cat_counts[cat]["blocking"] += 1
        else:
            cat_counts[cat]["non_blocking"] += 1

    lines.append("\nCATEGORY BREAKDOWN:")
    for cat, counts in sorted(cat_counts.items()):
        lines.append(f"  {cat}: {counts['blocking']} blocking, {counts['non_blocking']} non-blocking")

    lines.append("\n--- BLOCKING PATENTS ---")
    for p in blocking:
        pn     = _g(p, "patent_number", "Patent Number")
        jur    = _g(p, "jurisdiction", "Jurisdiction")
        cat    = _g(p, "claim_category", "Step 1 Claim Category")
        bcat   = _g(p, "blocking_category", "Blocking Category")
        filed  = _g(p, "filing_date", "Filing Date")
        grant  = _g(p, "grant_date", "Grant Date")
        reason = _g(p, "reason", "Reason")
        s3_ev  = _g(p, "step3_evidence_summary", "Step 3 Evidence Summary")

        lines.append(f"\n  PATENT: {pn} ({jur})")
        lines.append(f"  Claim Category: {cat}")
        lines.append(f"  Blocking Category: {bcat}")
        lines.append(f"  Filed: {filed} | Granted: {grant}")
        lines.append(f"  Reason: {reason}")
        if str(s3_ev) != "N/A":
            lines.append(f"  Scientific Evidence: {str(s3_ev)[:300]}")

    lines.append("\n--- NON-BLOCKING PATENTS ---")
    for p in non_blocking:
        pn     = _g(p, "patent_number", "Patent Number")
        jur    = _g(p, "jurisdiction", "Jurisdiction")
        cat    = _g(p, "claim_category", "Step 1 Claim Category")
        filed  = _g(p, "filing_date", "Filing Date")
        reason = _g(p, "reason", "Reason")
        s3_bar = _g(p, "step3_is_technical_barrier", "Step 3 Technical Barrier")
        s4_ind = _g(p, "step4_is_blocking_indicator", "Step 4 Blocking Indicator")

        exit_step = "Step 2"
        if str(s3_bar).lower() == "false":
            exit_step = "Step 3"
        elif str(s4_ind).lower() == "false":
            exit_step = "Step 4"
        elif str(_g(p, "tag", "Tag")) == "NON-BLOCKING" and str(s4_ind).lower() == "true":
            exit_step = "Step 5"

        lines.append(f"\n  PATENT: {pn} ({jur})")
        lines.append(f"  Claim Category: {cat}")
        lines.append(f"  Filed: {filed}")
        lines.append(f"  Exited At: {exit_step}")
        lines.append(f"  Reason: {str(reason)[:200]}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Gemini analysis call
# ─────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a senior pharmaceutical patent analyst writing a concise
executive report on the blocking patent landscape for {drug_name}.

Below is the complete patent analysis data from our automated pipeline.

{patent_summary}

WRITE A REPORT WITH EXACTLY 3 SECTIONS:

SECTION 1 — OVERALL ANALYSIS (1 short paragraph)
Write a concise executive summary. State the total patents analysed,
how many are blocking vs non-blocking, and the key claim categories involved.
End with a one-sentence headline takeaway about the strength of the patent
protection for this drug.

SECTION 2 — BLOCKING PATENTS (2-4 paragraphs)
Analyse the blocking patents. Highlight the most significant ones and explain
WHY they block. Group by claim category where useful. For each key patent:
  - The patent number and jurisdiction
  - What it protects and why it creates a barrier
  - The scientific/regulatory evidence supporting the blocking classification
Focus on insight, not listing.

SECTION 3 — NON-BLOCKING PATENTS (1-2 paragraphs)
Summarise patterns in the non-blocking patents. At which analysis steps did
most exit? What categories are they in? Why don't they block?

FORMATTING RULES:
- Start each section with: "SECTION 1: OVERALL ANALYSIS" etc.
- Plain text paragraphs. No bullet points, no markdown, no tables.
- Keep under 800 words. Must fit on 2 PDF pages.
- Professional analytical tone.
- Patent format: US10159713B2 (US) or EP3296310B1 (EP).
- Start directly with SECTION 1. No preamble or sign-off.
"""


def _call_gemini(drug_name: str, patent_summary: str) -> Optional[str]:
    gc     = _get_gemini()
    prompt = ANALYSIS_PROMPT.format(drug_name=drug_name, patent_summary=patent_summary)

    print(f"[GEMINI] Requesting analysis for {drug_name}...")
    try:
        response = gc.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=204800),
        )
        text = (response.text or "").strip()
        print(f"[GEMINI] Received {len(text)} chars")
        return text
    except Exception as e:
        print(f"[GEMINI] Error: {e}")
        return None


# ─────────────────────────────────────────────
# Parse Gemini output into sections
# ─────────────────────────────────────────────

def _parse_sections(text: str) -> Dict[str, str]:
    sections = {"overall": "", "blocking": "", "non_blocking": ""}
    parts = re.split(r"SECTION\s*\d\s*[:—\-]\s*", text, flags=re.IGNORECASE)
    section_keys = ["overall", "blocking", "non_blocking"]
    for i, part in enumerate(parts[1:]):
        if i < len(section_keys):
            cleaned = re.sub(r"^[A-Z\s&\-]+\n", "", part.strip(), count=1).strip()
            sections[section_keys[i]] = cleaned
    return sections


# ─────────────────────────────────────────────
# GCS Upload (same destination as other reports)
# ─────────────────────────────────────────────

def _upload_to_gcs(local_pdf: str, drug_name: str) -> str:
    """Upload PDF to gs://{GCS_BUCKET}/Cognito_new/reports/{drug}/IP/Blocking_Analysis.pdf"""
    from google.cloud import storage

    safe_drug = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    blob_name = f"{GCS_BASE_PATH}/{safe_drug}/{GCS_SUBFOLDER}/{GCS_FILENAME}"
    gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)

    # Archive existing version
    try:
        existing = bucket.blob(blob_name)
        if existing.exists():
            ts_str = (existing.updated or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
            archive_name = (
                f"{GCS_BASE_PATH}/{safe_drug}/{GCS_SUBFOLDER}"
                f"/archive/{ts_str}_{GCS_FILENAME}"
            )
            bucket.copy_blob(existing, bucket, archive_name)
            print(f"    archived prior version -> gs://{GCS_BUCKET}/{archive_name}")
    except Exception as e:
        print(f"    [WARN] archive step failed: {e}")

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_pdf, content_type="application/pdf")
    print(f"    Uploaded -> {gcs_uri}")
    return gcs_uri


# ─────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────

def _text_to_paragraphs(text: str, style) -> List:
    elements = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        para = para.replace("\n", " ")
        # Bold patent numbers
        for prefix in ("US", "EP", "CN", "IN", "BR", "AU", "RU", "RK", "CA", "TW", "MX", "JP"):
            para = re.sub(rf"\b({prefix}\d{{6,}}[A-Z]\d*)\b", r"<b>\1</b>", para)
        elements.append(Paragraph(para, style))
    return elements


def generate_ai_blocking_report(
    patents:       List[Dict],
    drug_name:     str,
    analysis_date: str = None,
    output_path:   str = None,
) -> str:
    analysis_date = analysis_date or datetime.now().strftime("%Y-%m-%d")
    styles = _build_styles()

    all_patents = _filter_patents(patents)
    if not all_patents:
        print("[ERROR] No patents found")
        return ""

    blocking     = [p for p in all_patents if _g(p, "tag", "Tag") == "BLOCKING"]
    non_blocking = [p for p in all_patents if _g(p, "tag", "Tag") == "NON-BLOCKING"]
    patent_summary = _build_patent_summary(all_patents)

    analysis_text = _call_gemini(drug_name, patent_summary)
    if not analysis_text:
        print("[ERROR] Gemini did not return analysis")
        return ""

    sections = _parse_sections(analysis_text)

    # ── Build PDF ──
    story = []

    story.append(Paragraph(drug_name, styles["title"]))
    story.append(Paragraph('Blocking Patent Analysis', styles["subtitle"]))
    story.append(Paragraph(
        f"Analysis Date: {analysis_date}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Patents Analysed: {len(all_patents)}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<font color='{_RED.hexval()}'>Blocking: {len(blocking)}</font>&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<font color='{_GREEN.hexval()}'>Non-Blocking: {len(non_blocking)}</font>",
        styles["meta"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_MED_BLUE, spaceAfter=8))

    # Category breakdown table
    cat_counts = defaultdict(lambda: [0, 0])
    for p in all_patents:
        cat = _g(p, "claim_category", "Step 1 Claim Category", default="Other")
        if _g(p, "tag", "Tag") == "BLOCKING":
            cat_counts[cat][0] += 1
        else:
            cat_counts[cat][1] += 1

    th = styles["th"]
    td = styles["td"]
    tl = styles["td_left"]
    t_rows = [[Paragraph("<b>Category</b>", th), Paragraph("<b>Blocking</b>", th),
                Paragraph("<b>Non-Blocking</b>", th)]]
    for cat in sorted(cat_counts.keys()):
        b, nb = cat_counts[cat]
        t_rows.append([
            Paragraph(cat, tl),
            Paragraph(f'<font color="{_RED.hexval()}">{b}</font>' if b else "0", td),
            Paragraph(f'<font color="{_GREEN.hexval()}">{nb}</font>' if nb else "0", td),
        ])

    cat_table = Table(t_rows, colWidths=[200, 70, 85])
    cat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _DARK_BLUE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LIGHT_GREY]),
        ("BOX",           (0, 0), (-1, -1), 0.5, _MED_BLUE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D0D0")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(cat_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Overall Analysis", styles["heading"]))
    story.extend(_text_to_paragraphs(sections.get("overall", "Analysis not available."), styles["body"]))

    story.append(Paragraph("2. Blocking Patents", styles["heading"]))
    story.extend(_text_to_paragraphs(sections.get("blocking", "No blocking patent analysis available."), styles["body"]))

    story.append(Paragraph("3. Non-Blocking Patents", styles["heading"]))
    story.extend(_text_to_paragraphs(sections.get("non_blocking", "No non-blocking analysis available."), styles["body"]))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GREY, spaceAfter=3))
    story.append(Paragraph(
        f"Report Date: {datetime.now().strftime('%d-%b-%Y')}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Analysis Date: {analysis_date}",
        styles["footer"],
    ))

    # Write PDF locally then upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        doc = SimpleDocTemplate(
            tmp_path, pagesize=A4,
            topMargin=16 * mm, bottomMargin=12 * mm,
            leftMargin=16 * mm, rightMargin=16 * mm,
            title=f"{drug_name} — Blocking Patent Analysis",
            author="ADK Pipeline",
        )
        doc.build(story)
        print(f"[REPORT] PDF built: {tmp_path}")

        gcs_uri = _upload_to_gcs(tmp_path, drug_name)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return gcs_uri


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI-powered Blocking Patent Analysis PDF -> GCS"
    )
    parser.add_argument("drug_name", nargs="?", default=None,
                        help="Drug name (reads from BigQuery/cache)")
    parser.add_argument("--excel", default=None,
                        help="Path to pipeline Excel file")
    args = parser.parse_args()

    if not args.drug_name and not args.excel:
        print("[INFO] No drug name or --excel provided — loading all drugs from BigQuery")

    patents, drug_name, analysis_date = load_patents(
        drug_name=args.drug_name, excel_path=args.excel
    )

    if not patents:
        print(f"[ERROR] No patent data found")
        sys.exit(1)

    all_patents = _filter_patents(patents)
    blocking = [p for p in all_patents if _g(p, "tag", "Tag") == "BLOCKING"]
    non_blocking = [p for p in all_patents if _g(p, "tag", "Tag") == "NON-BLOCKING"]

    print(f"[INFO] {drug_name}: {len(all_patents)} patents "
          f"({len(blocking)} blocking, {len(non_blocking)} non-blocking)")

    result = generate_ai_blocking_report(all_patents, drug_name, analysis_date)
    if result:
        print(f"[DONE] {result}")
