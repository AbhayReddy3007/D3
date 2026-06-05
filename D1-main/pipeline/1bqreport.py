"""
scoring_report.py - LOE Calculation (Primary Market) Report Generator
"""
import argparse, os, re, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass
load_dotenv(override=True)

import pandas as pd
from google import genai
from google.genai import types
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)

# ─── Constants for flexible column matching ───
FORECASTED_COL = "No Of Forecasted Patents"  # canonical name after normalization

# ─── GCS / BigQuery Configuration (from .env) ───
GCS_BUCKET      = os.getenv("GCS_BUCKET",      "cognito-gcs")
GCS_BASE_PATH   = os.getenv("GCS_BASE_PATH",   "Cognito_new/reports")
GCS_FILE_NAME   = os.getenv("GCS_FILE_NAME",   "LOE_Report(Primary_Market).pdf")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

BQ_PROJECT_ID   = os.getenv("BQ_PROJECT_ID",   "cognito-prod-394707")
BQ_DATASET_ID   = os.getenv("BQ_DATASET_ID",   "cognito_prod_datamart")
BQ_TABLE_ID     = os.getenv("BQ_TABLE_ID",     "Master_LOE")
BQ_LOCATION     = os.getenv("BQ_LOCATION",     "asia-south1")

def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None  # Use ADC (Application Default Credentials)

# ─── Gemini client ───
_gemini = None
def _get_gemini():
    global _gemini
    if _gemini is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit(
                "ERROR: GOOGLE_API_KEY or GEMINI_API_KEY must be set.\n"
                "Add it to your .env file, e.g.:\n"
                "  GOOGLE_API_KEY=your_key_here"
            )
        _gemini = genai.Client(api_key=api_key)
    return _gemini

# ─── PDF Styles ───
_NAVY        = colors.HexColor("#1B2A4A")
_BLUE        = colors.HexColor("#2C5F8A")
_LIGHT_BLUE  = colors.HexColor("#E8F0FE")
_LIGHT_GREEN = colors.HexColor("#E6F4EA")
_RED         = colors.HexColor("#B3261E")
_GREY        = colors.HexColor("#5F6368")
_LGREY       = colors.HexColor("#F1F3F4")
_WHITE       = colors.white

# Score 1 = >13 yrs = Very High Barrier (strongest protection)
# Score 5 = <=6 yrs = Minimal Barrier (weakest protection)
_SCORE_CLR = {5: "#0D652D", 4: "#1A7A3A", 3: "#E8A317", 2: "#D4570A", 1: "#B3261E"}
_SCORE_LBL = {1: "Very High Barrier", 2: "High Barrier", 3: "Moderate Barrier",
              4: "Low Barrier", 5: "Minimal Barrier"}

def _styles():
    b = getSampleStyleSheet()
    return {
        "title":    ParagraphStyle("T",  parent=b["Title"],    fontSize=18, leading=22, textColor=_NAVY, spaceAfter=2),
        "meta":     ParagraphStyle("M",  parent=b["Normal"],   fontSize=8, leading=10, textColor=_GREY, spaceAfter=6),
        "h1":       ParagraphStyle("H1", parent=b["Heading2"], fontSize=11, leading=14, textColor=_NAVY, spaceBefore=4, spaceAfter=2),
        "h2":       ParagraphStyle("H2", parent=b["Heading3"], fontSize=9.5, leading=12, textColor=_BLUE, spaceBefore=3, spaceAfter=1),
        "body":     ParagraphStyle("BD", parent=b["Normal"],   fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=2),
        "legend":   ParagraphStyle("LG", parent=b["Normal"],   fontSize=7, leading=9, textColor=_GREY, spaceAfter=1),
        "th":       ParagraphStyle("TH", parent=b["Normal"],   fontSize=7.5, leading=9, textColor=_WHITE, alignment=TA_CENTER),
        "td":       ParagraphStyle("TD", parent=b["Normal"],   fontSize=7.5, leading=9, alignment=TA_CENTER),
        "tdl":      ParagraphStyle("TL", parent=b["Normal"],   fontSize=7.5, leading=9, alignment=TA_LEFT),
        "td_wrap":  ParagraphStyle("TW", parent=b["Normal"],   fontSize=7.5, leading=10, alignment=TA_LEFT, wordWrap="LTR"),
        "footer":   ParagraphStyle("FT", parent=b["Normal"],   fontSize=7, leading=9, textColor=_GREY, alignment=TA_CENTER),
    }

# ─── Data helpers ───
def _v(row, col, default="N/A"):
    val = row.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    s = str(val).strip()
    return default if s.lower() in ("", "nan", "none", "n/a") else s

def _num(val):
    try: return float(val)
    except: return None

def _clean_year(val_str):
    if val_str == "N/A": return val_str
    n = _num(val_str)
    if n is not None and n == int(n): return str(int(n))
    return val_str

def _fmt(val):
    if val is None: return "N/A"
    return str(int(val)) if val == int(val) else str(round(val, 1))

def _score(avg):
    if avg is None: return None
    if avg <= 6: return 5
    if avg <= 8: return 4
    if avg <= 11: return 3
    if avg <= 13: return 2
    return 1


def _find_forecasted_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        normalized = c.lower().replace(" ", "").replace("_", "")
        if normalized in (
            "noofforecastedpatents", "numberofforecastedpatents",
            "forecastedpatents", "noofforecasted",
        ):
            return c
    return None


def _normalize_forecasted_col(df: pd.DataFrame) -> pd.DataFrame:
    found_col = _find_forecasted_col(df)
    if found_col is not None:
        df[FORECASTED_COL] = pd.to_numeric(
            df[found_col], errors="coerce"
        ).fillna(0).astype(int)
        print(f"  [INFO] Found forecasted column: '{found_col}' -> "
              f"unique values: {sorted(df[FORECASTED_COL].unique())}")
    else:
        print(f"  [WARN] Forecasted patents column not found. "
              f"Available columns: {list(df.columns)}")
        df[FORECASTED_COL] = 0
    return df


def _is_forecasted(row_or_filing_date, filing_date_str=None):
    if isinstance(row_or_filing_date, dict):
        type_val = _v(row_or_filing_date, "Type", "")
        if type_val.strip().lower() == "forecasted":
            return True
        filing_date_str = _v(row_or_filing_date, "Filing Date", "N/A")

    if not filing_date_str or filing_date_str == "N/A":
        return True
    s = str(filing_date_str).strip()
    n = _num(s)
    if n is not None and n == int(n):
        s = str(int(n))
    return bool(re.match(r'^\d{4}$', s))


def _parse_forecasted_pn(pn: str, jurisdiction: str = "") -> str:
    parts = [p.strip() for p in pn.split("+")]
    if len(parts) >= 3 and parts[1].lower() == "forecasted":
        jur   = parts[0].upper()
        ptype = " ".join(parts[2:])
        return f"Forecasted patent for {ptype} in {jur}"
    return pn


def _find_controlling(blocking_df):
    if blocking_df.empty: return None
    tmp = blocking_df.copy()
    tmp["_fs"] = tmp["Filing Date"].astype(str)
    return tmp.sort_values("_fs", ascending=False).iloc[0]


def _is_fc_row(row: dict) -> bool:
    return _v(row, "Type", "").strip().lower() == "forecasted"


def _fc_count(subset_df: pd.DataFrame) -> int:
    if "Type" not in subset_df.columns:
        return 0
    fc_rows = subset_df[subset_df["Type"].str.strip().str.lower() == "forecasted"]
    if fc_rows.empty or FORECASTED_COL not in fc_rows.columns:
        return 0
    return int(fc_rows[FORECASTED_COL].apply(_num).dropna().sum())


def _existing_count(subset_df: pd.DataFrame) -> int:
    if "Type" not in subset_df.columns:
        return len(subset_df)
    return int((subset_df["Type"].str.strip().str.lower() != "forecasted").sum())


# ─── Rationale generator ───

def _generate_rationale(d: Dict) -> str:
    """
    Build a plain-English rationale for the Average Years to Entry and LOE Score
    section. Always produces a non-empty string even when YTE / score are N/A
    (e.g. all patents are forecasted and Years to Entry is not yet calculable).
    """
    si   = _num(d["loe_score"])
    band = _SCORE_LBL.get(int(si), "N/A") if si else "N/A"

    parts = []

    # Score line
    if si:
        parts.append(
            f"LOE Score {d['loe_score']}/5 ({band}): "
            f"avg YTE = ({d['us_yte']} + {d['ep_yte']}) / 2 = {d['avg_yte_us_ep']} yrs"
        )
    else:
        parts.append(
            "LOE Score not yet calculable (Years to Entry unavailable — "
            "controlling patents are forecasted and not yet filed)"
        )

    # Controlling patent details
    ctrl_parts = []
    if d["us_ctrl_pn_display"] != "N/A":
        us_d = f"US controlling patent: {d['us_ctrl_pn_display']}"
        if d["us_ctrl_expiry"] != "N/A":
            us_d += f" (filed {d['us_ctrl_filed']}, expiry {d['us_ctrl_expiry']})"
        elif d["us_ctrl_filed"] != "N/A":
            us_d += f" (filed {d['us_ctrl_filed']})"
        if d["us_yte"] != "N/A":
            us_d += f", YTE {d['us_yte']}"
        ctrl_parts.append(us_d)
    if d["ep_ctrl_pn_display"] != "N/A":
        ep_d = f"EP controlling patent: {d['ep_ctrl_pn_display']}"
        if d["ep_ctrl_expiry"] != "N/A":
            ep_d += f" (filed {d['ep_ctrl_filed']}, expiry {d['ep_ctrl_expiry']})"
        elif d["ep_ctrl_filed"] != "N/A":
            ep_d += f" (filed {d['ep_ctrl_filed']})"
        if d["ep_yte"] != "N/A":
            ep_d += f", YTE {d['ep_yte']}"
        ctrl_parts.append(ep_d)
    if ctrl_parts:
        parts.append("; ".join(ctrl_parts))

    # Portfolio summary (always available)
    fc_note = ""
    if d["n_us_forecasted"] > 0 or d["n_ep_forecasted"] > 0:
        fc_note = (f" (including {d['n_us_forecasted']} US "
                   f"+ {d['n_ep_forecasted']} EP forecasted patents not yet filed)")
    parts.append(
        f"Portfolio: {d['total_us']} US ({d['n_us_blocking']} blocking) "
        f"and {d['total_ep']} EP ({d['n_ep_blocking']} blocking) patents{fc_note}"
    )

    return ". ".join(parts) + "."


def _extract_drug_data(df: pd.DataFrame) -> Dict:
    us = df[df["Jurisdiction"].str.upper() == "US"]
    ep = df[df["Jurisdiction"].str.upper() == "EP"]
    us_b = us[us["Tag"] == "BLOCKING"];   ep_b = ep[ep["Tag"] == "BLOCKING"]
    us_n = us[us["Tag"] == "NON-BLOCKING"]; ep_n = ep[ep["Tag"] == "NON-BLOCKING"]

    us_ctrl = _find_controlling(us_b)
    ep_ctrl = _find_controlling(ep_b)

    us_yte = _num(_v(us_ctrl, "Years to Entry") if us_ctrl is not None else None)
    ep_yte = _num(_v(ep_ctrl, "Years to Entry") if ep_ctrl is not None else None)
    vals = [v for v in [us_yte, ep_yte] if v is not None]
    avg  = round(sum(vals)/len(vals), 1) if vals else None

    sample = df.iloc[0]
    def _ctrl(c, col, default="N/A"):
        return _v(c, col) if c is not None else default

    us_b_list = us_b.to_dict("records"); ep_b_list = ep_b.to_dict("records")
    us_n_list = us_n.to_dict("records"); ep_n_list = ep_n.to_dict("records")

    n_us_existing  = _existing_count(us);   n_ep_existing  = _existing_count(ep)
    n_us_fc        = _fc_count(us);          n_ep_fc        = _fc_count(ep)

    n_us_b_existing = _existing_count(us_b); n_ep_b_existing = _existing_count(ep_b)
    n_us_b_fc       = _fc_count(us_b);       n_ep_b_fc       = _fc_count(ep_b)

    total_us = n_us_existing + n_us_fc
    total_ep = n_ep_existing + n_ep_fc

    n_us_non_existing = _existing_count(us_n); n_ep_non_existing = _existing_count(ep_n)
    n_us_non_fc       = _fc_count(us_n);       n_ep_non_fc       = _fc_count(ep_n)
    n_us_non_total    = n_us_non_existing + n_us_non_fc
    n_ep_non_total    = n_ep_non_existing + n_ep_non_fc
    n_us_b_total      = n_us_b_existing + n_us_b_fc
    n_ep_b_total      = n_ep_b_existing + n_ep_b_fc

    us_ctrl_row = us_ctrl.to_dict() if us_ctrl is not None else {}
    ep_ctrl_row = ep_ctrl.to_dict() if ep_ctrl is not None else {}

    us_ctrl_pn = _ctrl(us_ctrl, "Patent Number")
    ep_ctrl_pn = _ctrl(ep_ctrl, "Patent Number")

    yr = datetime.now().year

    def _derive_est_approval(ctrl, jurisdiction_prefix):
        raw = _clean_year(_ctrl(ctrl, "Est. Approval Year"))
        if raw != "N/A":
            return raw
        phase = _ctrl(ctrl, "Phase").strip().lower() if ctrl is not None else ""
        if phase == "phase 3":
            return str(yr + 3)
        elif phase == "phase 2":
            return str(yr + 5)
        return "N/A"

    us_est_approval = _derive_est_approval(us_ctrl, "US")
    ep_est_approval = _derive_est_approval(ep_ctrl, "EP")

    # ── Rationale is generated after all metrics are computed (see return below) ──
    # Placeholder; will be filled by _generate_rationale once the dict is assembled.
    rationale = ""   # overwritten immediately after return dict is built

    result = {
        "drug_name": _v(sample, "Drug Name"),
        "rationale": rationale,                       # ← NEW
        "total_us": total_us, "total_ep": total_ep,
        "n_us_blocking": n_us_b_total, "n_ep_blocking": n_ep_b_total,
        "n_us_non": n_us_non_total, "n_ep_non": n_ep_non_total,
        "n_us_existing": n_us_existing, "n_ep_existing": n_ep_existing,
        "n_us_forecasted": n_us_fc, "n_ep_forecasted": n_ep_fc,
        "n_us_b_existing": n_us_b_existing, "n_ep_b_existing": n_ep_b_existing,
        "n_us_b_forecasted": n_us_b_fc, "n_ep_b_forecasted": n_ep_b_fc,
        "us_yte": _fmt(us_yte), "ep_yte": _fmt(ep_yte),
        "avg_yte_us_ep": _fmt(avg), "loe_score": _fmt(_score(avg)),
        "us_ctrl_pn":            us_ctrl_pn,
        "us_ctrl_pn_display":    _parse_forecasted_pn(us_ctrl_pn, "US"),
        "us_ctrl_cat":           _ctrl(us_ctrl, "Step 1 Claim Category"),
        "us_ctrl_filed":         _clean_year(_ctrl(us_ctrl, "Filing Date")),
        "us_ctrl_grant":         _ctrl(us_ctrl, "Grant Date"),
        "us_ctrl_pte":           _ctrl(us_ctrl, "PTE (months)"),
        "us_ctrl_ped":           _ctrl(us_ctrl, "Pediatric Exclusivity", "No"),
        "us_ctrl_expiry":        _clean_year(_ctrl(us_ctrl, "Controlling Patent Expiry Year")),
        "us_excl_yr":            _clean_year(_ctrl(us_ctrl, "Exclusivity Year")),
        "us_ctrl_yte":           _fmt(_num(_ctrl(us_ctrl, "Years to Entry"))),
        "us_approval":           _ctrl(us_ctrl, "Approval Date"),
        "us_est_approval":       us_est_approval,
        "us_phase":              _ctrl(us_ctrl, "Phase"),
        "us_ctrl_is_forecasted": _is_forecasted(us_ctrl_row) if us_ctrl is not None else False,
        "ep_ctrl_pn":            ep_ctrl_pn,
        "ep_ctrl_pn_display":    _parse_forecasted_pn(ep_ctrl_pn, "EP"),
        "ep_ctrl_cat":           _ctrl(ep_ctrl, "Step 1 Claim Category"),
        "ep_ctrl_filed":         _clean_year(_ctrl(ep_ctrl, "Filing Date")),
        "ep_ctrl_grant":         _ctrl(ep_ctrl, "Grant Date"),
        "ep_ctrl_pte":           _ctrl(ep_ctrl, "PTE (months)"),
        "ep_ctrl_expiry":        _clean_year(_ctrl(ep_ctrl, "Controlling Patent Expiry Year")),
        "ep_excl_yr":            _clean_year(_ctrl(ep_ctrl, "Exclusivity Year")),
        "ep_ctrl_yte":           _fmt(_num(_ctrl(ep_ctrl, "Years to Entry"))),
        "ep_approval":           _ctrl(ep_ctrl, "Approval Date"),
        "ep_est_approval":       ep_est_approval,
        "ep_phase":              _ctrl(ep_ctrl, "Phase"),
        "ep_ctrl_is_forecasted": _is_forecasted(ep_ctrl_row) if ep_ctrl is not None else False,
        "us_blocking_list": us_b_list, "ep_blocking_list": ep_b_list,
        "us_non_list": us_n_list, "ep_non_list": ep_n_list,
    }
    result['rationale'] = _generate_rationale(result)
    return result
# ─── Gemini calls ───

def _build_data_block(d: Dict) -> str:
    bp_lines = []
    for jur, patents in [("US", d["us_blocking_list"]), ("EP", d["ep_blocking_list"])]:
        ctrl_pn = d["us_ctrl_pn"] if jur == "US" else d["ep_ctrl_pn"]
        for p in patents:
            pn         = _v(p, "Patent Number")
            fd         = _v(p, "Filing Date")
            is_fc      = _is_forecasted(p)
            pn_display = _parse_forecasted_pn(pn, jur) if is_fc else pn
            fc_tag     = " [FORECASTED]" if is_fc else ""
            ctrl_flag  = " [CONTROLLING]" if pn == ctrl_pn else ""
            bp_lines.append(
                f"{pn_display} ({jur}){ctrl_flag}{fc_tag} | {_v(p, 'Step 1 Claim Category')} | "
                f"Filed:{_clean_year(fd)} | PTE:{_v(p, 'PTE (months)')}m | "
                f"Expiry:{_clean_year(_v(p, 'Controlling Patent Expiry Year'))} | "
                f"YTE:{_fmt(_num(_v(p, 'Years to Entry')))}"
            )
    nb = {}
    for lst, jur in [(d["us_non_list"], "US"), (d["ep_non_list"], "EP")]:
        cats = {}
        for p in lst:
            c = _v(p, "Step 1 Claim Category", "Other")
            cats[c] = cats.get(c, 0) + 1
        nb[jur] = cats

    us_fc_label = " (FORECASTED)" if d["us_ctrl_is_forecasted"] else ""
    ep_fc_label = " (FORECASTED)" if d["ep_ctrl_is_forecasted"] else ""
    nb_us = nb.get("US", {}); nb_ep = nb.get("EP", {})
    bp_text = "\n".join(bp_lines) if bp_lines else "None"
    yr = datetime.now().year

    us_ctrl_line = (
        f"US CONTROLLING: {d['us_ctrl_pn_display']}{us_fc_label} | Cat:{d['us_ctrl_cat']} | "
        f"Filed:{d['us_ctrl_filed']} | PTE:{d['us_ctrl_pte']}m | Ped:{d['us_ctrl_ped']} | "
        f"Expiry:{d['us_ctrl_expiry']} | ExclYr:{d['us_excl_yr']}"
    )
    if d['us_approval'] != "N/A":
        us_ctrl_line += f" | Approval:{d['us_approval']}"
    if d['us_est_approval'] != "N/A":
        us_ctrl_line += f" | EstApproval:{d['us_est_approval']}"
    if d['us_phase'] != "N/A":
        us_ctrl_line += f" | Phase:{d['us_phase']}"
    us_ctrl_line += f" | YTE:{d['us_ctrl_yte']}"

    ep_ctrl_line = (
        f"EP CONTROLLING: {d['ep_ctrl_pn_display']}{ep_fc_label} | Cat:{d['ep_ctrl_cat']} | "
        f"Filed:{d['ep_ctrl_filed']} | PTE:{d['ep_ctrl_pte']}m | "
        f"Expiry:{d['ep_ctrl_expiry']} | ExclYr:{d['ep_excl_yr']}"
    )
    if d['ep_approval'] != "N/A":
        ep_ctrl_line += f" | Approval:{d['ep_approval']}"
    if d['ep_est_approval'] != "N/A":
        ep_ctrl_line += f" | EstApproval:{d['ep_est_approval']}"
    if d['ep_phase'] != "N/A":
        ep_ctrl_line += f" | Phase:{d['ep_phase']}"
    ep_ctrl_line += f" | YTE:{d['ep_ctrl_yte']}"

    excl_formula = (
        f"ExclYr(US)=ApprovalYr+5(+0.5 if ped). ExclYr(EP)=ApprovalYr+10. "
        f"If approval/estimated approval is unavailable: "
        f"Phase 3 -> EstApproval = {yr}+3 = {yr+3}; Phase 2 -> EstApproval = {yr}+5 = {yr+5}."
    )

    rationale_block = (
        f"RATIONALE FROM INPUT DATA:\n{d['rationale']}\n\n"
        if d['rationale'] != "N/A" else ""
    )

    return (
        f"DRUG: {d['drug_name']}\n"
        f"SCORES: LOE Score={d['loe_score']}, Avg YTE(US&EP)={d['avg_yte_us_ep']}, "
        f"US YTE={d['us_yte']}, EP YTE={d['ep_yte']}\n\n"
        f"{us_ctrl_line}\n"
        f"{ep_ctrl_line}\n\n"
        f"ALL BLOCKING PATENTS:\n{bp_text}\n\n"
        f"PORTFOLIO: US {d['total_us']} ({d['n_us_existing']} existing + {d['n_us_forecasted']} forecasted, "
        f"{d['n_us_blocking']}B/{d['n_us_non']}NB) | EP {d['total_ep']} ({d['n_ep_existing']} existing + "
        f"{d['n_ep_forecasted']} forecasted, {d['n_ep_blocking']}B/{d['n_ep_non']}NB)\n"
        f"NON-BLOCKING: US {nb_us}, EP {nb_ep}\n\n"
        f"{rationale_block}"
        f"FORECASTED PATENTS NOTE: Patents marked [FORECASTED] are not yet filed. "
        f"They are identified by names like 'Forecasted patent for Device in US'. "
        f"Never use encoded strings like 'US+Forecasted+Device' in the report text.\n\n"
        f"FORMULAS: CtrlPatent=latest-filed BLOCKING patent per jurisdiction. "
        f"Expiry=FilingYr+20+(PTE/12). {excl_formula} "
        f"YTE=max(Expiry,ExclYr)-{yr}. "
        f"AvgYTE=mean(US_YTE,EP_YTE). LOE Score: <=6->5, 7-8->4, 9-11->3, 12-13->2, >13->1."
    )


def _gemini_call(prompt: str) -> str:
    """Single Gemini call with retry on rate limit."""
    gc = _get_gemini()
    for attempt in range(5):
        try:
            resp = gc.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=250000),
            )
            return (resp.text or "").strip()
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 15 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                return f"[Generation failed: {e}]"
    return "[Generation failed after retries]"


def generate_page1_text(d: Dict) -> str:
    data = _build_data_block(d)
    yr   = datetime.now().year

    rationale_instruction = ""
    if d["rationale"] != "N/A":
        rationale_instruction = (
            "\n- The RATIONALE FROM INPUT DATA block contains analyst notes from the "
            "source table. Incorporate key points from this rationale into the "
            "AVERAGE YEARS TO ENTRY AND LOE SCORE section where relevant, but keep "
            "the section within the 60-80 word limit."
        )

    prompt = f"""You are a senior pharmaceutical patent analyst writing a concise LOE (Primary Market) Scoring report for {d['drug_name']}.

Write clear, analytical paragraphs that fit within one PDF page. Be concise and precise. Write in plain text only. No markdown, no bullets, no asterisks. Use ALL CAPS for section headers only. Each section must be 60-80 words.

STRICT RULES:
- Forecasted patents have names like "Forecasted patent for Device in US" or "Forecasted patent for Compound in EP". Always use this human-readable format. NEVER write encoded strings like "US+Forecasted+Device".
- If a controlling patent is forecasted, clearly state it is a forecasted patent (not yet filed, predicted to be filed in that year).
- Refer to the scoring framework as "LOE (Primary Market) Scoring".
- Show actual arithmetic with real numbers in EVERY section (e.g., "2029 + 5 = 2034").
- Use clean year numbers (2026 not 2026.0).
- Every sentence must reference a specific number, date, or patent identifier.
- Do not use "N/Am" — if a value is unavailable, state it briefly.{rationale_instruction}

DATA:
{data}

Write exactly these four sections. Each must be a single focused paragraph of 60-80 words.

REGULATORY EXCLUSIVITY
State the approval or estimated approval year for each jurisdiction ONLY if available in the data. For US: add 5 years (plus 0.5 if pediatric exclusivity applies) and show the arithmetic. For EP: add 10 years and show the arithmetic. State the resulting exclusivity year for each.
If the approval date or estimated approval year is unavailable for a jurisdiction, do NOT mention approval date at all for that jurisdiction. Instead, if an estimated approval year was derived from the Phase (Phase 3 = current year + 3, Phase 2 = current year + 5), use that and show the arithmetic. If neither approval nor phase-based estimate is available, simply omit the exclusivity calculation for that jurisdiction.

CONTROLLING PATENT TERM
Identify the controlling patent for each jurisdiction (latest filing date among blocking patents). State the patent identifier, filing date, any PTE applied, and show: Filing Year + 20 + PTE/12 = Expiry Year.

YEARS TO ENTRY
For each jurisdiction, identify whether the expiry year or exclusivity year is the binding constraint and show: max(Expiry, Exclusivity) - {yr} = Years to Entry.

AVERAGE YEARS TO ENTRY AND LOE SCORE
Present: ({d['us_yte']} + {d['ep_yte']}) / 2 = {d['avg_yte_us_ep']} years. State the LOE (Primary Market) Score of {d['loe_score']} out of 5, name the score band, and briefly note the commercial implication. If rationale data is present, incorporate one key insight from it.

Start directly with REGULATORY EXCLUSIVITY. No introduction or conclusion outside these four sections."""
    return _gemini_call(prompt)


def generate_page2_text(d: Dict) -> str:
    data = _build_data_block(d)
    prompt = f"""You are a senior pharmaceutical patent analyst writing the Blocking Patent Analysis section of a concise LOE report for {d['drug_name']}.

Write clear, analytical paragraphs that fit within one PDF page alongside a patent table. Be concise and precise. Write in plain text only. No markdown, no bullets, no asterisks. Use ALL CAPS for section headers only. Each section must be 60-80 words.

STRICT RULES:
- Forecasted patents have names like "Forecasted patent for Device in US" or "Forecasted patent for Compound in EP". Always use this human-readable format. NEVER write encoded strings like "US+Forecasted+Device".
- Distinguish clearly between existing and forecasted patents.
- Reference specific patent identifiers in every paragraph.
- Use clean year numbers (2026 not 2026.0).
- If PTE is not applicable or not granted, state that briefly rather than writing "N/Am".

DATA:
{data}

Write exactly these four sections, each a single focused paragraph of 60-80 words.

BLOCKING PATENT LANDSCAPE
State the total blocking patents per jurisdiction, breaking down existing vs forecasted. Summarise the claim categories covered and the overall portfolio strategy.

CONTROLLING PATENT SELECTION
For each jurisdiction, name the controlling patent (or its human-readable forecasted label), state why it was selected (latest filing date), and compare briefly to the next most recent blocking patent.

KEY BLOCKING PATENTS
Analyse the most significant blocking patents beyond the controlling patent. Name each by identifier, claim category, filing date, and expiry.

NON-BLOCKING PATENTS
Summarise the non-blocking portfolio: count per jurisdiction, claim categories, and why they are classified as non-blocking.

Start directly with BLOCKING PATENT LANDSCAPE. No introduction or conclusion outside these four sections."""
    return _gemini_call(prompt)


# ─── PDF builder ───

def _trim_to_height(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated  = text[:max_chars]
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    return truncated[:last_period + 1] if last_period > 0 else truncated


def _render_ai(text, story, s_h2, s_body):
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        is_hdr = (re.match(r'^[0-9]+\.\s+[A-Z\s&]+$', line)
                  or (line.isupper() and 4 < len(line) < 80 and not line.startswith("[")))
        if is_hdr:
            story.append(Spacer(1, 0.5 * mm))
            story.append(Paragraph(line, s_h2))
        else:
            story.append(Paragraph(line, s_body))


def build_report(d: Dict, page1_text: str, page2_text: str, output_path: str) -> str:
    page1_text = _trim_to_height(page1_text, 2800)
    page2_text = _trim_to_height(page2_text, 2500)

    st  = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=14*mm, bottomMargin=10*mm, leftMargin=17*mm, rightMargin=17*mm,
        title=f"{d['drug_name'].title()} - LOE Calculation (Primary Market) Report",
        author="ADK Pipeline",
    )
    W     = A4[0] - 34*mm
    story = []

    # ── PAGE 1 ──
    story.append(Paragraph(f"{d['drug_name'].title()} - LOE Calculation (Primary Market)", st["title"]))
    story.append(Paragraph(
        f"Analysis Date: {datetime.now().strftime('%d-%b-%Y')}&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"US Patents: {d['total_us']} ({d['n_us_existing']} existing + {d['n_us_forecasted']} forecasted, "
        f"{d['n_us_blocking']} blocking)&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"EP Patents: {d['total_ep']} ({d['n_ep_existing']} existing + {d['n_ep_forecasted']} forecasted, "
        f"{d['n_ep_blocking']} blocking)", st["meta"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_BLUE, spaceAfter=5))

    # ── Scorecard (now includes Rationale row) ──
    si     = _num(d["loe_score"])
    sc_clr = _SCORE_CLR.get(int(si), "#5F6368") if si else "#5F6368"
    sc_lbl = _SCORE_LBL.get(int(si), "N/A")     if si else "N/A"
    ss     = str(int(si)) if si else "N/A"

    # Header + metric rows (unchanged layout)
    sc_rows = [
        # Header row
        [Paragraph("<b>LOE Score</b>", st["th"]),
         Paragraph("<b>Avg Years (US &amp; EP)</b>", st["th"]),
         Paragraph("<b>US Years to Entry</b>", st["th"]),
         Paragraph("<b>EP Years to Entry</b>", st["th"])],
        # Value row
        [Paragraph(
             f'<font color="{sc_clr}" size="22"><b>{ss}</b></font>'
             f'<font size="9" color="#5F6368"> / 5</font>',
             ParagraphStyle("sc", alignment=TA_CENTER, leading=26)),
         Paragraph(f'<font size="14"><b>{d["avg_yte_us_ep"]}</b></font>',
                   ParagraphStyle("av", alignment=TA_CENTER, leading=26)),
         Paragraph(f'<font size="12"><b>{d["us_yte"]}</b></font>',
                   ParagraphStyle("uv", alignment=TA_CENTER, leading=26)),
         Paragraph(f'<font size="12"><b>{d["ep_yte"]}</b></font>',
                   ParagraphStyle("ev", alignment=TA_CENTER, leading=26))],
        # Score label row
        [Paragraph(f'<font color="{sc_clr}"><b>{sc_lbl}</b></font>',
                   ParagraphStyle("sl", alignment=TA_CENTER, fontSize=8, leading=10)),
         Paragraph("", st["td"]), Paragraph("", st["td"]), Paragraph("", st["td"])],
    ]

    # ── Rationale row (spans all 4 columns) ──
    rationale_text = d["rationale"] if d["rationale"] != "N/A" else "No rationale provided."
    sc_rows.append([
        Paragraph("<b>Rationale</b>", st["th"]),
        Paragraph(rationale_text, st["td_wrap"]),
        Paragraph("", st["td"]),
        Paragraph("", st["td"]),
    ])

    sct = Table(sc_rows, colWidths=[W * 0.25] * 4)
    sct.setStyle(TableStyle([
        # Existing styles
        ("BACKGROUND",   (0, 0), (-1, 0), _NAVY),
        ("BACKGROUND",   (0, 1), (-1, 1), _LIGHT_BLUE),
        ("BACKGROUND",   (0, 2), (-1, 2), _LGREY),
        ("BOX",          (0, 0), (-1, -1), 0.75, _BLUE),
        ("INNERGRID",    (0, 0), (-1, -1), 0.4, colors.HexColor("#C0C0C0")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING",(0, 1), (-1, 1), 5),
        # Rationale row (row index 3): navy header cell + light grey content, spans cols 1-3
        ("BACKGROUND",   (0, 3), (0, 3), _NAVY),
        ("BACKGROUND",   (1, 3), (-1, 3), _LGREY),
        ("SPAN",         (1, 3), (-1, 3)),          # merge columns 1-3 for rationale text
        ("TOPPADDING",   (0, 3), (-1, 3), 4),
        ("BOTTOMPADDING",(0, 3), (-1, 3), 4),
        ("LEFTPADDING",  (1, 3), (-1, 3), 4),
        ("RIGHTPADDING", (1, 3), (-1, 3), 4),
        ("VALIGN",       (0, 3), (-1, 3), "TOP"),
    ]))
    story.append(sct)
    story.append(Spacer(1, 3*mm))
    _render_ai(page1_text, story, st["h2"], st["body"])

    story.append(PageBreak())

    # ── PAGE 2 ──
    story.append(Paragraph("Blocking Patent Analysis", st["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_BLUE, spaceAfter=3))

    all_blocking = d["us_blocking_list"] + d["ep_blocking_list"]
    if all_blocking:
        th = st["th"]; td = st["td"]; tl = st["tdl"]
        rows = [[Paragraph("<b>Patent</b>", th), Paragraph("<b>Jur</b>", th),
                 Paragraph("<b>Category</b>", th), Paragraph("<b>Filed</b>", th),
                 Paragraph("<b>Expiry</b>", th), Paragraph("<b>Controlling</b>", th)]]
        for p in all_blocking:
            pn      = _v(p, "Patent Number")
            jur     = _v(p, "Jurisdiction")
            is_ctrl = ((pn == d["us_ctrl_pn"] and jur.upper() == "US") or
                       (pn == d["ep_ctrl_pn"] and jur.upper() == "EP"))
            is_fc   = _is_forecasted(p)
            display_pn = _parse_forecasted_pn(pn, jur) if is_fc else pn
            rows.append([
                Paragraph(f"<b>{display_pn}</b>" if is_ctrl else display_pn, tl),
                Paragraph(jur, td),
                Paragraph(_v(p, "Step 1 Claim Category"), td),
                Paragraph(_clean_year(_v(p, "Filing Date")), td),
                Paragraph(_clean_year(_v(p, "Controlling Patent Expiry Year")), td),
                Paragraph(f'<font color="{_RED.hexval()}"><b>YES</b></font>' if is_ctrl else chr(8212), td),
            ])

        bt = Table(rows, colWidths=[W*0.27, W*0.07, W*0.20, W*0.14, W*0.14, W*0.18], repeatRows=1)
        bs = [
            ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
            ("BOX",        (0, 0), (-1, -1), 0.5, _BLUE),
            ("INNERGRID",  (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D0D0")),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LGREY]),
        ]
        for i, p in enumerate(all_blocking, 1):
            pn  = _v(p, "Patent Number")
            jur = _v(p, "Jurisdiction").upper()
            is_fc   = _is_forecasted(p)
            is_ctrl = ((pn == d["us_ctrl_pn"] and jur == "US") or
                       (pn == d["ep_ctrl_pn"] and jur == "EP"))
            if is_fc:
                bs.append(("BACKGROUND", (0, i), (-1, i), _LIGHT_GREEN))
            elif is_ctrl:
                bs.append(("BACKGROUND", (0, i), (-1, i), _LIGHT_BLUE))
        bt.setStyle(TableStyle(bs))
        story.append(bt)
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph(
            '<font color="#1A7A3A">&#9632;</font> Green = Forecasted patents '
            '(predicted, not yet filed)',
            st["legend"],
        ))
        story.append(Spacer(1, 2*mm))

    _render_ai(page2_text, story, st["h2"], st["body"])

    doc.build(story)
    return str(Path(output_path).resolve())


# ─── GCS Upload ───

def _upload_to_gcs(local_path: str, drug_name: str) -> str:
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit(
            "ERROR: google-cloud-storage is required.\n"
            "Run: pip install google-cloud-storage"
        )

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_FILE_NAME}"
    gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"

    print(f"  Uploading to GCS: {gcs_uri}")
    try:
        credentials = _get_credentials()
        client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(blob_name)
        blob.upload_from_filename(local_path, content_type="application/pdf")
        print(f"  Upload successful: {gcs_uri}")
    except Exception as e:
        print(f"  [ERROR] GCS upload failed for {drug_name}: {e}")
        raise

    return gcs_uri


# ─── Main processing ───

def process_drug(drug_name: str, drug_df: pd.DataFrame, output_dir: Path):
    """Process one drug. Returns (local_pdf_path, rationale_string)."""
    print(f"\n  [{drug_name}] Extracting data...")
    d = _extract_drug_data(drug_df)

    print(f"  [{drug_name}] US: {d['n_us_blocking']}B/{d['n_us_non']}NB "
          f"({d['n_us_existing']} existing + {d['n_us_forecasted']} forecasted) | "
          f"EP: {d['n_ep_blocking']}B/{d['n_ep_non']}NB "
          f"({d['n_ep_existing']} existing + {d['n_ep_forecasted']} forecasted)")
    print(f"  [{drug_name}] US ctrl: {d['us_ctrl_pn_display']} (filed {d['us_ctrl_filed']}) "
          f"{'[FORECASTED]' if d['us_ctrl_is_forecasted'] else ''} | "
          f"EP ctrl: {d['ep_ctrl_pn_display']} (filed {d['ep_ctrl_filed']}) "
          f"{'[FORECASTED]' if d['ep_ctrl_is_forecasted'] else ''}")
    print(f"  [{drug_name}] US YTE: {d['us_yte']} | EP YTE: {d['ep_yte']} | "
          f"Avg: {d['avg_yte_us_ep']} | LOE Score: {d['loe_score']}")
    print(f"  [{drug_name}] Rationale: {d['rationale']}")

    print(f"  [{drug_name}] Generating page 1 (scoring explanation)...")
    p1 = generate_page1_text(d)

    print(f"  [{drug_name}] Generating page 2 (blocking analysis)...")
    p2 = generate_page2_text(d)

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    out  = str(output_dir / f"{safe}_loe_report.pdf")

    print(f"  [{drug_name}] Building PDF...")
    local_path = build_report(d, p1, p2, out)

    print(f"  [{drug_name}] Uploading to GCS...")
    gcs_uri = _upload_to_gcs(local_path, drug_name)

    print(f"  [{drug_name}] Saved locally : {local_path}")
    print(f"  [{drug_name}] Saved to GCS  : {gcs_uri}")

    return local_path, d["rationale"]


def _load_from_bigquery() -> pd.DataFrame:
    print(f"  Authenticating (service account file: {CREDENTIALS_PATH or 'ADC'})")
    client = _get_bq_client()

    table_ref = f"`{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`"
    query = f"""
    SELECT * EXCEPT(rn) FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY Patent_Number
            ORDER BY created_at DESC
        ) AS rn
        FROM {table_ref}
    ) WHERE rn = 1
    """
    print(f"  Running ROW_NUMBER dedup query on Master_LOE")

    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df)} rows from BigQuery.")
    return df


def _ensure_rationale_column(bq_client) -> str:
    """
    Ensure the Rationale STRING, created_at TIMESTAMP, and updated_at TIMESTAMP
    columns exist in Master_LOE.  Returns the *exact* column name used for
    Drug_Name in BQ (preserving original casing) so the UPDATE queries can
    match correctly.

    created_at records the UTC date-time at which the report was first generated
    for a given drug and is never overwritten on subsequent runs.
    updated_at is set to the current UTC timestamp on every run.
    """
    table_id = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
    table    = bq_client.get_table(table_id)

    # Discover real Drug_Name column (case-insensitive lookup, return original casing)
    drug_name_col    = None
    has_rationale    = False
    has_created_at   = False
    has_updated_at   = False
    for field in table.schema:
        if field.name.lower() in ("drug_name", "drugname", "drug name"):
            drug_name_col = field.name   # exact BQ casing
        if field.name.lower() == "rationale":
            has_rationale = True
        if field.name.lower() == "created_at":
            has_created_at = True
        if field.name.lower() == "updated_at":
            has_updated_at = True

    if drug_name_col is None:
        # Fallback: print all columns so the user can diagnose
        all_cols = [f.name for f in table.schema]
        print(f"  [BQ][WARN] Could not find Drug_Name column. "
              f"Available columns: {all_cols}")
        drug_name_col = "Drug_Name"   # best guess

    print(f"  [BQ] Drug name column in BQ: '{drug_name_col}'")

    if has_rationale:
        print("  [BQ] 'Rationale' column already exists — skipping ALTER TABLE.")
    else:
        print("  [BQ] Adding 'Rationale' column to Master_LOE ...")
        ddl = f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS Rationale STRING"
        bq_client.query(ddl).result()
        print("  [BQ] 'Rationale' column added.")

    if has_created_at:
        print("  [BQ] 'created_at' column already exists — skipping ALTER TABLE.")
    else:
        print("  [BQ] Adding 'created_at' column to Master_LOE ...")
        ddl = (f"ALTER TABLE `{table_id}` "
               f"ADD COLUMN IF NOT EXISTS created_at TIMESTAMP")
        bq_client.query(ddl).result()
        print("  [BQ] 'created_at' column added.")

    if has_updated_at:
        print("  [BQ] 'updated_at' column already exists — skipping ALTER TABLE.")
    else:
        print("  [BQ] Adding 'updated_at' column to Master_LOE ...")
        ddl = (f"ALTER TABLE `{table_id}` "
               f"ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP")
        bq_client.query(ddl).result()
        print("  [BQ] 'updated_at' column added.")

    return drug_name_col

def _write_rationale_to_bigquery(
    bq_client,
    drug_rationales: dict,   # {drug_name: rationale_string}
    drug_name_col: str = "Drug_Name",   # actual BQ column name (discovered at load time)
) -> None:
    """
    Write Rationale, created_at, and updated_at back to BigQuery using one
    UPDATE per drug.

    created_at is only set when it is currently NULL (i.e. the first time the
    report is generated for that drug).  updated_at is always set to the
    current UTC run timestamp so downstream consumers can tell how fresh each
    rationale is.

    Uses parameterised queries (QueryJobConfig with query_parameters) so drug
    names and rationale text are never string-interpolated into SQL —
    eliminating escaping bugs and SQL injection risk entirely.
    """
    if not drug_rationales:
        return

    table_id = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

    from google.cloud import bigquery as _bq

    # Single UTC timestamp shared by all drugs in this run.
    run_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    total_affected = 0
    for drug_name, rationale in drug_rationales.items():
        sql = f"""
UPDATE `{table_id}`
SET Rationale  = @rationale,
    created_at = COALESCE(created_at,
                          PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S UTC', @run_timestamp)),
    updated_at = PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S UTC', @run_timestamp)
WHERE `{drug_name_col}` = @drug_name
"""
        job_config = _bq.QueryJobConfig(
            query_parameters=[
                _bq.ScalarQueryParameter("rationale",      "STRING", rationale),
                _bq.ScalarQueryParameter("run_timestamp",  "STRING", run_timestamp),
                _bq.ScalarQueryParameter("drug_name",      "STRING", drug_name),
            ]
        )
        print(f"  [BQ] UPDATE Rationale + created_at/updated_at for: {drug_name!r} "
              f"(ts={run_timestamp})")
        try:
            job = bq_client.query(sql, job_config=job_config)
            job.result()
            affected = job.num_dml_affected_rows or 0
            total_affected += affected
            print(f"         rows affected: {affected}")
            if affected == 0:
                print(f"  [BQ][WARN] 0 rows matched for {drug_name!r} "
                      f"— check that the drug name in BQ exactly matches.")
        except Exception as e:
            print(f"  [BQ][ERROR] UPDATE failed for {drug_name!r}: {e}")
            raise

    print(f"  [BQ] Done — total rows updated: {total_affected}.")

def _get_bq_client():
    """Return an authenticated BigQuery client."""
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("ERROR: google-cloud-bigquery is required.")

    credentials = _get_credentials()
    return bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )


def main():
    # Credentials resolved automatically (service account file or ADC)

    parser = argparse.ArgumentParser(description="Generate LOE Calculation (Primary Market) Report")
    parser.add_argument("-o", "--output_dir", default=None, help="Output directory (default: reports/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LOE CALCULATION (PRIMARY MARKET) REPORT GENERATOR")
    print("=" * 60)
    print(f"  Source : BigQuery ({BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID})")
    print(f"  Output : {output_dir}/")
    print(f"  GCS    : gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{{drug_name}}/{GCS_FILE_NAME}")

    # ── Single authenticated BQ client reused for load + schema + write ──
    bq_client = _get_bq_client()

    # ── Ensure Rationale column exists in BigQuery before we do anything else ──
    drug_name_col = _ensure_rationale_column(bq_client)

    df = _load_from_bigquery()

    # Normalise column names: BQ uses underscores, code expects spaces
    df.columns = [c.strip().replace("_", " ") for c in df.columns]
    if "Drug Name" not in df.columns:
        sys.exit("ERROR: 'Drug Name' column not found in BigQuery table")

    # Normalize the Type column
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    # Normalize the forecasted patents column (flexible name matching)
    df = _normalize_forecasted_col(df)

    drugs = df["Drug Name"].dropna().unique()
    print(f"\n  Found {len(drugs)} drug(s): {list(drugs)}")

    results          = []   # [(drug_name, local_path)]
    drug_rationales  = {}   # {drug_name: rationale_string}  — written back to BQ at end

    for i, name in enumerate(drugs, 1):
        print(f"\n[{i}/{len(drugs)}] Processing: {name}")
        ddf = df[df["Drug Name"] == name].copy()
        if "Jurisdiction" in ddf.columns:
            ddf = ddf[ddf["Jurisdiction"].str.upper().isin(["US", "EP"])]
        if ddf.empty:
            print(f"  [{name}] No US/EP patents - skipping")
            continue
        path, rationale = process_drug(name, ddf, output_dir)
        results.append((name, path))
        drug_rationales[name] = rationale

    # ── Write all generated rationales back to BigQuery in one MERGE job ──
    print("\n  Writing Rationale values back to BigQuery ...")
    _write_rationale_to_bigquery(bq_client, drug_rationales, drug_name_col)

    print(f"\n{'='*60}")
    print(f"  Generated {len(results)} report(s):")
    for n, p in results:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", n)
        gcs  = f"gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{safe}/{GCS_FILE_NAME}"
        print(f"    {n}:")
        print(f"      Local : {p}")
        print(f"      GCS   : {gcs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
