"""
reports.py — Unified Report Orchestrator
==========================================
Runs all 6 patent report generators and uploads every output to:
    gs://cognito-gcs/Cognito_new/reports/{drug_name}/IP/

Usage:
    python reports.py

Prerequisites (.env file):
    GEMINI_API_KEY=your-key           (or GOOGLE_API_KEY)
    CREDENTIALS_PATH=/path/to/sa.json  (service-account JSON)

Optional overrides:
    BQ_PROJECT_ID   (default: cognito-prod-394707)
    BQ_DATASET_ID   (default: cognito_prod_datamart)
    BQ_LOCATION     (default: asia-south1)
    GCS_BUCKET      (default: cognito-gcs)
"""

import os
import re
import sys
import time
import traceback
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # Not needed on Cloud Run


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
GCS_BUCKET    = os.getenv("GCS_BUCKET",      "cognito-prod")
GCS_BASE_PATH = "Cognito_new/reports"
GCS_SUBFOLDER = "IP"                          # ← new sub-folder

CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

SCRIPT_DIR = Path(__file__).resolve().parent

# Global drug filter — set by --drugs CLI arg; None means "all drugs"
DRUG_FILTER = None

def _filter_drugs(drug_list):
    """Filter a list of drug names to only those in DRUG_FILTER (if set)."""
    if DRUG_FILTER is None:
        return list(drug_list)
    return [d for d in drug_list if d.strip().lower() in DRUG_FILTER]

# Report manifest: (module_file, report_label, gcs_filename)
REPORT_MANIFEST = [
    ("1bqreport.py",      "LOE Primary Market",      "Primary Market Entry Horizon.pdf"),
    ("2bqreport.py",      "Patent Strength",          "Patent Strength and Invalidity Opportunity.pdf"),
    ("3bqreport.py",      "Patent Thicket",           "Patent Thicket and Circumvention Feasibility.pdf"),
    ("4bqreport.py",      "Secondary Market LOE",     "Global Launch Sequencing & Arbitrage.pdf"),
    ("PTE_analysis",      "PTE Analysis",             "PTE Analysis.pdf"),
    ("bq_block.py",       "Blocking Analysis",        "Blocking Analysis.pdf"),
    ("forecast_report.py","Forecast Report",          "Forecast_Report.pdf"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def _validate_env():
    """Exit early if required env vars are missing."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.\n"
            "Add it to your .env file:\n"
            "  GEMINI_API_KEY=your-key"
        )
    # Credentials resolved automatically (service account file or ADC)


# ══════════════════════════════════════════════════════════════════════════════
#  GCS UPLOAD — unified uploader writing to …/{drug_name}/IP/
# ══════════════════════════════════════════════════════════════════════════════
def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None  # Use ADC (Application Default Credentials)

def _get_gcs_client():
    from google.cloud import storage
    credentials = _get_credentials()
    return storage.Client(project=os.getenv("BQ_PROJECT_ID", "cognito-prod-394707"), credentials=credentials)


def _docx_to_pdf(docx_path: str, pdf_path: str) -> str:
    """Convert DOCX to PDF using LibreOffice headless."""
    import subprocess
    output_dir = str(Path(pdf_path).parent)
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", output_dir, docx_path],
            check=True, capture_output=True, timeout=120,
        )
        # LibreOffice names output after the input stem
        lo_output = str(Path(output_dir) / (Path(docx_path).stem + ".pdf"))
        if lo_output != pdf_path and Path(lo_output).exists():
            Path(lo_output).rename(pdf_path)
        print(f"    Converted to PDF: {pdf_path}")
        return pdf_path
    except FileNotFoundError:
        print("    [WARN] LibreOffice not found — uploading DOCX as-is")
        # Fallback: just copy docx as the "pdf" path so upload works
        import shutil
        shutil.copy2(docx_path, pdf_path)
        return pdf_path
    except Exception as e:
        print(f"    [WARN] PDF conversion failed ({e}) — uploading DOCX as-is")
        import shutil
        shutil.copy2(docx_path, pdf_path)
        return pdf_path


def _save_archive_copy(bucket, local_path: str, safe_name: str,
                       gcs_filename: str, content_type: str = "application/pdf") -> None:
    """Save a timestamped copy of the file in the archive/ subfolder.

    Called AFTER the main upload so both the live file and the archive
    copy are written from the same local source in one operation.
    """
    try:
        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_name = (
            f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}"
            f"/archive/{ts_str}_{gcs_filename}"
        )
        blob = bucket.blob(archive_name)
        blob.upload_from_filename(local_path, content_type=content_type)
        print(f"    📦 archive copy → gs://{GCS_BUCKET}/{archive_name}")
    except Exception as arch_exc:
        print(f"    [WARN] archive copy failed for {gcs_filename}: {arch_exc}")


def _delete_old_blob(bucket, safe_name: str, old_filename: str) -> None:
    """Delete an old file from GCS (e.g. remove .docx after switching to .pdf)."""
    try:
        old_blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}/{old_filename}"
        old_blob = bucket.blob(old_blob_name)
        if old_blob.exists():
            old_blob.delete()
            print(f"    🗑️  deleted old file → gs://{GCS_BUCKET}/{old_blob_name}")
    except Exception as e:
        print(f"    [WARN] could not delete old {old_filename}: {e}")


def upload_to_gcs(local_path: str, drug_name: str, gcs_filename: str) -> str:
    """
    Upload a local file to:
        gs://cognito-prod/Cognito_new/reports/{drug_name}/IP/{gcs_filename}
    Also saves a timestamped archive copy alongside.
    Returns the GCS URI.
    """
    from google.cloud import storage  # noqa: F811

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
    blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}/{gcs_filename}"
    gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_name)

    ct = "application/pdf"
    if gcs_filename.endswith(".docx"):
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # ── Upload the new version ──
    blob.upload_from_filename(local_path, content_type=ct)

    # ── Save timestamped archive copy ──
    _save_archive_copy(bucket, local_path, safe_name, gcs_filename, ct)

    return gcs_uri


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE PATCHING — override each module's upload function so everything
#  goes to …/{drug_name}/IP/ instead of …/{drug_name}/
# ══════════════════════════════════════════════════════════════════════════════
def _load_module(filename: str):
    """Import a module from SCRIPT_DIR by filename (handles missing .py extension)."""
    filepath = SCRIPT_DIR / filename
    if not filepath.exists():
        # Try with .py extension appended (e.g. PTE_analysis -> PTE_analysis.py)
        filepath_py = SCRIPT_DIR / (filename + ".py")
        if filepath_py.exists():
            filepath = filepath_py
        else:
            raise FileNotFoundError(
                f"Report script not found: tried '{filepath}' and '{filepath_py}'"
            )

    module_name = filepath.stem.replace("-", "_")  # safe module name
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_module_env(mod, filename: str):
    """
    Patch hardcoded credentials and GCS paths so every module:
      - uses the CREDENTIALS_PATH env var (not hardcoded Windows paths)
      - uploads to gs://cognito-prod/Cognito_new/reports/{drug_name}/IP/
    """
    creds = CREDENTIALS_PATH

    # Fix credential paths in submodules
    if hasattr(mod, "CREDENTIALS_PATH"):
        mod.CREDENTIALS_PATH = creds

    # Legacy variable names (in case any submodule still uses them)
    if hasattr(mod, "BQ_SERVICE_KEY"):
        mod.BQ_SERVICE_KEY = creds
    if hasattr(mod, "CREDENTIALS_PATH"):
        mod.CREDENTIALS_PATH = creds
    if hasattr(mod, "SERVICE_KEY_PATH"):
        mod.SERVICE_KEY_PATH = creds

    # ── Override GCS bucket & base path so internal uploads also go to the
    #    correct bucket (cognito-prod) and path (Cognito_new/reports) ──
    if hasattr(mod, "GCS_BUCKET"):
        mod.GCS_BUCKET = GCS_BUCKET

    if hasattr(mod, "GCS_BASE_PATH"):
        mod.GCS_BASE_PATH = f"{GCS_BASE_PATH}"

    # Patch the GCS_FILE_NAME for modules that have one (1bqreport, 2bqreport)
    # so the internal upload also uses the new filename
    GCS_FILENAME_MAP = {
        "1bqreport.py":  "Primary Market Entry Horizon.pdf",
        "2bqreport.py":  "Patent Strength and Invalidity Opportunity.pdf",
    }
    if filename in GCS_FILENAME_MAP and hasattr(mod, "GCS_FILE_NAME"):
        mod.GCS_FILE_NAME = GCS_FILENAME_MAP[filename]


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL REPORT RUNNERS
#  Each returns a list of (drug_name, local_file_path) tuples.
# ══════════════════════════════════════════════════════════════════════════════

def _run_1bqreport(mod) -> list:
    """LOE Calculation (Primary Market) — one PDF per drug."""
    import pandas as pd

    # Patch the module's internal _upload_to_gcs to write to …/{drug}/IP/
    _original_upload = mod._upload_to_gcs
    def _patched_upload(local_path, drug_name):
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}/{mod.GCS_FILE_NAME}"
        gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"
        print(f"  Uploading to GCS: {gcs_uri}")
        try:
            from google.cloud import storage
            client = storage.Client(project=os.getenv("BQ_PROJECT_ID", "cognito-prod-394707"), credentials=_get_credentials())
            bucket = client.bucket(GCS_BUCKET)
            blob   = bucket.blob(blob_name)
            blob.upload_from_filename(local_path, content_type="application/pdf")
            _save_archive_copy(bucket, local_path, safe_name, mod.GCS_FILE_NAME)
            print(f"  Upload successful: {gcs_uri}")
        except Exception as e:
            print(f"  [ERROR] GCS upload failed for {drug_name}: {e}")
            raise
        return gcs_uri
    mod._upload_to_gcs = _patched_upload

    # Credentials resolved automatically (service account file or ADC)
    bq_client = mod._get_bq_client()
    drug_name_col = mod._ensure_rationale_column(bq_client)

    df = mod._load_from_bigquery()
    df.columns = [c.strip().replace("_", " ") for c in df.columns]
    if "Drug Name" not in df.columns:
        print("    [SKIP] 'Drug Name' column not found")
        return []

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()
    df = mod._normalize_forecasted_col(df)

    drugs   = _filter_drugs(df["Drug Name"].dropna().unique())
    results = []
    drug_rationales = {}

    output_dir = SCRIPT_DIR / "reports" / "1_loe_primary"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in drugs:
        ddf = df[df["Drug Name"] == name].copy()
        if "Jurisdiction" in ddf.columns:
            ddf = ddf[ddf["Jurisdiction"].str.upper().isin(["US", "EP"])]
        if ddf.empty:
            continue
        local_path, rationale = mod.process_drug(name, ddf, output_dir)
        results.append((name, local_path))
        drug_rationales[name] = rationale

    mod._write_rationale_to_bigquery(bq_client, drug_rationales, drug_name_col)
    return results


def _run_2bqreport(mod) -> list:
    """Patent Strength Scoring — one DOCX per drug."""
    import pandas as pd
    import json as _json

    # Patch the module's upload_to_gcs for per-drug upload
    def _patched_upload(local_path, drug_names):
        from google.cloud import storage
        client = storage.Client(project=os.getenv("BQ_PROJECT_ID", "cognito-prod-394707"), credentials=_get_credentials())
        bucket = client.bucket(GCS_BUCKET)
        gcs_uris = []
        for drug_name in (drug_names if isinstance(drug_names, list) else [drug_names]):
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(drug_name))
            pdf_filename = mod.GCS_FILE_NAME.replace(".docx", ".pdf")
            blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_SUBFOLDER}/{pdf_filename}"
            gcs_uri   = f"gs://{GCS_BUCKET}/{blob_name}"
            print(f"  Uploading to GCS: {gcs_uri}")
            try:
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path, content_type="application/pdf")
                _save_archive_copy(bucket, local_path, safe_name, pdf_filename)
                # Delete old .docx if it exists
                _delete_old_blob(bucket, safe_name, mod.GCS_FILE_NAME)
                print(f"  Upload successful: {gcs_uri}")
                gcs_uris.append(gcs_uri)
            except Exception as e:
                print(f"  [ERROR] GCS upload failed for drug '{drug_name}': {e}")
                raise
        return gcs_uris
    mod.upload_to_gcs = _patched_upload

    # Load all data
    data = mod.load_from_bigquery()
    df_final = data.get("final", pd.DataFrame())
    if df_final.empty:
        print("    [SKIP] patent_strength_table is empty")
        return []

    # Replace all pandas NA/NaT/NaN with "N/A" so json.dumps works.
    df_final = df_final.astype(object).fillna("N/A")
    df_country = data.get("country_scores", pd.DataFrame())
    if not df_country.empty:
        df_country = df_country.astype(object).fillna("N/A")

    output_dir = SCRIPT_DIR / "reports" / "2_patent_strength"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate one report per drug
    drug_names = _filter_drugs(df_final["Drug Name"].dropna().unique().tolist())
    results = []

    for drug in drug_names:
        print(f"    [{drug}] Building Patent Strength report...")
        drug_final = df_final[df_final["Drug Name"] == drug].copy()

        drug_country = pd.DataFrame()
        if not df_country.empty and "Drug Name" in df_country.columns:
            drug_country = df_country[df_country["Drug Name"] == drug].copy()
        elif not df_country.empty and "drug_name" in df_country.columns:
            drug_country = df_country[df_country["drug_name"] == drug].copy()

        drug_data = {
            "final": drug_final,
            "country_scores": drug_country,
        }
        # Preserve any other keys from the original data dict
        for k, v in data.items():
            if k not in ("final", "country_scores"):
                drug_data[k] = v

        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", drug)
        pdf_path  = str(output_dir / f"{safe}_Patent_Strength.pdf")

        try:
            mod.build_report(drug_data, pdf_path)
            results.append((drug, pdf_path))
        except Exception as exc:
            print(f"    [ERROR] Patent Strength report failed for '{drug}': {exc}")

    return results


def _run_3bqreport(mod) -> list:
    """Patent Thicket — one PDF per drug."""
    import tempfile

    api_key = os.getenv("GEMINI_API_KEY", "")
    drugs_data = mod.read_bq_data()
    if not drugs_data:
        print("    [SKIP] No drug data in BQ thicket tables")
        return []

    output_dir = SCRIPT_DIR / "reports" / "3_patent_thicket"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for drug_name, drug_data in sorted(drugs_data.items()):
        if DRUG_FILTER is not None and drug_name.strip().lower() not in DRUG_FILTER:
            continue
        print(f"    [{drug_name}] Generating narrative ...")
        if api_key and mod.GEMINI_AVAILABLE:
            narrative = mod.call_gemini_for_narrative(drug_data, api_key)
        else:
            narrative = mod._fallback_narrative(drug_data)

        try:
            mod.write_narrative_to_bigquery(drug_name, narrative)
        except Exception as exc:
            print(f"    [WARN] BQ write-back failed for '{drug_name}': {exc}")

        entry = {
            "drug_name":          drug_name,
            "score_data":         drug_data.get("score_data", {}),
            "circumvention_data": drug_data.get("circumvention_data", {}),
            "narrative":          narrative,
        }

        safe_drug = drug_name.replace("/", "_").replace(" ", "_")
        pdf_path  = str(output_dir / f"{safe_drug}.pdf")

        try:
            if mod.USE_DIRECT_HTML_RENDER:
                mod.convert_entry_to_pdf_direct(entry, pdf_path)
            else:
                docx_path = str(output_dir / f"{safe_drug}.docx")
                mod.build_document([entry], docx_path)
                pdf_path = mod.convert_docx_to_pdf(docx_path)
            results.append((drug_name, pdf_path))
        except Exception as exc:
            print(f"    [ERROR] PDF generation failed for '{drug_name}': {exc}")

    return results


def _run_4bqreport(mod) -> list:
    """Secondary Market LOE / Geographic Arbitrage — one PDF per drug."""
    import tempfile

    shortlisted, arb_df = mod.load_data_from_bigquery()
    if shortlisted.empty or "Drug Name" not in shortlisted.columns:
        print("    [SKIP] Shortlisted table is empty or missing Drug Name")
        return []

    drugs   = _filter_drugs(sorted(shortlisted["Drug Name"].dropna().unique()))
    results = []

    output_dir = SCRIPT_DIR / "reports" / "4_secondary_market"
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    for drug in drugs:
        drug_sl  = shortlisted[shortlisted["Drug Name"] == drug].copy()
        drug_arb = (
            arb_df[arb_df["Drug Name"] == drug].copy()
            if not arb_df.empty and "Drug Name" in arb_df.columns
            else pd.DataFrame()
        )

        safe = re.sub(r'[^\w\s-]', '', drug).strip().replace(' ', '_')
        drug_out = output_dir / safe
        drug_out.mkdir(parents=True, exist_ok=True)

        try:
            pdf_path = mod._build_drug_report(drug, drug_sl, drug_arb, str(drug_out))
            results.append((drug, pdf_path))
        except Exception as exc:
            print(f"    [ERROR] Report failed for '{drug}': {exc}")

    return results


def _run_pte_analysis(mod) -> list:
    """Regulatory Exclusivity & PTE Analysis — one PDF per drug."""
    import pandas as pd

    shortlisted, arb_df = mod.load_data_from_bigquery()
    if shortlisted.empty or "Drug Name" not in shortlisted.columns:
        print("    [SKIP] Shortlisted table empty or missing Drug Name")
        return []

    drugs   = _filter_drugs(sorted(shortlisted["Drug Name"].dropna().unique()))
    results = []

    output_dir = SCRIPT_DIR / "reports" / "5_pte_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    for drug in drugs:
        drug_sl  = shortlisted[shortlisted["Drug Name"] == drug].copy()
        drug_arb = (
            arb_df[arb_df["Drug Name"] == drug].copy()
            if not arb_df.empty and "Drug Name" in arb_df.columns
            else pd.DataFrame()
        )

        safe = re.sub(r'[^\w\s-]', '', drug).strip().replace(' ', '_')
        pdf_path = str(output_dir / f"{safe}_PTE_Analysis.pdf")

        try:
            mod._build_drug_report(drug, drug_sl, drug_arb, pdf_path)
            results.append((drug, pdf_path))
        except Exception as exc:
            print(f"    [ERROR] PTE report failed for '{drug}': {exc}")

    return results


def _run_forecast_report(mod) -> list:
    """Forecast Report — one PDF per drug.

    Delegates to forecast_report.py's `generate_report(drug, output_dir)`,
    which loads its own data from BigQuery (patent_forecast_scored +
    Master_LOE) and writes a `<drug>_Forecast_Report.pdf` into output_dir.
    We then rename each output to the manifest-standard `Forecast_Report.pdf`
    in a per-drug subfolder so the upload step can pick it up by drug.
    """
    output_dir = SCRIPT_DIR / "reports" / "7_forecast_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover drugs the same way forecast_report itself does, so we don't
    # depend on any other table for the list.
    drugs = mod._list_drugs() if hasattr(mod, "_list_drugs") else []
    if not drugs:
        print("    [SKIP] No drugs available from forecast_report._list_drugs()")
        return []
    drugs = _filter_drugs(drugs)

    results = []
    for drug in drugs:
        try:
            pdf_path = mod.generate_report(drug, str(output_dir))
            if pdf_path and Path(pdf_path).exists():
                results.append((drug, pdf_path))
            else:
                print(f"    [WARN] Forecast report produced no file for '{drug}'")
        except Exception as exc:
            print(f"    [ERROR] Forecast report failed for '{drug}': {exc}")
    return results


def _run_bq_block(mod) -> list:
    """Blocking Patent Analysis — one PDF per drug."""
    import tempfile

    patents, drug_name, analysis_date = mod.load_patents()
    if not patents:
        print("    [SKIP] No patent data found")
        return []

    # Group patents by drug name
    drug_groups = {}
    for p in patents:
        dn = str(p.get("Drug Name", drug_name or "Unknown"))
        drug_groups.setdefault(dn, []).append(p)

    output_dir = SCRIPT_DIR / "reports" / "6_blocking"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for dn, drug_patents in drug_groups.items():
        if DRUG_FILTER is not None and dn.strip().lower() not in DRUG_FILTER:
            continue
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", dn)
        pdf_path = str(output_dir / f"{safe}_Blocking_analysis.pdf")

        # We need to build the PDF locally instead of letting the module
        # upload directly, so we replicate the core generation logic but
        # write to a local file.
        try:
            all_p = mod._filter_patents(drug_patents)
            if not all_p:
                continue

            blocking     = [p for p in all_p if mod._g(p, "tag", "Tag") == "BLOCKING"]
            non_blocking = [p for p in all_p if mod._g(p, "tag", "Tag") == "NON-BLOCKING"]

            patent_summary = mod._build_patent_summary(all_p)
            analysis_text  = mod._call_gemini(dn, patent_summary)
            if not analysis_text:
                print(f"    [WARN] Gemini returned empty for {dn}")
                continue

            sections = mod._parse_sections(analysis_text)
            styles   = mod._build_styles()

            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
            )
            from reportlab.lib import colors
            from collections import defaultdict

            story = []
            ad = analysis_date or datetime.now().strftime("%Y-%m-%d")

            story.append(Paragraph(dn, styles["title"]))
            story.append(Paragraph("Blocking Patent Analysis", styles["subtitle"]))
            story.append(Paragraph(
                f"Analysis Date: {ad}&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Patents Analysed: {len(all_p)}&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<font color='{mod._RED.hexval()}'>Blocking: {len(blocking)}</font>&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<font color='{mod._GREEN.hexval()}'>Non-Blocking: {len(non_blocking)}</font>",
                styles["meta"],
            ))
            story.append(HRFlowable(width="100%", thickness=1.5, color=mod._MED_BLUE, spaceAfter=8))

            cat_counts = defaultdict(lambda: [0, 0])
            for p in all_p:
                cat = mod._g(p, "claim_category", "Step 1 Claim Category", default="Other")
                if mod._g(p, "tag", "Tag") == "BLOCKING":
                    cat_counts[cat][0] += 1
                else:
                    cat_counts[cat][1] += 1

            th = styles["th"]; td = styles["td"]; tl = styles["td_left"]
            t_rows = [[Paragraph("<b>Category</b>", th), Paragraph("<b>Blocking</b>", th),
                        Paragraph("<b>Non-Blocking</b>", th)]]
            for cat in sorted(cat_counts.keys()):
                b, nb = cat_counts[cat]
                t_rows.append([
                    Paragraph(cat, tl),
                    Paragraph(f'<font color="{mod._RED.hexval()}">{b}</font>' if b else "0", td),
                    Paragraph(f'<font color="{mod._GREEN.hexval()}">{nb}</font>' if nb else "0", td),
                ])
            cat_table = Table(t_rows, colWidths=[200, 70, 85])
            cat_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), mod._DARK_BLUE),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [mod._WHITE, mod._LIGHT_GREY]),
                ("BOX",           (0, 0), (-1, -1), 0.5, mod._MED_BLUE),
                ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D0D0")),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(cat_table)
            story.append(Spacer(1, 6))

            story.append(Paragraph("1. Overall Analysis", styles["heading"]))
            story.extend(mod._text_to_paragraphs(sections.get("overall", "N/A"), styles["body"]))
            story.append(Paragraph("2. Blocking Patents", styles["heading"]))
            story.extend(mod._text_to_paragraphs(sections.get("blocking", "N/A"), styles["body"]))
            story.append(Paragraph("3. Non-Blocking Patents", styles["heading"]))
            story.extend(mod._text_to_paragraphs(sections.get("non_blocking", "N/A"), styles["body"]))

            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.5, color=mod._GREY, spaceAfter=3))
            story.append(Paragraph(
                f"Report Date: {datetime.now().strftime('%d-%b-%Y')}&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Analysis Date: {ad}", styles["footer"],
            ))

            doc = SimpleDocTemplate(
                pdf_path, pagesize=A4,
                topMargin=16 * mm, bottomMargin=12 * mm,
                leftMargin=16 * mm, rightMargin=16 * mm,
                title=f"{dn} — Blocking Patent Analysis",
                author="ADK Pipeline",
            )
            doc.build(story)
            results.append((dn, pdf_path))

        except Exception as exc:
            print(f"    [ERROR] Blocking report failed for '{dn}': {exc}")
            traceback.print_exc()

    return results


# Map module filenames → runner functions
RUNNERS = {
    "1bqreport.py":       _run_1bqreport,
    "2bqreport.py":       _run_2bqreport,
    "3bqreport.py":       _run_3bqreport,
    "4bqreport.py":       _run_4bqreport,
    "PTE_analysis":       _run_pte_analysis,
    "bq_block.py":        _run_bq_block,
    "forecast_report.py": _run_forecast_report,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified report generator")
    parser.add_argument("--drugs", nargs="+", default=None,
                        help="Filter to specific drugs. If omitted, all drugs are processed.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these report scripts (e.g. --only 2bqreport.py forecast_report.py)")
    args = parser.parse_args()

    # Store drug filter globally so runners can access it
    global DRUG_FILTER
    DRUG_FILTER = set(d.strip().lower() for d in args.drugs) if args.drugs else None

    # Report filter
    only_reports = set(args.only) if args.only else None

    start = time.time()

    print("=" * 70)
    print("  COGNITO — UNIFIED REPORT GENERATOR")
    print("=" * 70)
    print(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GCS target  : gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{{drug_name}}/{GCS_SUBFOLDER}/")
    print(f"  Scripts dir  : {SCRIPT_DIR}")
    print("=" * 70)

    _validate_env()

    total_uploaded = []
    total_failed   = []

    for filename, label, gcs_filename in REPORT_MANIFEST:
        # Skip reports not in --only filter
        if only_reports and filename not in only_reports:
            continue

        print(f"\n{'─'*70}")
        print(f"  [{REPORT_MANIFEST.index((filename, label, gcs_filename)) + 1}/{len(REPORT_MANIFEST)}]  {label}")
        print(f"  Script: {filename}  →  GCS: …/IP/{gcs_filename}")
        print(f"{'─'*70}")

        runner = RUNNERS.get(filename)
        if runner is None:
            print(f"    [ERROR] No runner registered for {filename}")
            total_failed.append((filename, "No runner"))
            continue

        try:
            # Load and patch the module
            mod = _load_module(filename)
            _patch_module_env(mod, filename)

            # Run the report generator
            results = runner(mod)

            if not results:
                print(f"    [WARN] No output produced for {label}")
                continue

            # Upload each result to GCS under …/{drug_name}/IP/
            for drug_name, local_path in results:
                if not local_path or not Path(local_path).exists():
                    print(f"    [WARN] File not found: {local_path}")
                    continue
                try:
                    gcs_uri = upload_to_gcs(local_path, drug_name, gcs_filename)
                    total_uploaded.append((label, drug_name, gcs_uri))
                    print(f"    ✓ {drug_name} → {gcs_uri}")
                except Exception as exc:
                    print(f"    ✗ Upload failed for {drug_name}: {exc}")
                    total_failed.append((label, drug_name, str(exc)))

        except (Exception, SystemExit) as exc:
            print(f"    [FATAL] {label} failed: {exc}")
            traceback.print_exc()
            total_failed.append((label, str(exc)))

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)

    print(f"\n{'='*70}")
    print(f"  REPORT GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration    : {elapsed}s")
    print(f"  Uploaded    : {len(total_uploaded)} file(s)")
    print(f"  Failed      : {len(total_failed)} file(s)")

    if total_uploaded:
        print(f"\n  ── Uploaded Files ──")
        for label, drug, uri in total_uploaded:
            print(f"    {drug:30s}  {label}")
            print(f"      → {uri}")

    if total_failed:
        print(f"\n  ── Failures ──")
        for entry in total_failed:
            print(f"    {entry}")

    print(f"\n{'='*70}")
    print(f"  All reports target: gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{{drug_name}}/{GCS_SUBFOLDER}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
