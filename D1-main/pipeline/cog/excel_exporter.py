"""
excel_exporter.py
──────────────────
Handles:
  - Per-drug Excel export  (export_to_excel)
  - Combined multi-drug Excel export  (export_combined_excel)

Output is written to GCS under:
    gs://{bucket}/pipeline_cache/patent_exports/
"""

import io
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import gcs_cache

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

_EXCEL_SUBFOLDER = "patent_exports"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _format_approval_date(raw: Optional[str]) -> Optional[str]:
    """Normalises approval dates to DD-MMM-YYYY. Returns None if unparseable."""
    if not raw or str(raw).lower() in ("none", "null", "n/a", ""):
        return None
    raw = str(raw).strip()
    formats = [
        "%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%B %d, %Y", "%b. %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d-%b-%Y", "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%d-%b-%Y")
        except ValueError:
            continue
    return raw


def _auto_width(ws) -> None:
    """Sets column widths based on content, capped at 60 characters."""
    for col in ws.columns:
        max_len = max(
            (len(str(cell.value)) if cell.value is not None else 0)
            for cell in col
        )
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)


# ─────────────────────────────────────────────
# Per-drug export
# ─────────────────────────────────────────────

def export_to_excel(
    drug_name:     str,
    patents:       List[Dict],
    analysis_date: str,
) -> Optional[str]:
    """
    Exports patent analysis results for a single drug to an Excel file in GCS.

    Returns:
        GCS URI of the created Excel file, or None on failure.
    """
    try:
        rows = []
        for p in patents:
            jurisdiction = (p.get("jurisdiction") or "").upper()
            approval_date = _format_approval_date(
                p.get("approval_date_us") if jurisdiction == "US"
                else p.get("approval_date_eu") if jurisdiction in ("EP", "EU")
                else p.get(f"approval_date_{jurisdiction.lower()}")
            )
            approval_source = (
                p.get("approval_date_us_source") if jurisdiction == "US"
                else p.get("approval_date_eu_source") if jurisdiction in ("EP", "EU")
                else p.get(f"approval_date_{jurisdiction.lower()}_source")
            )
            phase = p.get("phase_at_filing")

            rows.append({
                "Drug Name":                      drug_name,
                "Patent Number":                  p.get("patent_number", ""),
                "Jurisdiction":                   p.get("jurisdiction", ""),
                "Tag":                            p.get("tag", ""),
                "Blocking Category":              p.get("blocking_category") or "N/A",
                "Reason":                         p.get("reason") or "N/A",

                "Step 1 Claim Category":          p.get("claim_category") or "N/A",

                "Step 2 Matched Elements":        (
                    ", ".join(k for k, v in (p.get("step2_elements_present") or {}).items() if v)
                    or ("N/A" if p.get("tag") == "BLOCKING" else "None matched")
                ),

                "S2: Active Ingredient & Form":   (
                    str((p.get("step2_elements_present") or {}).get("active_ingredient_and_form", "N/A"))
                    if p.get("step2_elements_present") is not None else "N/A"
                ),
                "S2: Formulation Details":        (
                    str((p.get("step2_elements_present") or {}).get("formulation_details", "N/A"))
                    if p.get("step2_elements_present") is not None else "N/A"
                ),
                "S2: Route of Administration":    (
                    str((p.get("step2_elements_present") or {}).get("route_of_administration", "N/A"))
                    if p.get("step2_elements_present") is not None else "N/A"
                ),
                "S2: Device Description":         (
                    str((p.get("step2_elements_present") or {}).get("device_description", "N/A"))
                    if p.get("step2_elements_present") is not None else "N/A"
                ),
                "S2: Combination Tech/Process":   (
                    str((p.get("step2_elements_present") or {}).get("combination_tech_process", "N/A"))
                    if p.get("step2_elements_present") is not None else "N/A"
                ),

                "Step 3 Technical Barrier":       (
                    "Yes" if p.get("step3_is_technical_barrier") is True
                    else "No" if p.get("step3_is_technical_barrier") is False
                    else "N/A"
                ),
                "Step 3 Confidence":              p.get("step3_confidence") or "N/A",
                "Step 3 Evidence Type":           p.get("step3_evidence_type") or "N/A",
                "Step 3 Evidence Summary":        p.get("step3_evidence_summary") or "N/A",

                "Step 4 Blocking Indicator":      (
                    "Yes" if p.get("step4_is_blocking_indicator") is True
                    else "No" if p.get("step4_is_blocking_indicator") is False
                    else "N/A"
                ),
                "Step 4 Confidence":              p.get("step4_confidence") or "N/A",
                "Step 4 Regulatory Failure if Removed": (
                    "Yes" if p.get("step4_regulatory_failure_if_removed") is True
                    else "No" if p.get("step4_regulatory_failure_if_removed") is False
                    else "N/A"
                ),
                "Step 4 Bridging Studies Required": (
                    "Yes" if p.get("step4_bridging_studies_required") is True
                    else "No" if p.get("step4_bridging_studies_required") is False
                    else "N/A"
                ),
                "Step 4 Formulation Consistent Across Phases": (
                    "Yes" if p.get("step4_formulation_consistent_across_phases") is True
                    else "No" if p.get("step4_formulation_consistent_across_phases") is False
                    else "N/A"
                ),
                "Step 4 Reason":                  p.get("step4_reason") or "N/A",

                "Step 5 Novel & Difficult":       (
                    "Yes" if p.get("step5_is_novel_and_difficult") is True
                    else "No" if p.get("step5_is_novel_and_difficult") is False
                    else "N/A"
                ),
                "Step 5 Novelty Signal":          p.get("step5_novelty_signal") or "N/A",
                "Step 5 First-in-Class":          (
                    "Yes" if p.get("step5_first_in_class") is True
                    else "No" if p.get("step5_first_in_class") is False
                    else "N/A"
                ),
                "Step 5 Prior Failed Attempts":   (
                    "Yes" if p.get("step5_prior_failed_attempts") is True
                    else "No" if p.get("step5_prior_failed_attempts") is False
                    else "N/A"
                ),
                "Step 5 Complex Implementation":  (
                    "Yes" if p.get("step5_complex_implementation") is True
                    else "No" if p.get("step5_complex_implementation") is False
                    else "N/A"
                ),
                "Step 5 Confidence":              p.get("step5_confidence") or "N/A",
                "Step 5 Reason":                  p.get("step5_reason") or "N/A",

                "Filing Date":                    p.get("filing_date") or "Unknown",
                "Grant Date":                     p.get("grant_date") or "Not yet granted",
                "PTE (months)":                   p.get("pte") if p.get("pte") is not None else "N/A",
                "Pediatric Exclusivity":          "Yes" if p.get("pediatric_exclusivity") else "No",
                "Phase":                          phase if phase else "Info N/A",
                "Launch Date":                    "",
                "Approval Date":                  approval_date or "N/A",
                "Approval Date Source":           approval_source or "N/A",
                "Est. Approval Year":             p.get("estimated_approval_year") or "N/A",
                "Exclusivity Year":               p.get("exclusivity_year") or "N/A",
                "Controlling Patent Expiry Year": p.get("controlling_patent_expiry_year") or "N/A",
                "Years to Entry":                 p.get("years_to_entry") if p.get("years_to_entry") is not None else "N/A",
                "Avg Years to Entry":             p.get("avg_years_to_entry") if p.get("avg_years_to_entry") is not None else "N/A",
                "Score":                          p.get("score") if p.get("score") is not None else "N/A",
                "Avg Years to Entry (US & EP)":   p.get("avg_years_to_entry_us_ep") if p.get("avg_years_to_entry_us_ep") is not None else "N/A",
                "IP Dimension 1 Score":           p.get("ip_dimension_1_score") if p.get("ip_dimension_1_score") is not None else "N/A",
                "Source File":                    p.get("source_file", ""),
            })

        df          = pd.DataFrame(rows)
        safe_drug   = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        safe_date   = analysis_date.replace("-", "")
        filename    = f"{safe_drug}_{safe_date}.xlsx"

        print(f"[EXCEL] Writing to GCS: {_EXCEL_SUBFOLDER}/{filename}")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Patents")
            _auto_width(writer.sheets["Patents"])
        buf.seek(0)

        uri = gcs_cache.write_bytes(
            _EXCEL_SUBFOLDER,
            filename,
            buf.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        print(f"[EXCEL] ✓ Exported: {uri}")
        return uri

    except ImportError as e:
        print(f"[EXCEL] Missing dependency: {e}")
    except Exception as e:
        import traceback
        print(f"[EXCEL] Export failed: {e}")
        print(traceback.format_exc())

    return None


# ─────────────────────────────────────────────
# Combined multi-drug export
# ─────────────────────────────────────────────

def export_combined_excel(analysis_date: str) -> Optional[str]:
    """
    Reads all per-drug Excel files from GCS for the given analysis_date
    and combines them into a single 'combined_<date>.xlsx' file.

    Returns:
        GCS URI of the combined Excel file, or None on failure.
    """
    try:
        safe_date = analysis_date.replace("-", "")

        # List all Excel files in the GCS cache subfolder
        all_files = gcs_cache.list_blobs(_EXCEL_SUBFOLDER, suffix=".xlsx")
        excel_files = [
            f for f in all_files
            if f.endswith(f"_{safe_date}.xlsx") and not f.startswith("combined_")
        ]

        if not excel_files:
            print(f"[COMBINED EXCEL] No per-drug files found for date {analysis_date}")
            return None

        print(f"[COMBINED EXCEL] Combining {len(excel_files)} file(s)...")

        dfs = []
        for fname in sorted(excel_files):
            try:
                data = gcs_cache.read_bytes(_EXCEL_SUBFOLDER, fname)
                if data:
                    df = pd.read_excel(io.BytesIO(data), sheet_name="Patents")
                    dfs.append(df)
                    print(f"[COMBINED EXCEL] + {fname} ({len(df)} rows)")
            except Exception as e:
                print(f"[COMBINED EXCEL] Could not read {fname}: {e}")

        if not dfs:
            print("[COMBINED EXCEL] No data to combine.")
            return None

        combined = pd.concat(dfs, ignore_index=True, join="outer").fillna("N/A")
        combined_filename = f"combined_{safe_date}.xlsx"

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            combined.to_excel(writer, index=False, sheet_name="All Patents")
            _auto_width(writer.sheets["All Patents"])
        buf.seek(0)

        uri = gcs_cache.write_bytes(
            _EXCEL_SUBFOLDER,
            combined_filename,
            buf.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        print(
            f"[COMBINED EXCEL] ✓ Saved: {uri} "
            f"({len(combined)} total rows)"
        )
        return uri

    except Exception as e:
        import traceback
        print(f"[COMBINED EXCEL] Failed: {e}")
        print(traceback.format_exc())
        return None
