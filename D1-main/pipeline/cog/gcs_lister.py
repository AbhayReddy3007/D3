"""
gcs_lister.py
─────────────
Lists patent PDF filenames from a GCS bucket for a given drug name.
No downloading — metadata only.
"""

import os
import re
from pathlib import Path
from typing import List

# ─────────────────────────────────────────────
# Config (read from environment)
# ─────────────────────────────────────────────

GCS_BUCKET_NAME     = os.getenv("GCS_BUCKET_NAME")
GCS_PATENTS_PREFIX  = os.getenv("GCS_PATENTS_PREFIX", "patents")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

if GCS_BUCKET_NAME:
    print(f"[GCS] Config loaded: gs://{GCS_BUCKET_NAME}/{GCS_PATENTS_PREFIX}/")
else:
    print("[GCS] WARNING: GCS_BUCKET_NAME not set — patent files cannot be loaded")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, strip, collapse spaces/hyphens/underscores for fuzzy matching."""
    return re.sub(r"[\s\-_]+", "", name.lower().strip())


def get_gcs_client():
    """
    Returns an authenticated GCS storage client.
    Uses service account file if provided, otherwise ADC (Cloud Run).
    """
    from google.cloud import storage as gcs_storage
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        from google.oauth2 import service_account as sa
        creds = sa.Credentials.from_service_account_file(CREDENTIALS_PATH)
        return gcs_storage.Client(credentials=creds)
    return gcs_storage.Client()


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

def list_drug_pdf_filenames_from_gcs(drug_name: str) -> List[dict]:
    """
    Lists PDF filenames for a drug from GCS without downloading.
    Performs fuzzy folder matching on drug name.

    Returns:
        List of {"filename": str, "blob_name": str} dicts.
    """
    if not GCS_BUCKET_NAME:
        print("[GCS] GCS_BUCKET_NAME not set — cannot list patent files")
        return []

    client    = get_gcs_client()
    prefix    = GCS_PATENTS_PREFIX.rstrip("/") + "/"
    drug_norm = normalize(drug_name)

    print(f"[GCS] Listing PDFs for '{drug_name}' under gs://{GCS_BUCKET_NAME}/{prefix}")

    all_blobs = list(client.list_blobs(GCS_BUCKET_NAME, prefix=prefix))
    print(f"[GCS] Found {len(all_blobs)} total objects under prefix")

    prefix_depth = len(prefix.split("/")) - 1
    drug_folders: dict = {}
    for blob in all_blobs:
        parts = blob.name.split("/")
        if len(parts) > prefix_depth + 1:
            folder_name = parts[prefix_depth]
            norm        = normalize(folder_name)
            if norm not in drug_folders:
                drug_folders[norm] = "/".join(parts[:prefix_depth + 1]) + "/"

    print(f"[GCS] Drug folders found: {list(drug_folders.keys())}")

    if drug_norm not in drug_folders:
        print(
            f"[GCS] No folder matching '{drug_name}' (normalised: '{drug_norm}'). "
            f"Available: {list(drug_folders.keys())}"
        )
        return []

    matched_prefix = drug_folders[drug_norm]
    print(f"[GCS] Matched folder prefix: {matched_prefix}")

    pdf_blobs = [
        b for b in all_blobs
        if b.name.startswith(matched_prefix)
        and b.name.lower().endswith(".pdf")
        and not b.name.endswith("/")
    ]

    if not pdf_blobs:
        print(f"[GCS] No PDFs found in gs://{GCS_BUCKET_NAME}/{matched_prefix}")
        return []

    result = [
        {"filename": Path(b.name).name, "blob_name": b.name}
        for b in sorted(pdf_blobs, key=lambda b: b.name)
    ]
    print(f"[GCS] Found {len(result)} PDF(s): {[r['filename'] for r in result]}")
    return result
