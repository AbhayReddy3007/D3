"""
gcs_cache.py
────────────
Shared GCS-backed cache utility for all pipeline modules.

Replaces local filesystem caching (results_cache/, analysis_cache/,
.indexer_progress/, patent_exports/, etc.) with GCS blob operations
so that nothing is written to the ephemeral Docker container filesystem.

All cache files are stored under:
    gs://{GCS_BUCKET_NAME}/pipeline_cache/{subfolder}/...

Env vars:
    GCS_BUCKET_NAME   — required (same bucket the pipeline already uses)
    GCS_CACHE_PREFIX  — optional, default: "pipeline_cache"
"""

import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import storage

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

GCS_BUCKET_NAME  = os.getenv("GCS_BUCKET_NAME")
GCS_CACHE_PREFIX = os.getenv("GCS_CACHE_PREFIX", "pipeline_cache")

_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

_client: Optional[storage.Client] = None


def _get_client() -> storage.Client:
    """Lazily create and cache a GCS client."""
    global _client
    if _client is None:
        if _CREDENTIALS_PATH and os.path.exists(_CREDENTIALS_PATH):
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(_CREDENTIALS_PATH)
            _client = storage.Client(credentials=creds)
        else:
            _client = storage.Client()
    return _client


def _bucket():
    """Return the GCS bucket object."""
    if not GCS_BUCKET_NAME:
        raise RuntimeError("GCS_BUCKET_NAME not set — cannot use GCS cache")
    return _get_client().bucket(GCS_BUCKET_NAME)


def _blob_path(subfolder: str, *parts: str) -> str:
    """Build a GCS blob path: {prefix}/{subfolder}/{parts...}"""
    safe_parts = [re.sub(r"[^a-zA-Z0-9_.-]", "_", p) for p in parts]
    return "/".join([GCS_CACHE_PREFIX, subfolder] + safe_parts)


# ─────────────────────────────────────────────
# Core operations
# ─────────────────────────────────────────────

def write_json(subfolder: str, filename: str, data: dict, drug_name: str = None) -> str:
    """
    Write a JSON dict to GCS.

    Args:
        subfolder:  Cache category (e.g. "results_cache", "analysis_cache")
        filename:   Filename within the subfolder
        data:       Dict to serialize as JSON
        drug_name:  Optional drug subdirectory

    Returns:
        GCS URI of the written blob.
    """
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    blob = _bucket().blob(blob_name)
    content = json.dumps(data, indent=2, default=str)
    blob.upload_from_string(content, content_type="application/json")
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    return uri


def read_json(subfolder: str, filename: str, drug_name: str = None) -> Optional[dict]:
    """
    Read a JSON dict from GCS. Returns None if not found.
    """
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    blob = _bucket().blob(blob_name)
    if not blob.exists():
        return None
    try:
        content = blob.download_as_text(encoding="utf-8")
        return json.loads(content)
    except Exception:
        return None


def delete_blob(subfolder: str, filename: str, drug_name: str = None) -> bool:
    """Delete a single blob. Returns True if deleted."""
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    blob = _bucket().blob(blob_name)
    if blob.exists():
        blob.delete()
        return True
    return False


def blob_exists(subfolder: str, filename: str, drug_name: str = None) -> bool:
    """Check if a blob exists."""
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    return _bucket().blob(blob_name).exists()


def list_blobs(subfolder: str, drug_name: str = None, suffix: str = None) -> List[str]:
    """
    List blob names under a subfolder (optionally within a drug subdir).
    Returns just the filename portion (last path component).
    """
    parts = [drug_name] if drug_name else []
    prefix = _blob_path(subfolder, *parts) + "/"
    blobs = _get_client().list_blobs(GCS_BUCKET_NAME, prefix=prefix)
    names = []
    for b in blobs:
        name = b.name.split("/")[-1]
        if suffix and not name.endswith(suffix):
            continue
        if name:
            names.append(name)
    return names


def delete_prefix(subfolder: str, drug_name: str = None) -> int:
    """Delete all blobs under a prefix. Returns count deleted."""
    parts = [drug_name] if drug_name else []
    prefix = _blob_path(subfolder, *parts) + "/"
    blobs = list(_get_client().list_blobs(GCS_BUCKET_NAME, prefix=prefix))
    for b in blobs:
        b.delete()
    return len(blobs)


# ─────────────────────────────────────────────
# Binary file operations (Excel, CSV, PDF, etc.)
# ─────────────────────────────────────────────

def write_bytes(subfolder: str, filename: str, data: bytes,
                content_type: str = "application/octet-stream",
                drug_name: str = None) -> str:
    """Write raw bytes to GCS. Returns GCS URI."""
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    blob = _bucket().blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{GCS_BUCKET_NAME}/{blob_name}"


def read_bytes(subfolder: str, filename: str, drug_name: str = None) -> Optional[bytes]:
    """Read raw bytes from GCS. Returns None if not found."""
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    blob = _bucket().blob(blob_name)
    if not blob.exists():
        return None
    return blob.download_as_bytes()


def write_file(subfolder: str, filename: str, local_path: str,
               content_type: str = None, drug_name: str = None) -> str:
    """Upload a local file to GCS. Returns GCS URI."""
    parts = [drug_name, filename] if drug_name else [filename]
    blob_name = _blob_path(subfolder, *parts)
    blob = _bucket().blob(blob_name)

    # Auto-detect content type
    if content_type is None:
        ext = Path(filename).suffix.lower()
        ct_map = {
            ".json": "application/json",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv":  "text/csv",
            ".pdf":  "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        content_type = ct_map.get(ext, "application/octet-stream")

    blob.upload_from_filename(local_path, content_type=content_type)
    return f"gs://{GCS_BUCKET_NAME}/{blob_name}"


def get_bytes_io(subfolder: str, filename: str, drug_name: str = None) -> Optional[io.BytesIO]:
    """Download a blob into a BytesIO object. Returns None if not found."""
    data = read_bytes(subfolder, filename, drug_name=drug_name)
    if data is None:
        return None
    return io.BytesIO(data)
