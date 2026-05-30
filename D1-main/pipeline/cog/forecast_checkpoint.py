"""
forecast_checkpoint.py
──────────────────────
Per-(step, drug) completion markers, stored in GCS via `cog.gcs_cache`.

Why this exists
───────────────
The forecast stages (step3/step4/step5/step6) are expensive — each does
work per drug that takes minutes (LLM calls, BQ writes, vector-DB ingest).
When a pipeline run is interrupted partway through, we don't want the
next run to redo every drug from scratch. Each successful per-drug
subprocess writes a small JSON marker here; the next run reads it back
and skips drugs that are already done.

The marker only records *that* the step finished — the actual data
products are still written to their normal destinations (BQ tables,
GCS CSVs, vector DBs). That separation matters: if you need to redo a
drug, clearing its marker is enough; the real outputs will simply be
overwritten on the next run.

GCS layout
──────────
    {GCS_CACHE_PREFIX}/forecast_checkpoints/
        step3/__global__/done.json
        step4/<drug>/done.json
        step5/<drug>/done.json
        step6/<drug>/done.json

Each `done.json` looks like:
    {
      "step": "step4",
      "drug": "Semaglutide",
      "completed_at": "2026-05-30T12:34:56Z",
      "run_id": "<optional>"
    }

Public surface
──────────────
    step_key_from_script(path)  - normalise a script path/name to a stable key
    is_done(step, drug)         - has this (step, drug) been marked complete?
    mark_done(step, drug)       - write the marker
    clear(step, drug=None)      - remove markers (for one drug, or all)

The module degrades gracefully when the GCS bucket isn't configured:
`is_done` returns False (so nothing gets skipped) and `mark_done` logs
and returns. The pipeline still works, it just won't checkpoint.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import gcs_cache

# All checkpoint blobs live under this subfolder inside the cache prefix
# (gcs_cache prepends its own GCS_CACHE_PREFIX). One subfolder, with
# step-and-drug separation handled via gcs_cache's drug_name/filename
# arguments below.
_SUBFOLDER  = "forecast_checkpoints"
_FILENAME   = "done.json"
_GLOBAL_KEY = "__global__"  # used as drug_name for steps that have no per-drug fan-out


def step_key_from_script(script: str | os.PathLike) -> str:
    """Normalise a script identifier ('step4', 'step4.py', '/x/y/step4.py') to 'step4'.

    Using the bare filename without extension keeps marker paths stable even
    if the script moves directory.
    """
    name = Path(str(script)).name
    if name.endswith(".py"):
        name = name[:-3]
    return name


def _composite_drug(step: str, drug: Optional[str]) -> str:
    """gcs_cache scopes by drug_name; we use it to scope by (step, drug)."""
    if drug is None or not str(drug).strip():
        drug = _GLOBAL_KEY
    return f"{step}/{drug}"


def is_done(step: str, drug: Optional[str] = None) -> bool:
    """True if the (step, drug) marker exists in GCS.

    Returns False (never skip) when GCS isn't available — callers should
    treat False as 'no checkpoint info, run the step'.
    """
    step = step_key_from_script(step)
    try:
        return gcs_cache.blob_exists(_SUBFOLDER, _FILENAME, drug_name=_composite_drug(step, drug))
    except Exception as e:
        print(f"[CHECKPOINT] is_done({step}, {drug}) failed: {e} — assuming not done")
        return False


def mark_done(step: str, drug: Optional[str] = None, extra: Optional[dict] = None) -> Optional[str]:
    """Write the marker. Returns the GCS URI, or None if it couldn't be written."""
    step = step_key_from_script(step)
    payload = {
        "step": step,
        "drug": drug or _GLOBAL_KEY,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if k not in payload})
    try:
        uri = gcs_cache.write_json(
            _SUBFOLDER, _FILENAME, payload, drug_name=_composite_drug(step, drug)
        )
        return uri
    except Exception as e:
        print(f"[CHECKPOINT] mark_done({step}, {drug}) failed: {e}")
        return None


def clear(step: Optional[str] = None, drug: Optional[str] = None) -> int:
    """Remove markers.

    - clear(step, drug)  : remove the one (step, drug) marker
    - clear(step)        : remove all drug markers under this step
    - clear()            : remove all forecast checkpoint markers

    Returns the number of blobs deleted (best-effort).
    """
    deleted = 0
    try:
        if step is not None:
            step = step_key_from_script(step)
            if drug is not None:
                if gcs_cache.delete_blob(_SUBFOLDER, _FILENAME,
                                         drug_name=_composite_drug(step, drug)):
                    deleted += 1
            else:
                # delete every drug marker under this step. gcs_cache has
                # delete_prefix that accepts a drug_name; the drug_name
                # acts as a path prefix, so passing just `step` removes
                # the whole tree below it.
                deleted += gcs_cache.delete_prefix(_SUBFOLDER, drug_name=step)
        else:
            deleted += gcs_cache.delete_prefix(_SUBFOLDER)
    except Exception as e:
        print(f"[CHECKPOINT] clear(step={step}, drug={drug}) failed: {e}")
    return deleted
