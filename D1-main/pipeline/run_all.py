#!/usr/bin/env python3
"""
run_all.py — Pipeline orchestrator (Cloud Run Jobs + local)
──────────────────────────────────────────────────────────────
Runs the full LOE pipeline, parallelising drug-level work
within each stage using a process pool.

Drug discovery:
  Drugs are sourced from the `clinical_efficacy` BigQuery table
  (the same table used by phase_fetcher.py). Only drugs present
  in that table are processed.

Cloud Run Jobs support:
  When CLOUD_RUN_TASK_COUNT > 1 each container only processes its
  shard of drugs (determined by CLOUD_RUN_TASK_INDEX). This avoids
  every task duplicating the full workload.

Pipeline order (sequential between stages):
  1. Patents   — patent pipeline + BQ upload, parallelised across drugs
  2. Forecast  — steps 3→4→5→6 (sequential), each step parallelised across drugs
  3. Merge     — merge_to_master_loe.py (single step, runs once, only appends new rows)
  4. IPD       — ipd2bq + ipd3bq + ipd4bq all parallelised across drugs
  5. Reports   — litigation_analysis → litigation_report_generator → reports.py

Usage:
  python run_all.py                   # Refresh scores + all reports (default)
  python run_all.py --mode all        # full pipeline
  python run_all.py --mode patents    # only patents stage
  python run_all.py --mode forecast   # forecast → merge → IPD → reports (skips patents)
  python run_all.py --mode ipd        # only IPD stage
  python run_all.py --mode reports    # litigation analysis → litigation report → reports
  python run_all.py --mode refresh-scores  # ipd3 score refresh + all reports (default)
  python run_all.py --workers 6       # limit parallelism (default: 10)
  python run_all.py --dry-run         # print commands without executing
"""

import argparse
import functools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent
BQ_SCRIPT      = BASE_DIR / "2bq.py"
FORECAST_DIR   = BASE_DIR / "forecast-main"
MERGE_SCRIPT   = BASE_DIR / "merge_to_master_loe.py"
IPD2_SCRIPT    = BASE_DIR / "ipd2bq.py"
IPD3_SCRIPT    = BASE_DIR / "ipd3bq.py"
IPD4_SCRIPT    = BASE_DIR / "ipd4bq.py"
REPORTS_SCRIPT              = BASE_DIR / "reports.py"
LITIGATION_ANALYSIS_SCRIPT  = BASE_DIR / "litigation_analysis.py"
LITIGATION_REPORT_SCRIPT    = BASE_DIR / "litigation_report_generator.py"
BLOCKING_REPORT_SCRIPT      = BASE_DIR / "blocking_report_ai.py"

DEFAULT_WORKERS = 10

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
PY     = sys.executable


# ── Cloud Run Jobs sharding ───────────────────────────────────────────────────

def shard_drugs(drugs: list) -> list:
    """
    Return only this task's slice of drugs when running inside a
    Cloud Run Job with multiple tasks.

    Cloud Run sets:
      CLOUD_RUN_TASK_INDEX  — 0-based index of this task
      CLOUD_RUN_TASK_COUNT  — total number of tasks

    If these vars are absent (local run) the full list is returned.
    """
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))

    if task_count <= 1:
        return drugs  # local / single-task — process everything

    # Distribute drugs round-robin so the load is even when len(drugs)
    # is not evenly divisible by task_count.
    shard = [d for i, d in enumerate(drugs) if i % task_count == task_index]

    print(
        f"[SHARD] task {task_index + 1}/{task_count} → "
        f"{len(shard)} of {len(drugs)} drug(s): {shard}"
    )
    return shard


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(text):
    print(f"\n{BOLD}{'═' * 64}")
    print(f"  {text}")
    print(f"{'═' * 64}{RESET}\n")


def run_step(label, cmd, cwd=None, dry_run=False):
    """Run a single subprocess, abort on failure."""
    print(f"{YELLOW}▶ {label}{RESET}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"  cwd: {cwd}")
    if dry_run:
        print(f"  {YELLOW}[DRY RUN] skipped{RESET}\n")
        return
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"FAILED (exit {result.returncode}) after {elapsed:.1f}s — {label}"
        )
    print(f"  {GREEN}✓ Done in {elapsed:.1f}s{RESET}\n")


# ── Drug discovery from BigQuery clinical_efficacy ────────────────────────────

def discover_drugs():
    """
    Discover drugs from the clinical_efficacy BigQuery table.

    Uses the same env vars as phase_fetcher.py:
      BQ_PROJECT_ID  — GCP project  (fallback: PROJECT_ID)
      BQ_DATASET_ID  — BQ dataset
      BQ_TABLE_NAME  — table name   (fallback: 'clinical_efficacy')

    Returns a sorted list of distinct, non-empty molecule names.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from google.cloud import bigquery
    from google.oauth2 import service_account

    project_id = (
        os.getenv("BQ_PROJECT_ID")
        or os.getenv("PROJECT_ID")
        or os.getenv("BQ_UPLOAD_PROJECT")
    )
    dataset_id = os.getenv("BQ_DATASET_ID")
    table_name = os.getenv("BQ_TABLE_NAME", "clinical_efficacy")

    if not project_id or not dataset_id:
        print(f"{RED}ERROR: BQ_PROJECT_ID / PROJECT_ID and BQ_DATASET_ID must be set{RESET}")
        sys.exit(1)

    fq_table = f"{project_id}.{dataset_id}.{table_name}"
    print(f"[DISCOVERY] Querying distinct drugs from {fq_table} ...")

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        creds  = service_account.Credentials.from_service_account_file(creds_path)
        client = bigquery.Client(credentials=creds, project=project_id)
    else:
        client = bigquery.Client(project=project_id)

    query = f"""
    SELECT DISTINCT TRIM(molecule_name) AS drug
    FROM `{fq_table}`
    WHERE molecule_name IS NOT NULL
      AND TRIM(molecule_name) != ''
    ORDER BY drug
    """

    try:
        rows = client.query(query).result()
        drugs = [row.drug for row in rows]
    except Exception as e:
        print(f"{RED}ERROR: Failed to query {fq_table}: {e}{RESET}")
        sys.exit(1)

    if not drugs:
        print(f"{RED}ERROR: No drugs found in {fq_table}{RESET}")
        sys.exit(1)

    print(f"[DISCOVERY] {len(drugs)} drug(s) from {fq_table}")
    for i, d in enumerate(drugs):
        print(f"  {i + 1:>3}. {d}")

    return drugs


# ── Per-drug worker functions (run in child processes) ────────────────────────
# Each function processes ONE drug and returns (drug, ok, error_msg).
# They must be top-level functions so ProcessPoolExecutor can pickle them.

def _patent_worker(drug, dry_run):
    """Patent pipeline + BQ upload for a single drug."""
    try:
        run_step(
            f"Patent pipeline: {drug}",
            [PY, "-m", "cog.main", drug],
            cwd=BASE_DIR, dry_run=dry_run,
        )
        run_step(
            f"BQ upload: {drug}",
            [PY, str(BQ_SCRIPT), "--drug", drug],
            dry_run=dry_run,
        )
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


def _forecast_step_worker(drug, dry_run, step_script=None, step_label=None,
                          extra_args=None, drug_flag="--drug", resume=False):
    """Run a single forecast step for one drug.

    Parameter order matches run_parallel's convention: (drug, dry_run, ...).
    The remaining kwargs are bound via functools.partial in run_forecast
    (a top-level function + partial is picklable, whereas a nested closure
    is not — that's what causes
    "Can't pickle local object 'run_forecast.<locals>._step_worker'").

    drug_flag controls how the drug is passed to the script
    (e.g. "--drug" for step4/6, "--drugs" for step5).

    resume=True checks the GCS checkpoint marker for (step, drug) and skips
    the subprocess if it's already been completed in a prior run. The
    marker is only written after a successful subprocess return — failures
    don't get marked, so a re-run will retry them.
    """
    # Import lazily so worker processes don't pay the cost when checkpoint
    # support is unused, and so a missing GCS config doesn't break imports
    # at module load time.
    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception as _e:
        _ckpt = None
        if resume:
            print(f"[CHECKPOINT] unavailable ({_e}) — running {drug} without skip")

    if resume and _ckpt is not None and not dry_run:
        if _ckpt.is_done(step_script, drug):
            print(f"  [SKIP] {step_label}: {drug} — already marked done in GCS")
            return (drug, True, None)

    try:
        cmd = [PY, str(step_script), drug_flag, drug]
        if extra_args:
            cmd.extend(extra_args)
        run_step(
            f"{step_label}: {drug}",
            cmd,
            dry_run=dry_run,
        )
        if _ckpt is not None and not dry_run:
            _ckpt.mark_done(step_script, drug)
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


def _step5_company_worker(company_drugs, dry_run, step_script=None,
                          step_label=None, resume=False):
    """Run step5 for one (company, drugs) pair.

    Why this exists separately from _forecast_step_worker:
    step5 reviews a COMPANY's strategy, not a drug's. Calling it per-drug
    fans out redundantly when one company owns multiple drugs in the set,
    AND it fails outright when the orchestrator passes only `--drugs <x>`
    without `--company`, because step5's batch fallback queries an
    optional BQ_SOURCE_TABLE that often isn't populated. So we resolve
    drugs → unique companies upstream and call step5 once per company,
    passing all the company's drugs together via `--drugs A,B,C`.

    company_drugs is a tuple/list: (company_name, [drug1, drug2, ...]).
    The orchestrator's run_parallel calls worker_fn(item, dry_run), so
    company_drugs arrives as the first positional arg.

    Checkpoint key is the company name (not a drug), so re-runs skip
    already-reviewed companies cleanly.
    """
    company, drugs_for_company = company_drugs

    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    if resume and _ckpt is not None and not dry_run:
        if _ckpt.is_done(step_script, company):
            print(f"  [SKIP] {step_label}: {company} — already marked done in GCS")
            return (company, True, None)

    try:
        cmd = [PY, str(step_script), "--company", company]
        if drugs_for_company:
            cmd.extend(["--drugs", ",".join(drugs_for_company)])
        run_step(
            f"{step_label}: {company} ({len(drugs_for_company)} drug(s))",
            cmd,
            dry_run=dry_run,
        )
        if _ckpt is not None and not dry_run:
            _ckpt.mark_done(step_script, company)
        return (company, True, None)
    except Exception as e:
        return (company, False, str(e))


def _resolve_drug_to_company(drugs):
    """Map a list of drug names to their innovator companies.

    Queries the merged forecast_s3 BQ table (which step3 writes with
    --upload) for (drug_name, innovator). Returns a dict
    {company_name: [drug1, drug2, ...]}. Returns {} if the table is
    missing/empty/inaccessible — step5 will then be skipped with a
    warning rather than failing the pipeline.

    Drugs without a recognised innovator get bucketed under 'Unknown',
    which we drop (step5 with company='Unknown' produces nothing useful).
    """
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except Exception as e:
        print(f"[STEP5 PREP] BigQuery client unavailable ({e}); skipping company resolution")
        return {}

    project_id = (
        os.getenv("BQ_PROJECT_ID")
        or os.getenv("PROJECT_ID")
        or os.getenv("BQ_UPLOAD_PROJECT")
    )
    dataset_id = os.getenv("BQ_DATASET_ID") or "cognito_prod_datamart"
    table_name = os.getenv("BQ_FORECAST_S3_TABLE", "forecast_s3")
    if not project_id:
        print("[STEP5 PREP] BQ_PROJECT_ID not set — cannot resolve drugs→companies")
        return {}

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        creds  = service_account.Credentials.from_service_account_file(creds_path)
        client = bigquery.Client(credentials=creds, project=project_id)
    else:
        client = bigquery.Client(project=project_id)

    fq_table = f"{project_id}.{dataset_id}.{table_name}"
    drugs_param = [str(d) for d in drugs]

    query = f"""
      SELECT DISTINCT drug_name, innovator
      FROM `{fq_table}`
      WHERE drug_name IN UNNEST(@drugs)
        AND innovator IS NOT NULL
        AND TRIM(innovator) NOT IN ('', 'Unknown', 'unknown', 'N/A', 'nan')
    """
    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("drugs", "STRING", drugs_param),
            ]
        )
        df = client.query(query, job_config=job_config).to_dataframe()
    except Exception as e:
        print(f"[STEP5 PREP] forecast_s3 query failed ({e}); skipping company resolution")
        return {}

    if df.empty:
        print(f"[STEP5 PREP] No (drug→innovator) rows found in {fq_table} for shard")
        return {}

    mapping = {}
    for _, row in df.iterrows():
        company = str(row["innovator"]).strip()
        drug    = str(row["drug_name"]).strip()
        if not company or not drug:
            continue
        mapping.setdefault(company, []).append(drug)

    # Deduplicate the drug lists within each company
    for c in mapping:
        mapping[c] = sorted(set(mapping[c]))

    found = sum(len(v) for v in mapping.values())
    print(f"[STEP5 PREP] Resolved {found}/{len(drugs)} drug(s) → {len(mapping)} company(ies)")
    return mapping


def _run_forecast_step_global(step_label, step_script, extra_args=None,
                              dry_run=False, resume=False, drugs=None):
    """Run a forecast step that operates on a batch of drugs in one invocation.

    Used for step3.py. Historically step3 processed the entire dataset in
    one go (no sharding), which meant every Cloud Run task redid the
    full-dataset work. step3 now accepts `--drugs A,B,C` so the orchestrator
    can pass only this task's drug slice; the checkpoint key is also
    scoped to that slice so different tasks don't overwrite each other's
    done-markers.

    Args:
        drugs: Optional list of drug names for this shard. When provided,
            appended as `--drugs <comma-list>` to the subprocess command.
            When None, step3 falls back to processing all drugs.
    """
    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    # Build a stable per-shard checkpoint key so two tasks with different
    # drug slices keep separate markers. When drugs is None it's the
    # "all drugs" run and we use a single global marker.
    if drugs:
        # Sort + join so the key is deterministic regardless of input order.
        shard_key = "drugs__" + ",".join(sorted(drugs))[:180]  # bound length
    else:
        shard_key = None  # global

    if resume and _ckpt is not None and not dry_run and _ckpt.is_done(step_script, shard_key):
        scope = f"shard ({len(drugs)} drug(s))" if drugs else "global"
        print(f"  [SKIP] {step_label} [{scope}] — already marked done in GCS")
        return

    cmd = [PY, str(step_script)]
    if extra_args:
        cmd.extend(extra_args)
    if drugs:
        cmd.extend(["--drugs", ",".join(drugs)])
    run_step(step_label, cmd, dry_run=dry_run)

    if _ckpt is not None and not dry_run:
        _ckpt.mark_done(step_script, shard_key)


def _ipd_worker(drug, dry_run, resume=False):
    """ipd2bq + ipd3bq + ipd4bq for a single drug.

    Checkpointed as three separate keys ('ipd2bq', 'ipd3bq', 'ipd4bq') because the
    three scripts are independent — if ipd3bq fails after ipd2bq succeeded,
    a re-run with resume should skip ipd2bq and retry only ipd3bq.
    """
    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    # ipd2bq
    skip_2 = (
        resume and _ckpt is not None and not dry_run
        and _ckpt.is_done("ipd2bq", drug)
    )
    if skip_2:
        print(f"  [SKIP] IPD2 BQ upload: {drug} — already marked done in GCS")
    else:
        try:
            run_step(
                f"IPD2 BQ upload: {drug}",
                [PY, str(IPD2_SCRIPT), drug],
                dry_run=dry_run,
            )
            if _ckpt is not None and not dry_run:
                _ckpt.mark_done("ipd2bq", drug)
        except Exception as e:
            return (drug, False, f"ipd2bq: {e}")

    # ipd3bq
    skip_3 = (
        resume and _ckpt is not None and not dry_run
        and _ckpt.is_done("ipd3bq", drug)
    )
    if skip_3:
        print(f"  [SKIP] IPD3 BQ upload: {drug} — already marked done in GCS")
    else:
        try:
            run_step(
                f"IPD3 BQ upload: {drug}",
                [PY, str(IPD3_SCRIPT), drug],
                dry_run=dry_run,
            )
            if _ckpt is not None and not dry_run:
                _ckpt.mark_done("ipd3bq", drug)
        except Exception as e:
            return (drug, False, f"ipd3bq: {e}")

    # ipd4bq — now per-drug (accepts a positional drug argument)
    skip_4 = (
        resume and _ckpt is not None and not dry_run
        and _ckpt.is_done("ipd4bq", drug)
    )
    if skip_4:
        print(f"  [SKIP] IPD4 BQ upload: {drug} — already marked done in GCS")
    else:
        try:
            run_step(
                f"IPD4 BQ upload: {drug}",
                [PY, str(IPD4_SCRIPT), drug],
                dry_run=dry_run,
            )
            if _ckpt is not None and not dry_run:
                _ckpt.mark_done("ipd4bq", drug)
        except Exception as e:
            return (drug, False, f"ipd4bq: {e}")

    return (drug, True, None)


def run_ipd4_global(dry_run=False, resume=False):
    """DEPRECATED: ipd4bq now runs per-drug inside _ipd_worker.

    Kept for backward compatibility if called directly. When called,
    it runs ipd4bq without a drug filter (processes all drugs).
    """
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

    if task_count > 1 and task_index != 0:
        print(f"{YELLOW}[SHARD] IPD4 skipped on task {task_index} "
              f"(only task 0 runs the global ipd4bq step){RESET}\n")
        return

    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    if resume and _ckpt is not None and not dry_run and _ckpt.is_done("ipd4bq", None):
        print(f"{YELLOW}[SKIP] IPD4 BQ upload — already marked done in GCS{RESET}\n")
        return

    try:
        run_step(
            "IPD4 BQ upload (global)",
            [PY, str(IPD4_SCRIPT)],
            dry_run=dry_run,
        )
        if _ckpt is not None and not dry_run:
            _ckpt.mark_done("ipd4bq", None)
    except Exception as e:
        print(f"{RED}✗ IPD4 BQ upload failed: {e}{RESET}")
        raise


def run_reports_global(drugs, dry_run=False, resume=False):
    """Run reports.py with the sharded drug list.

    Now runs on ALL tasks — each task generates reports only for its
    shard of drugs via the --drugs filter.
    """
    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    # Filter out drugs already completed
    drugs_to_run = list(drugs)
    if resume and _ckpt is not None and not dry_run:
        drugs_to_run = [d for d in drugs_to_run if not _ckpt.is_done("reports", d)]
        skipped = len(drugs) - len(drugs_to_run)
        if skipped:
            print(f"{YELLOW}[SKIP] Reports: {skipped} drug(s) already done{RESET}")

    if not drugs_to_run:
        print(f"{YELLOW}All drugs already have reports — skipping.{RESET}\n")
        return

    try:
        run_step(
            f"Reports ({len(drugs_to_run)} drug(s))",
            [PY, str(REPORTS_SCRIPT), "--drugs"] + drugs_to_run,
            dry_run=dry_run,
        )
        if _ckpt is not None and not dry_run:
            for d in drugs_to_run:
                _ckpt.mark_done("reports", d)
    except Exception as e:
        print(f"{RED}✗ Reports failed: {e}{RESET}")
        raise


# ── Parallel runner ──────────────────────────────────────────────────────────

def run_parallel(label, worker_fn, drugs, workers, dry_run=False):
    """
    Run worker_fn(drug, dry_run) in parallel across all drugs.
    Prints progress and aborts if any drug fails.
    """
    if not drugs:
        print(f"{YELLOW}No drugs to process for this task. Skipping.{RESET}\n")
        return

    total    = len(drugs)
    failed   = []
    done     = 0

    print(f"  Parallelising {total} drug(s) across {min(workers, total)} worker(s)\n")

    with ProcessPoolExecutor(max_workers=min(workers, total)) as pool:
        futures = {
            pool.submit(worker_fn, drug, dry_run): drug
            for drug in drugs
        }
        for future in as_completed(futures):
            item = futures[future]
            # `item` may be a string (drug name) or a (name, payload) tuple
            # depending on the worker. Use the first element for display
            # when it's a tuple/list.
            display_name = item[0] if isinstance(item, (tuple, list)) else item
            try:
                drug_name, ok, err = future.result()
            except Exception as exc:
                drug_name, ok, err = display_name, False, str(exc)

            done += 1
            if ok:
                print(f"  {GREEN}[{done}/{total}] ✓ {drug_name}{RESET}")
            else:
                print(f"  {RED}[{done}/{total}] ✗ {drug_name}: {err}{RESET}")
                failed.append(drug_name)

    if failed:
        # Per-drug failures should not nuke the whole pipeline run — a flaky
        # PDF, a transient BQ glitch, or a single drug with weird data
        # shouldn't kill the work for every other drug. Only halt if
        # every drug failed (i.e. it's clearly a systemic issue).
        succeeded = total - len(failed)
        print(
            f"\n{YELLOW}⚠ {label} — {len(failed)}/{total} drug(s) failed: "
            f"{failed}{RESET}"
        )
        if succeeded == 0:
            print(
                f"{RED}  All drugs failed for this stage — pipeline halted "
                f"(likely a systemic issue, not per-drug data).{RESET}"
            )
            sys.exit(1)
        else:
            print(
                f"  {GREEN}Continuing with {succeeded}/{total} successful "
                f"drug(s).{RESET} Re-run the pipeline for failed drugs "
                f"individually to investigate."
            )
            return

    print(f"\n  {GREEN}✓ {label} — all {total} drug(s) completed{RESET}\n")


# ── Pipeline stages ──────────────────────────────────────────────────────────

def run_patents(drugs, workers, dry_run=False):
    banner("PATENT PROCESSING")
    run_parallel("Patents", _patent_worker, drugs, workers, dry_run)


def run_forecast(drugs, workers, dry_run=False, resume=False):
    banner("FORECASTING PIPELINE" + (" [RESUME]" if resume else ""))

    # Each entry: (label, script, extra_args, kind)
    #   kind = "global"   -> one invocation per shard, uses --drugs A,B,C
    #   kind = "per-drug" -> one invocation per drug, fans out in parallel
    #   kind = "per-company" -> resolve drugs→companies, fan out per COMPANY
    #
    # step3 used to be "global with no --drugs"; it now accepts --drugs so
    #   the orchestrator can shard it across Cloud Run tasks.
    # step5 was previously fanned out per-drug, which broke because step5
    #   wants --company (not --drug). It also did redundant work when one
    #   company owned multiple drugs. We now resolve drugs→companies first
    #   and fan out per company.
    forecast_steps = [
        ("Step 3 — IP Landscape + Layering + Filing Analysis",
         FORECAST_DIR / "step3.py", ["--upload"], "global"),
        ("Step 4 — Innovator Filing Patterns",
         FORECAST_DIR / "step4.py", None, "per-drug", "--drug"),
        ("Step 5 — Business Strategy Review",
         FORECAST_DIR / "step5.py", None, "per-company", None),
        ("Step 6 — Patent Forecast Generator",
         FORECAST_DIR / "step6.py", None, "per-drug", "--drug"),
    ]

    for entry in forecast_steps:
        # Tolerate both 4-tuple (global) and 5-tuple (per-drug/per-company) shapes
        if len(entry) == 4:
            step_label, step_script, extra_args, kind = entry
            drug_flag = None
        else:
            step_label, step_script, extra_args, kind, drug_flag = entry

        print(f"{BOLD}  {step_label}{RESET}")

        if kind == "global":
            # Global-style step (step3): runs once per task with a --drugs
            # shard, so each Cloud Run task only processes its own slice
            # instead of the whole dataset every time.
            try:
                _run_forecast_step_global(step_label, step_script, extra_args,
                                          dry_run, resume=resume, drugs=drugs)
                print(f"  {GREEN}✓ {step_label}{RESET}")
            except Exception as e:
                print(f"  {RED}✗ {step_label}: {e}{RESET}")
                sys.exit(1)
            continue

        if kind == "per-company":
            # Resolve drugs → unique companies (via forecast_s3 BQ table
            # written by step3 --upload above). If the table is missing
            # or no companies resolve, skip step5 with a warning rather
            # than failing the pipeline; downstream steps don't depend
            # on step5's output.
            print(f"  [STEP5 PREP] Resolving drugs → companies for {len(drugs)} drug(s)...")
            company_map = _resolve_drug_to_company(drugs)
            if not company_map:
                print(
                    f"  {YELLOW}⚠ {step_label} — no companies resolved; "
                    f"skipping. (Run step3 first, or check that forecast_s3 "
                    f"has innovator data.){RESET}\n"
                )
                continue

            # Build [(company, [drugs]), ...] in deterministic order
            items = sorted(company_map.items(), key=lambda kv: kv[0])
            print(f"  [STEP5 PREP] Will run {len(items)} company review(s)")

            step_worker = functools.partial(
                _step5_company_worker,
                step_script=step_script,
                step_label=step_label,
                resume=resume,
            )
            # run_parallel expects "drugs" but accepts any iterable of items
            run_parallel(step_label, step_worker, items, workers, dry_run)
            continue

        # Per-drug step (step4, step6): one invocation per drug
        step_worker = functools.partial(
            _forecast_step_worker,
            step_script=step_script,
            step_label=step_label,
            extra_args=extra_args,
            drug_flag=drug_flag,
            resume=resume,
        )

        run_parallel(step_label, step_worker, drugs, workers, dry_run)


def run_merge(dry_run=False, resume=False):
    """
    Merge runs once globally.  When sharded across Cloud Run tasks,
    only task 0 executes the merge; other tasks skip it.

    resume=True checks a global 'merge' checkpoint and skips if marked done.
    """
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

    if task_count > 1 and task_index != 0:
        print(f"{YELLOW}[SHARD] Merge skipped on task {task_index} "
              f"(only task 0 runs merge){RESET}\n")
        return

    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    if resume and _ckpt is not None and not dry_run and _ckpt.is_done("merge", None):
        print(f"{YELLOW}[SKIP] MERGE → Master_LOE — already marked done in GCS{RESET}\n")
        return

    banner("MERGE → Master_LOE")
    run_step("merge_to_master_loe.py", [PY, str(MERGE_SCRIPT)], dry_run=dry_run)
    if _ckpt is not None and not dry_run:
        _ckpt.mark_done("merge", None)


def run_ipd(drugs, workers, dry_run=False, resume=False):
    banner("IPD → BIGQUERY" + (" [RESUME]" if resume else ""))

    # Bind `resume` into the workers via functools.partial — same picklability
    # rule as the forecast workers (top-level function + partial = OK,
    # nested closure = not OK).
    ipd_worker  = functools.partial(_ipd_worker,  resume=resume)

    # ipd2bq + ipd3bq + ipd4bq — all parallelised across drugs
    # ipd4bq now accepts a --drug filter and processes only that drug's
    # rows from Master_LOE, so it can safely run per-drug in parallel.
    run_parallel("IPD (ipd2bq + ipd3bq + ipd4bq)", ipd_worker, drugs, workers, dry_run)


def run_litigation_analysis(drugs, dry_run=False, resume=False):
    """Run litigation_analysis.py with the sharded drug list.

    Unlike the report generators, this runs on ALL Cloud Run tasks —
    each task processes only its shard of drugs (already filtered by
    shard_drugs() in main()).
    """
    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    # Filter out drugs already completed in checkpoint
    drugs_to_run = list(drugs)
    if resume and _ckpt is not None and not dry_run:
        drugs_to_run = [d for d in drugs_to_run if not _ckpt.is_done("litigation_analysis", d)]
        skipped = len(drugs) - len(drugs_to_run)
        if skipped:
            print(f"{YELLOW}[SKIP] Litigation analysis: {skipped} drug(s) already done{RESET}")

    if not drugs_to_run:
        print(f"{YELLOW}All drugs already have litigation analysis — skipping.{RESET}\n")
        return

    try:
        run_step(
            f"Litigation Analysis ({len(drugs_to_run)} drug(s))",
            [PY, str(LITIGATION_ANALYSIS_SCRIPT), "--drugs"] + drugs_to_run,
            dry_run=dry_run,
        )
        # Mark each drug as done
        if _ckpt is not None and not dry_run:
            for d in drugs_to_run:
                _ckpt.mark_done("litigation_analysis", d)
    except Exception as e:
        print(f"{RED}✗ Litigation analysis failed: {e}{RESET}")
        raise


def _litigation_report_worker(drug, dry_run, resume=False):
    """Generate litigation report for a single drug."""
    try:
        from cog import forecast_checkpoint as _ckpt
    except Exception:
        _ckpt = None

    if resume and _ckpt is not None and not dry_run and _ckpt.is_done("litigation_report", drug):
        print(f"  [SKIP] Litigation report: {drug} — already done")
        return (drug, True, None)

    try:
        run_step(
            f"Litigation Report: {drug}",
            [PY, str(LITIGATION_REPORT_SCRIPT), "--drugs", drug, "--latest-only"],
            dry_run=dry_run,
        )
        if _ckpt is not None and not dry_run:
            _ckpt.mark_done("litigation_report", drug)
        return (drug, True, None)
    except Exception as e:
        return (drug, False, f"litigation_report: {e}")


def run_litigation_report(drugs, workers, dry_run=False, resume=False):
    """Run litigation_report_generator.py per drug, parallelised across tasks."""
    lit_report_worker = functools.partial(_litigation_report_worker, resume=resume)
    run_parallel("Litigation Reports", lit_report_worker, drugs, workers, dry_run)


def run_reports(drugs, workers, dry_run=False, resume=False):
    banner("REPORTS GENERATION" + (" [RESUME]" if resume else ""))

    # Step 1: Standard reports — all tasks, sharded per drug
    run_reports_global(drugs, dry_run=dry_run, resume=resume)

    # Step 2: Litigation analysis — runs on ALL tasks (sharded per drug)
    run_litigation_analysis(drugs, dry_run=dry_run, resume=resume)

    # Step 3: Litigation report — per drug, parallelised across tasks
    run_litigation_report(drugs, workers, dry_run=dry_run, resume=resume)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOE Pipeline orchestrator")
    parser.add_argument(
        "--mode",
        choices=["all", "patents", "forecast", "ipd", "reports", "refresh-scores", "blocking", "step6-reports", "forecast-reports"],
        default="forecast-reports",
        help=(
            "all              = full pipeline: patents → forecast → merge → ipd → reports\n"
            "patents          = patent pipeline only\n"
            "forecast         = forecast → merge → ipd → reports (skips patents)\n"
            "ipd              = IPD BQ upload only\n"
            "reports          = all reports (standard + litigation)\n"
            "refresh-scores   = rerun ipd3 score table + all reports\n"
            "blocking         = blocking report per drug\n"
            "step6-reports    = run forecast step6 + 2bqreport + forecast_report\n"
            "forecast-reports = 2bqreport + forecast_report only (default)"
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Max parallel workers per stage (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")

    # Resume / checkpoint controls.
    # Default policy: resume is ON when --mode is "forecast" (the user has
    # explicitly asked to start from there and almost certainly wants to
    # skip already-done drugs); OFF for other modes to keep their previous
    # behaviour unchanged. Use --no-resume to force a fresh run, or
    # --resume to enable it for other modes.
    # Checkpoint coverage spans the full forecast → merge → ipd → reports
    # chain. Each per-drug stage marks (step, drug) after success; merge is
    # a single global marker.
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=None,
        help=(
            "Skip drugs/stages already marked done in the GCS checkpoint "
            "store. Covers forecast (step3/4/5/6), merge, IPD (ipd2/3/4 "
            "all per-drug), and reports. Default ON for --mode forecast, "
            "OFF otherwise."
        ),
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force a fresh run, ignoring any existing checkpoints.",
    )
    parser.add_argument(
        "--clear-checkpoints",
        nargs="?", const="__ALL__", default=None, metavar="STEP",
        help=(
            "Delete checkpoints before running. Pass a step key "
            "(e.g. 'step4', 'ipd2bq', 'reports', 'merge') to clear just "
            "that stage, or no value to clear all. Use this after fixing "
            "a bug to force the next run to redo work."
        ),
    )

    args = parser.parse_args()

    t0 = time.time()

    task_index = os.environ.get("CLOUD_RUN_TASK_INDEX", "N/A")
    task_count = os.environ.get("CLOUD_RUN_TASK_COUNT", "N/A")
    banner(
        f"LOE PIPELINE — mode={args.mode} | workers={args.workers} | "
        f"task={task_index}/{task_count}"
    )

    # Discover drugs from clinical_efficacy BQ table, then shard for this task
    all_drugs = discover_drugs()
    drugs = shard_drugs(all_drugs)

    if not drugs:
        print(f"{YELLOW}This task has no drugs to process. Exiting cleanly.{RESET}")
        return

    # Resolve resume default based on mode (see argparse help above).
    if args.resume is None:
        args.resume = (args.mode == "forecast")

    # Clear checkpoints if requested (before any work starts).
    if args.clear_checkpoints is not None:
        try:
            from cog import forecast_checkpoint as _ckpt
            if args.clear_checkpoints == "__ALL__":
                n = _ckpt.clear()
                print(f"[CHECKPOINT] Cleared ALL forecast checkpoints ({n} blob(s) removed)")
            else:
                n = _ckpt.clear(args.clear_checkpoints)
                print(f"[CHECKPOINT] Cleared checkpoints for '{args.clear_checkpoints}' "
                      f"({n} blob(s) removed)")
        except Exception as e:
            print(f"{RED}[CHECKPOINT] Failed to clear: {e}{RESET}")

    if args.mode == "all":
        run_patents(drugs, args.workers, args.dry_run)
        run_forecast(drugs, args.workers, args.dry_run, resume=args.resume)
        run_merge(args.dry_run, resume=args.resume)
        run_ipd(drugs, args.workers, args.dry_run, resume=args.resume)
        run_reports(drugs, args.workers, args.dry_run, resume=args.resume)

    elif args.mode == "patents":
        run_patents(drugs, args.workers, args.dry_run)

    elif args.mode == "forecast":
        # Forecast onward: skip the patents stage but run everything after
        # forecast as well (merge → IPD → reports). Resume defaults to ON
        # for this mode so a re-run picks up where it left off across
        # every stage — step3/4/5/6, merge, ipd2/3/4, and reports.
        run_forecast(drugs, args.workers, args.dry_run, resume=args.resume)
        run_merge(args.dry_run, resume=args.resume)
        run_ipd(drugs, args.workers, args.dry_run, resume=args.resume)
        run_reports(drugs, args.workers, args.dry_run, resume=args.resume)

    elif args.mode == "ipd":
        run_ipd(drugs, args.workers, args.dry_run, resume=args.resume)

    elif args.mode == "reports":
        run_reports(drugs, args.workers, args.dry_run, resume=args.resume)

    elif args.mode == "refresh-scores":
        # Step 1: Rerun ipd3 with --refresh-scores (deletes old score rows, recomputes)
        banner("IPD3 REFRESH SCORES")
        for drug in drugs:
            try:
                run_step(
                    f"IPD3 refresh scores: {drug}",
                    [PY, str(IPD3_SCRIPT), drug, "--refresh-scores"],
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"{RED}✗ IPD3 refresh failed for {drug}: {e}{RESET}")

        # Step 2: Run all reports
        run_reports(drugs, args.workers, args.dry_run, resume=args.resume)

    elif args.mode == "blocking":
        banner("BLOCKING REPORT (per drug)")
        for drug in drugs:
            try:
                run_step(
                    f"Blocking Report: {drug}",
                    [PY, str(BLOCKING_REPORT_SCRIPT), drug],
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"{RED}✗ Blocking report failed for {drug}: {e}{RESET}")

    elif args.mode == "step6-reports":
        # Step 1: Run forecast step6 per drug
        banner("FORECAST STEP 6 (Patent Forecast Generator)")
        step6_script = FORECAST_DIR / "step6.py"
        step6_worker = functools.partial(
            _forecast_step_worker,
            step_script=step6_script,
            step_label="Step 6 — Patent Forecast Generator",
            extra_args=None,
            drug_flag="--drug",
            resume=args.resume,
        )
        run_parallel("Step 6 — Patent Forecast Generator", step6_worker, drugs, args.workers, args.dry_run)

        # Step 2: Run only 2bqreport + forecast_report (with --only filter)
        banner("REPORTS (2bqreport + forecast_report)")
        try:
            run_step(
                f"Reports — 2bqreport + forecast_report ({len(drugs)} drug(s))",
                [PY, str(REPORTS_SCRIPT), "--drugs"] + list(drugs)
                + ["--only", "2bqreport.py", "forecast_report.py"],
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"{RED}✗ Reports failed: {e}{RESET}")

    elif args.mode == "forecast-reports":
        banner("REPORTS (2bqreport + forecast_report + blocking)")
        try:
            run_step(
                f"Reports — 2bqreport + forecast_report ({len(drugs)} drug(s))",
                [PY, str(REPORTS_SCRIPT), "--drugs"] + list(drugs)
                + ["--only", "2bqreport.py", "forecast_report.py"],
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"{RED}✗ Reports failed: {e}{RESET}")

        # Blocking report per drug
        for drug in drugs:
            try:
                run_step(
                    f"Blocking Report: {drug}",
                    [PY, str(BLOCKING_REPORT_SCRIPT), drug],
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"{RED}✗ Blocking report failed for {drug}: {e}{RESET}")

    banner(f"DONE — {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
