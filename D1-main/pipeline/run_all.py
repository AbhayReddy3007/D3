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
  3. Merge     — merge_to_master_loe.py (single step, runs once)
  4. IPD       — ipd2bq + ipd3bq parallelised across drugs, then ipd4bq parallelised across drugs
  5. Reports   — reports.py parallelised across drugs

Usage:
  python run_all.py                   # full pipeline
  python run_all.py --mode patents    # only patents stage
  python run_all.py --mode forecast   # only forecast stage
  python run_all.py --mode ipd        # only IPD stage
  python run_all.py --mode reports    # only reports stage
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
REPORTS_SCRIPT = BASE_DIR / "reports.py"

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


def _forecast_step_worker(drug, dry_run, step_script=None, step_label=None, extra_args=None):
    """Run a single forecast step for one drug.

    Parameter order matches run_parallel's convention: (drug, dry_run, ...).
    The step_script/step_label/extra_args are bound via functools.partial
    in run_forecast (a top-level function + partial is picklable, whereas
    a nested closure is not — that's what causes
    "Can't pickle local object 'run_forecast.<locals>._step_worker'").
    """
    try:
        cmd = [PY, str(step_script), "--drug", drug]
        if extra_args:
            cmd.extend(extra_args)
        run_step(
            f"{step_label}: {drug}",
            cmd,
            dry_run=dry_run,
        )
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


def _ipd_worker(drug, dry_run):
    """ipd2bq + ipd3bq for a single drug."""
    try:
        run_step(
            f"IPD2 BQ upload: {drug}",
            [PY, str(IPD2_SCRIPT), drug],
            dry_run=dry_run,
        )
        run_step(
            f"IPD3 BQ upload: {drug}",
            [PY, str(IPD3_SCRIPT), drug],
            dry_run=dry_run,
        )
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


def _ipd4_worker(drug, dry_run):
    """ipd4bq for a single drug."""
    try:
        run_step(
            f"IPD4 BQ upload: {drug}",
            [PY, str(IPD4_SCRIPT), drug],
            dry_run=dry_run,
        )
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


def _reports_worker(drug, dry_run):
    """reports.py for a single drug."""
    try:
        run_step(
            f"Reports: {drug}",
            [PY, str(REPORTS_SCRIPT), drug],
            dry_run=dry_run,
        )
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


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
            drug = futures[future]
            try:
                drug_name, ok, err = future.result()
            except Exception as exc:
                drug_name, ok, err = drug, False, str(exc)

            done += 1
            if ok:
                print(f"  {GREEN}[{done}/{total}] ✓ {drug_name}{RESET}")
            else:
                print(f"  {RED}[{done}/{total}] ✗ {drug_name}: {err}{RESET}")
                failed.append(drug_name)

    if failed:
        print(f"\n{RED}✗ {label} — {len(failed)} drug(s) failed: {failed}{RESET}")
        print(f"  Pipeline halted.")
        sys.exit(1)

    print(f"\n  {GREEN}✓ {label} — all {total} drug(s) completed{RESET}\n")


# ── Pipeline stages ──────────────────────────────────────────────────────────

def run_patents(drugs, workers, dry_run=False):
    banner("PATENT PROCESSING")
    run_parallel("Patents", _patent_worker, drugs, workers, dry_run)


def run_forecast(drugs, workers, dry_run=False):
    banner("FORECASTING PIPELINE")

    forecast_steps = [
        ("Step 3 — IP Landscape + Layering + Filing Analysis",
         FORECAST_DIR / "step3.py", ["--upload"]),
        ("Step 4 — Innovator Filing Patterns",
         FORECAST_DIR / "step4.py", None),
        ("Step 5 — Business Strategy Review",
         FORECAST_DIR / "step5.py", None),
        ("Step 6 — Patent Forecast Generator",
         FORECAST_DIR / "step6.py", None),
    ]

    for step_label, step_script, extra_args in forecast_steps:
        print(f"{BOLD}  {step_label}{RESET}")

        # Use functools.partial over a top-level function so the worker is
        # picklable for ProcessPoolExecutor. A nested closure here would
        # raise "Can't pickle local object 'run_forecast.<locals>._step_worker'".
        step_worker = functools.partial(
            _forecast_step_worker,
            step_script=step_script,
            step_label=step_label,
            extra_args=extra_args,
        )

        run_parallel(step_label, step_worker, drugs, workers, dry_run)


def run_merge(dry_run=False):
    """
    Merge runs once globally.  When sharded across Cloud Run tasks,
    only task 0 executes the merge; other tasks skip it.
    """
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))

    if task_count > 1 and task_index != 0:
        print(f"{YELLOW}[SHARD] Merge skipped on task {task_index} "
              f"(only task 0 runs merge){RESET}\n")
        return

    banner("MERGE → Master_LOE")
    run_step("merge_to_master_loe.py", [PY, str(MERGE_SCRIPT)], dry_run=dry_run)


def run_ipd(drugs, workers, dry_run=False):
    banner("IPD → BIGQUERY")

    # ipd2bq + ipd3bq — parallelised across drugs
    run_parallel("IPD (ipd2bq + ipd3bq)", _ipd_worker, drugs, workers, dry_run)

    # ipd4bq — parallelised across drugs
    run_parallel("IPD4 BQ upload", _ipd4_worker, drugs, workers, dry_run)


def run_reports(drugs, workers, dry_run=False):
    banner("REPORTS GENERATION")
    run_parallel("Reports", _reports_worker, drugs, workers, dry_run)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOE Pipeline orchestrator")
    parser.add_argument(
        "--mode",
        choices=["all", "patents", "forecast", "ipd", "reports"],
        default="all",
        help=(
            "all      = full pipeline: patents → forecast → merge → ipd → reports (default)\n"
            "patents  = patent pipeline only\n"
            "forecast = forecast steps only\n"
            "ipd      = IPD BQ upload only\n"
            "reports  = reports only"
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Max parallel workers per stage (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
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

    if args.mode == "all":
        run_patents(drugs, args.workers, args.dry_run)
        run_forecast(drugs, args.workers, args.dry_run)
        run_merge(args.dry_run)
        run_ipd(drugs, args.workers, args.dry_run)
        run_reports(drugs, args.workers, args.dry_run)

    elif args.mode == "patents":
        run_patents(drugs, args.workers, args.dry_run)

    elif args.mode == "forecast":
        run_forecast(drugs, args.workers, args.dry_run)

    elif args.mode == "ipd":
        run_ipd(drugs, args.workers, args.dry_run)

    elif args.mode == "reports":
        run_reports(drugs, args.workers, args.dry_run)

    banner(f"DONE — {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
