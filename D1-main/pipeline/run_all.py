#!/usr/bin/env python3
"""
run_all.py — Cloud Run Jobs orchestrator
─────────────────────────────────────────
Modes:

  python run_all.py
    (default --mode launch)
    Submits THREE Cloud Run Jobs via the GCP API, waits for each before
    proceeding to the next, then runs reports.py locally:
      Job 1  — patents   (10 parallel tasks, drug-sharded)
      Job 2  — forecast  (1 task: steps 3-6 + merge)
      Job 3  — ipd       (10 parallel tasks, drug-sharded)
      Local  — reports.py (single step, runs here after all jobs finish)

  --mode patents   (inside Job 1, 10 parallel Cloud Run tasks)
    Each task takes its shard of drugs, runs the patent pipeline + BQ upload.

  --mode forecast  (inside Job 2, 1 Cloud Run task)
    Runs forecast step3→4→5→6, then merges loe_table + forecasted_loe → Master_LOE.

  --mode ipd       (inside Job 3, 10 parallel Cloud Run tasks)
    Each task takes its shard of drugs and runs ipd2bq + ipd3bq on that shard.
    Task index 0 also runs ipd4bq (global, runs once only).

  --mode all       (local sequential testing, no Cloud Run)
    Runs patents → forecast → merge → ipd (all drugs) → reports sequentially.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent
BQ_SCRIPT      = BASE_DIR / "2bq.py"
FORECAST_DIR   = BASE_DIR / "forecast-main"
MERGE_SCRIPT   = BASE_DIR / "merge_to_master_loe.py"
IPD2_SCRIPT    = BASE_DIR / "ipd2bq.py"
IPD3_SCRIPT    = BASE_DIR / "ipd3bq.py"
IPD4_SCRIPT    = BASE_DIR / "ipd4bq.py"
REPORTS_SCRIPT = BASE_DIR / "reports.py"

PARALLEL_TASKS = 10  # number of Cloud Run parallel tasks for patents + ipd jobs

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
PY     = sys.executable


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(text):
    print(f"\n{BOLD}{'═' * 64}")
    print(f"  {text}")
    print(f"{'═' * 64}{RESET}\n")


def run_step(label, cmd, cwd=None, dry_run=False):
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
        print(f"\n  {RED}✗ FAILED (exit {result.returncode}) after {elapsed:.1f}s{RESET}")
        print(f"  Pipeline halted at: {label}")
        sys.exit(result.returncode)
    print(f"  {GREEN}✓ Done in {elapsed:.1f}s{RESET}\n")


# ── GCS drug discovery ────────────────────────────────────────────────────────

def discover_drugs():
    """Discover all drug folders from GCS."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Not needed on Cloud Run
    from google.cloud import storage
    from google.oauth2 import service_account

    bucket_name = os.getenv("GCS_BUCKET_NAME")
    prefix      = os.getenv("GCS_PATENTS_PREFIX", "patents").rstrip("/") + "/"

    if not bucket_name:
        print(f"{RED}ERROR: GCS_BUCKET_NAME not set{RESET}")
        sys.exit(1)

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        creds  = service_account.Credentials.from_service_account_file(creds_path)
        client = storage.Client(credentials=creds)
    else:
        client = storage.Client()

    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    prefix_depth = len(prefix.split("/")) - 1
    folders = {}
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > prefix_depth + 1:
            folder = parts[prefix_depth]
            folders[folder] = True

    drugs = sorted(folders.keys())
    print(f"[DISCOVERY] {len(drugs)} drug(s) in gs://{bucket_name}/{prefix}")
    return drugs


def get_my_shard(drugs):
    """Split drugs by CLOUD_RUN_TASK_INDEX / CLOUD_RUN_TASK_COUNT."""
    idx   = int(os.getenv("CLOUD_RUN_TASK_INDEX", "0"))
    count = int(os.getenv("CLOUD_RUN_TASK_COUNT", "1"))
    if count <= 1:
        return drugs
    shard = [d for i, d in enumerate(drugs) if i % count == idx]
    print(f"[SHARD] Task {idx + 1}/{count} → {len(shard)} drug(s): {shard}")
    return shard


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_patents(dry_run=False):
    banner("PATENT PROCESSING (parallel shard)")
    drugs = discover_drugs()
    shard = get_my_shard(drugs)
    if not shard:
        print(f"{YELLOW}No drugs in this shard. Done.{RESET}")
        return
    cache_dir = BASE_DIR / "cog" / "results_cache"
    for i, drug in enumerate(shard, 1):
        run_step(
            f"[{i}/{len(shard)}] Patent pipeline: {drug}",
            [PY, "-m", "cog.main", drug],
            cwd=BASE_DIR, dry_run=dry_run,
        )
        run_step(
            f"[{i}/{len(shard)}] BQ upload: {drug}",
            [PY, str(BQ_SCRIPT), "--drug", drug, "--cache-dir", str(cache_dir)],
            dry_run=dry_run,
        )


def run_forecast(dry_run=False):
    banner("FORECASTING PIPELINE")
    for label, cmd in [
        ("Step 3 — IP Landscape + Layering + Filing Analysis",
         [PY, str(FORECAST_DIR / "step3.py"), "--upload"]),
        ("Step 4 — Innovator Filing Patterns",
         [PY, str(FORECAST_DIR / "step4.py")]),
        ("Step 5 — Business Strategy Review",
         [PY, str(FORECAST_DIR / "step5.py")]),
        ("Step 6 — Patent Forecast Generator",
         [PY, str(FORECAST_DIR / "step6.py")]),
    ]:
        run_step(label, cmd, dry_run=dry_run)


def run_merge(dry_run=False):
    banner("MERGE → Master_LOE")
    run_step("merge_to_master_loe.py", [PY, str(MERGE_SCRIPT)], dry_run=dry_run)


def run_ipd(dry_run=False):
    """
    IPD processing — drug-sharded across parallel tasks.

    ipd2bq and ipd3bq both accept an optional positional `drug` argument,
    so each task processes only its shard of drugs (same pattern as patents).

    ipd4bq has no drug filter — it is a global step that must run exactly
    once, so only task index 0 executes it after finishing its own shard.
    """
    banner("IPD → BIGQUERY (parallel shard)")
    drugs    = discover_drugs()
    shard    = get_my_shard(drugs)
    task_idx = int(os.getenv("CLOUD_RUN_TASK_INDEX", "0"))

    if not shard:
        print(f"{YELLOW}No drugs in this shard. Done.{RESET}")
    else:
        for i, drug in enumerate(shard, 1):
            run_step(
                f"[{i}/{len(shard)}] IPD2 BQ upload: {drug}",
                [PY, str(IPD2_SCRIPT), drug],
                dry_run=dry_run,
            )
            run_step(
                f"[{i}/{len(shard)}] IPD3 BQ upload: {drug}",
                [PY, str(IPD3_SCRIPT), drug],
                dry_run=dry_run,
            )

    # ipd4bq is global (no drug arg) — run once on task 0 only
    if task_idx == 0:
        run_step(
            "IPD4 → BQ upload (global, task 0 only)",
            [PY, str(IPD4_SCRIPT)],
            dry_run=dry_run,
        )
    else:
        print(f"{YELLOW}  [task {task_idx}] Skipping ipd4bq (runs on task 0 only){RESET}\n")


def run_reports(dry_run=False):
    banner("REPORTS GENERATION")
    run_step("reports.py", [PY, str(REPORTS_SCRIPT)], dry_run=dry_run)


# ── Cloud Run launcher ────────────────────────────────────────────────────────

def submit_cloud_run_job(job_name, mode, task_count, dry_run=False):
    """
    Submit a Cloud Run Job execution via `gcloud` CLI.
    Blocks (--wait) until the execution finishes before returning.

    Required env vars:
      CLOUDRUN_REGION   — e.g. asia-south1  (defaults to asia-south1)
      GCP_PROJECT_ID    — GCP project        (falls back to BQ_PROJECT_ID)
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Not needed on Cloud Run

    region  = os.getenv("CLOUDRUN_REGION", "asia-south1")
    project = os.getenv("GCP_PROJECT_ID",  os.getenv("BQ_PROJECT_ID", ""))

    if not project:
        print(f"{RED}ERROR: GCP_PROJECT_ID (or BQ_PROJECT_ID) not set in env{RESET}")
        sys.exit(1)

    cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        f"--region={region}",
        f"--project={project}",
        f"--tasks={task_count}",
        "--wait",                        # block until all tasks finish
        "--format=value(name)",
        "--args", f"--mode={mode}",      # tell the container which mode to run
    ]

    print(f"{YELLOW}▶ Submitting Cloud Run Job: {job_name}  "
          f"(mode={mode}, tasks={task_count}){RESET}")
    print(f"  cmd: {' '.join(cmd)}")

    if dry_run:
        print(f"  {YELLOW}[DRY RUN] skipped{RESET}\n")
        return

    t0     = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  {RED}✗ Cloud Run Job '{job_name}' FAILED after {elapsed:.1f}s{RESET}")
        print(result.stderr)
        sys.exit(result.returncode)

    execution_name = result.stdout.strip()
    print(f"  {GREEN}✓ '{job_name}' finished in {elapsed:.1f}s  "
          f"execution={execution_name}{RESET}\n")


def run_launch(dry_run=False):
    """
    Default entry point — invoked by `python run_all.py`.

    Submits Cloud Run Jobs one at a time, waiting for each to fully complete
    before the next one starts. After all three jobs are done, runs
    reports.py locally as the final step.

    Job names are read from env vars so you can configure them without
    changing this file:
      CLOUDRUN_JOB_PATENTS   (default: loe-patents-job)
      CLOUDRUN_JOB_FORECAST  (default: loe-forecast-job)
      CLOUDRUN_JOB_IPD       (default: loe-ipd-job)
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Not needed on Cloud Run

    patents_job  = os.getenv("CLOUDRUN_JOB_PATENTS",  "loe-patents-job")
    forecast_job = os.getenv("CLOUDRUN_JOB_FORECAST", "loe-forecast-job")
    ipd_job      = os.getenv("CLOUDRUN_JOB_IPD",      "loe-ipd-job")

    banner(f"LAUNCH — orchestrating Cloud Run Jobs  ({PARALLEL_TASKS} parallel tasks where applicable)")

    # 1. Patents — 10 parallel tasks, each processes its drug shard
    submit_cloud_run_job(patents_job,  mode="patents",  task_count=PARALLEL_TASKS, dry_run=dry_run)

    # 2. Forecast + merge — 1 task (global, runs once)
    submit_cloud_run_job(forecast_job, mode="forecast", task_count=1,              dry_run=dry_run)

    # 3. IPD — 10 parallel tasks, each processes its drug shard
    #    (task 0 additionally runs ipd4bq which has no drug filter)
    submit_cloud_run_job(ipd_job,      mode="ipd",      task_count=PARALLEL_TASKS, dry_run=dry_run)

    # 4. Reports — runs locally here, after all Cloud Run jobs are done
    run_reports(dry_run=dry_run)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOE Pipeline orchestrator")
    parser.add_argument(
        "--mode",
        choices=["launch", "patents", "forecast", "ipd", "all"],
        default="launch",
        help=(
            "launch   = submit Cloud Run Jobs + wait (default, run with: python run_all.py)\n"
            "patents  = Job 1 inside Cloud Run (patent pipeline, drug-sharded)\n"
            "forecast = Job 2 inside Cloud Run (forecast steps + merge, single task)\n"
            "ipd      = Job 3 inside Cloud Run (ipd2/3/4 BQ upload, drug-sharded)\n"
            "all      = local sequential run for testing (no Cloud Run)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    t0    = time.time()
    idx   = os.getenv("CLOUD_RUN_TASK_INDEX", "0")
    count = os.getenv("CLOUD_RUN_TASK_COUNT", "1")
    banner(f"LOE PIPELINE — mode={args.mode} | task {int(idx)+1}/{count}")

    if args.mode == "launch":
        run_launch(args.dry_run)

    elif args.mode == "patents":
        run_patents(args.dry_run)

    elif args.mode == "forecast":
        run_forecast(args.dry_run)
        run_merge(args.dry_run)

    elif args.mode == "ipd":
        run_ipd(args.dry_run)

    elif args.mode == "all":
        # Local sequential run — for testing without Cloud Run
        run_patents(args.dry_run)
        run_forecast(args.dry_run)
        run_merge(args.dry_run)
        run_ipd(args.dry_run)
        run_reports(args.dry_run)

    banner(f"DONE — {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
