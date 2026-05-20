#!/usr/bin/env python3
"""
run_all.py — Local parallel pipeline orchestrator
──────────────────────────────────────────────────
Runs the full LOE pipeline locally, parallelising drug-level work
within each stage using a process pool.

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


# ── GCS drug discovery ───────────────────────────────────────────────────────

def discover_drugs():
    """Discover all drug folders from GCS."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

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
            folders[parts[prefix_depth]] = True

    drugs = sorted(folders.keys())
    print(f"[DISCOVERY] {len(drugs)} drug(s) in gs://{bucket_name}/{prefix}")
    return drugs


# ── Per-drug worker functions (run in child processes) ────────────────────────
# Each function processes ONE drug and returns (drug, ok, error_msg).
# They must be top-level functions so ProcessPoolExecutor can pickle them.

def _patent_worker(drug, dry_run):
    """Patent pipeline + BQ upload for a single drug."""
    try:
        cache_dir = BASE_DIR / "cog" / "results_cache"
        run_step(
            f"Patent pipeline: {drug}",
            [PY, "-m", "cog.main", drug],
            cwd=BASE_DIR, dry_run=dry_run,
        )
        run_step(
            f"BQ upload: {drug}",
            [PY, str(BQ_SCRIPT), "--drug", drug, "--cache-dir", str(cache_dir)],
            dry_run=dry_run,
        )
        return (drug, True, None)
    except Exception as e:
        return (drug, False, str(e))


def _forecast_step_worker(drug, step_script, step_label, extra_args=None, dry_run=False):
    """Run a single forecast step for one drug."""
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
        print(f"{YELLOW}No drugs to process. Skipping.{RESET}\n")
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

        def _step_worker(drug, dry_run, _script=step_script, _label=step_label, _extra=extra_args):
            return _forecast_step_worker(drug, _script, _label, _extra, dry_run)

        run_parallel(step_label, _step_worker, drugs, workers, dry_run)


def run_merge(dry_run=False):
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
    banner(f"LOE PIPELINE — mode={args.mode} | workers={args.workers}")

    # Discover drugs once, reuse across stages
    drugs = discover_drugs()

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
