#!/usr/bin/env python3
"""
main.py — CLI entry point for the patent analysis pipeline.
Runs without Google ADK — plain asyncio from the terminal.

Usage:
    # Process a single drug (10 patents indexed in parallel)
    python -m cog.main semaglutide

    # Process a single drug, specific jurisdictions
    python -m cog.main semaglutide --jurisdictions US EP

    # Process ALL drugs found in GCS (one drug at a time, 10 patents in parallel each)
    python -m cog.main --all

    # Force re-index (ignore cache)
    python -m cog.main semaglutide --reindex

    # Override parallel patent count (default: 10)
    python -m cog.main semaglutide --concurrency 5
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not needed on Cloud Run

# Default number of patents to index in parallel within a single drug.
# Matches MAX_CONCURRENCY in indexer.py — override via --concurrency or
# the INDEXER_CONCURRENCY env var.
DEFAULT_CONCURRENCY = 10


def _print_single_drug_result(result: dict):
    """Pretty-print the result of a single-drug run."""
    drug   = result.get("drug_name", "?")
    cached = result.get("from_cache", False)
    err    = result.get("error")

    print(f"\n{'='*60}")
    print(f"  RESULT: {drug}")
    print(f"{'='*60}")

    if err:
        print(f"  ERROR: {err}")
        return

    print(f"  Analysis Date : {result.get('analysis_date')}")
    print(f"  Phase Source  : {result.get('phase_data_source')}")
    print(f"  From Cache    : {cached}")
    print(f"  Time          : {result.get('processing_time_seconds')}s")
    print(f"  Patents       : {len(result.get('patents', []))}")

    patents = result.get("patents", [])
    if patents:
        print(f"\n  {'Patent':<20} {'Jur':<5} {'Tag':<15} {'Category':<20} {'Filed':<12} {'Granted':<12}")
        print(f"  {'-'*84}")
        for p in patents:
            print(
                f"  {p.get('patent_number','?'):<20} "
                f"{p.get('jurisdiction','?'):<5} "
                f"{p.get('tag','?'):<15} "
                f"{(p.get('blocking_category') or 'N/A'):<20} "
                f"{(p.get('filing_date') or '?'):<12} "
                f"{(p.get('grant_date') or '?'):<12}"
            )

    src_files = result.get("source_files", [])
    if src_files:
        print(f"\n  Source Files ({len(src_files)}):")
        for f in src_files:
            print(f"    • {f}")

    excel = result.get("excel_path")
    combined = result.get("combined_excel_path")
    print(f"\n  Excel         : {excel or 'N/A'}")
    print(f"  Combined Excel: {combined or 'N/A'}")
    print()


def _print_all_drugs_result(result: dict):
    """Pretty-print the result of an all-drugs run."""
    print(f"\n{'='*60}")
    print(f"  ALL DRUGS — SUMMARY")
    print(f"{'='*60}")
    print(f"  Status    : {result.get('status')}")
    print(f"  Total     : {result.get('total_drugs', 0)}")
    print(f"  Succeeded : {result.get('succeeded', 0)}")
    print(f"  Failed    : {result.get('failed', 0)}")

    for r in result.get("results", []):
        status = r.get("status", "?")
        drug   = r.get("drug", "?")
        pats   = r.get("patents", "?")
        excel  = r.get("excel_path") or "-"
        err    = r.get("error", "")
        line   = f"  {drug:<25} {status:<10} patents={pats:<5}"
        if excel != "-":
            line += f"  excel={excel}"
        if err:
            line += f"  ERROR: {err}"
        print(line)

    print(f"\n  Excel Dir      : {result.get('excel_dir', 'N/A')}")
    print(f"  Combined Excel : {result.get('combined_excel_path', 'N/A')}")
    print()


async def run_single(drug_name: str, reindex: bool, jurisdictions: list, concurrency: int):
    from .tools import get_dimension_i_patent_data

    result = await get_dimension_i_patent_data(
        drug_name=drug_name,
        reindex=reindex,
        jurisdictions=jurisdictions or None,
        max_concurrency=concurrency,
    )
    _print_single_drug_result(result)
    return result


async def run_all(concurrency: int):
    from .agent import process_all_drugs

    result = await process_all_drugs(max_concurrency=concurrency)
    _print_all_drugs_result(result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Patent analysis pipeline — CLI mode (no ADK required)",
    )
    parser.add_argument(
        "drug_name",
        nargs="?",
        default=None,
        help="Drug name to analyse (e.g. semaglutide). Omit if using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL drug folders found in GCS.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing (ignore AlloyDB cache).",
    )
    parser.add_argument(
        "--jurisdictions",
        nargs="*",
        default=None,
        help="Filter to specific jurisdictions (e.g. US EP).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of patents to index in parallel within a single drug "
             f"(default: {DEFAULT_CONCURRENCY}). Also controllable via the "
             f"INDEXER_CONCURRENCY env var.",
    )

    args = parser.parse_args()

    if not args.all and not args.drug_name:
        parser.error("Provide a drug name or use --all")

    if args.all and args.drug_name:
        parser.error("Cannot use --all together with a specific drug name")

    t0 = time.time()

    if args.all:
        result = asyncio.run(run_all(concurrency=args.concurrency))
    else:
        result = asyncio.run(
            run_single(args.drug_name, args.reindex, args.jurisdictions, args.concurrency)
        )

    elapsed = time.time() - t0
    print(f"[DONE] Total wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
