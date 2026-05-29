"""
batch_innovator_patterns.py
───────────────────────────
Parallel batch processing + BigQuery storage
"""

import os
import sys
import csv
import asyncio
import argparse
import random
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv(override=True)

# ✅ CONFIG
MAX_CONCURRENT_DRUGS = 3
TIMEOUT_PER_DRUG = 300
MAX_RETRIES = 3
LOG_FILE = "batch_errors.log"

# ✅ BIGQUERY CONFIG
BQ_LOCATION = "asia-south1"
PROJECT_ID = "cognito-prod-394707"
DATASET_ID = "cognito_prod_datamart"
TABLE_ID = "filing_pattern_table"
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

OUTPUT_DIR = None  # No longer used — output goes to GCS

# Import GCS cache
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cog import gcs_cache

_STEP4_SUBFOLDER = "innovator_patterns"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def log_error(msg):
    print(f"[STEP4 ERROR] {msg}")

# ─────────────────────────────────────────────
# IMPORT LOCAL MODULES
# ─────────────────────────────────────────────
# The containing directory is `forecast-main`, which is NOT a valid Python
# package name (hyphens are illegal). The previous code tried
#   importlib.import_module(f"{_pkg}.gcs_lister")
# with _pkg = "forecast-main", which can never resolve. The actual location
# of the supporting modules is the sibling `cog/` package (an `__init__.py`
# is present there), so we import from `cog` directly — the same pattern
# this file already uses for `gcs_cache` above.
_here = Path(__file__).resolve().parent
_parent = _here.parent

for _p in [str(_here), str(_parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ✅ GCS — lives in `cog/`
from cog import gcs_lister as _gcs
get_gcs_client     = _gcs.get_gcs_client
GCS_BUCKET_NAME    = _gcs.GCS_BUCKET_NAME
GCS_PATENTS_PREFIX = _gcs.GCS_PATENTS_PREFIX

# ─────────────────────────────────────────────
# Gemini configuration (for inlined analysis)
# ─────────────────────────────────────────────
from google import genai
from google.genai import types

_GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_GEMINI_MODEL          = "gemini-2.5-flash"
_GEMINI_MAX_RETRIES    = 3      # API-level retries inside one analysis call
_GEMINI_INITIAL_BACKOFF = 2.0
_gemini_client = genai.Client(api_key=_GEMINI_API_KEY) if _GEMINI_API_KEY else None

_REGISTRIES = [
    {"name": "ClinicalTrials.gov",          "url": "https://clinicaltrials.gov/"},
    {"name": "ChiCTR",                      "url": "https://www.chictr.org.cn/"},
    {"name": "EU Clinical Trials Register", "url": "https://www.clinicaltrialsregister.eu/"},
    {"name": "CTRI",                        "url": "https://ctri.nic.in/"},
    {"name": "JRCT",                        "url": "https://rctportal.niph.go.jp/en/"},
    {"name": "ANZCTR",                      "url": "https://www.anzctr.org.au/"},
    {"name": "CRIS",                        "url": "https://cris.nih.go.kr/"},
    {"name": "ReBEC",                       "url": "https://ensaiosclinicos.gov.br/"},
]

# Optional vector-DB hook for storing analysis results. Lives in cog/ if
# present; missing in this checkout, so wrap in try/except and proceed without.
_vectordb_ingest = None
try:
    from cog import strategy_vectordb as _vdb  # type: ignore
    _vectordb_ingest = _vdb.ingest_innovator_pattern
    print("[VECTORDB] strategy_vectordb loaded — Step 4 results will be stored in innovator_filing_pattern_db")
except Exception as _ve:
    print(f"[VECTORDB] strategy_vectordb not available — results will only go to BQ/GCS: {_ve}")


# ─────────────────────────────────────────────
# Gemini call helpers (inlined from test_innovator_filing_patterns)
# ─────────────────────────────────────────────

def _sync_gemini_call(model: str, contents: list, config) -> str:
    """Synchronous streaming Gemini API call. Called via asyncio.to_thread."""
    if _gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY / GOOGLE_API_KEY not set — Step 4 cannot run "
            "innovator filing pattern analysis."
        )
    response_text = ""
    print("  Receiving", end="", flush=True)
    for chunk in _gemini_client.models.generate_content_stream(
        model=model, contents=contents, config=config,
    ):
        if chunk.text:
            response_text += chunk.text
            print(".", end="", flush=True)
    print(" Done!")
    return response_text.strip()


async def _gemini_call_with_search(prompt: str) -> str:
    """Gemini call with Google Search grounding + 429/quota retry backoff."""
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    tools    = [types.Tool(google_search=types.GoogleSearch())]
    config   = types.GenerateContentConfig(tools=tools)

    retry_count   = 0
    backoff_delay = _GEMINI_INITIAL_BACKOFF

    while retry_count <= _GEMINI_MAX_RETRIES:
        try:
            return await asyncio.to_thread(_sync_gemini_call, _GEMINI_MODEL, contents, config)
        except Exception as e:
            if any(err in str(e).lower() for err in ["429", "rate limit", "quota"]):
                retry_count += 1
                if retry_count > _GEMINI_MAX_RETRIES:
                    print(f"  Max retries exceeded: {e}")
                    raise
                print(f"  Rate limit hit, waiting {backoff_delay}s...")
                await asyncio.sleep(backoff_delay)
                backoff_delay *= 2
            else:
                print(f"  Gemini API error: {e}")
                raise
    return ""


def _parse_json_response(response: str) -> dict:
    """Parse JSON from Gemini response, tolerating markdown fences and preambles."""
    if not response:
        return {}
    import json
    response = response.strip()
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    json_start = response.find('{')
    if json_start > 0:
        response = response[json_start:]
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Response preview: {response[:300]}...")
        return {}


# ─────────────────────────────────────────────
# Core analysis (per drug)
# ─────────────────────────────────────────────

async def analyze_innovator_filing_patterns(drug_name: str) -> dict:
    """Analyse innovator patent filing patterns for a single drug.

    Returns a dict shaped like:
        {
          "innovators": [
            {"company": str, "characterization": str,
             "confidence": float, "rationale": str},
            ...
          ],
          "combined_characterization": str,
          "combined_confidence": float,
          "combined_rationale": str,
        }
    Returns an empty dict on failure.
    """
    print(f"\n{'='*80}")
    print(f"  INNOVATOR FILING PATTERN ANALYSIS — {drug_name}")
    print(f"{'='*80}")
    print(f"  Model: {_GEMINI_MODEL}")

    registry_urls = "\n".join(f"- {r['url']}" for r in _REGISTRIES)

    prompt = f"""You are a pharmaceutical patent intelligence analyst. Analyze the patent filing patterns for the innovator company behind "{drug_name}".

TASK:
1. Identify ALL innovator companies for {drug_name} — include co-developers, original discoverers, and licensors that hold patents on this drug. Many drugs have 2+ innovators (e.g., co-development agreements, in-licensed compounds).
2. For EACH innovator: search Espacenet, Google Patents, and SEC EDGAR to analyze their patent filing behavior independently.
3. Look at each company's patterns across their broader portfolio (not just {drug_name}).

DATA SOURCES (search ALL for complete data):
- https://worldwide.espacenet.com/
- https://patents.google.com/
- https://www.sec.gov/cgi-bin/browse-edgar
{registry_urls}

ANALYZE - For each innovator, evaluate whether they typically:

1. FILES MULTIPLE CONTINUATION APPLICATIONS:
   - Count continuation-in-part (CIP) and continuation applications
   - Look for chains of related applications spanning many years
   - Check if they file continuations near patent expiry to extend protection
   - Identify "submarine" continuation strategies (delayed prosecution)

2. EXPANDS PROTECTION LATE IN DEVELOPMENT:
   - Look for patents filed after Phase 3 trials or FDA approval
   - Identify late-filed method-of-use patents for new indications
   - Check for dosing regimen patents filed years after original compound patent
   - Look for "evergreening" tactics near original patent expiry

3. BUILDS DENSE PATENT THICKETS:
   - Count total patents covering a single drug (>20 = dense thicket)
   - Identify overlapping claims across multiple patents
   - Look for blocking patents that prevent generic entry from multiple angles
   - Check for picket fence strategies (many narrow patents vs few broad ones)

4. FILES MULTIPLE FORMULATION VARIATIONS:
   - Count polymorph/crystal form patents
   - Identify extended-release, controlled-release, or modified-release patents
   - Look for combination product patents (fixed-dose combinations)
   - Check for device/delivery system patents (auto-injectors, pens, patches)
   - Identify pediatric formulation patents
   - Look for salt form and prodrug patents

5. OTHER PATTERNS:
   - Patent family sizes across their portfolio
   - Geographic filing strategy (US-only vs worldwide)
   - Historical behavior consistency across other products

OUTPUT:
For EACH innovator, provide a SHORT 1-LINE TITLE characterizing their IP filing strategy.
Maximum 5 words, ideally 3-4 words. Be specific and situational based on evidence found.

Examples (create your own based on evidence):
- "Dense lifecycle patent thickets"
- "Formulation-focused secondary patents"
- "Minimal single composition patent"
- "Heavy continuation near expiry"
- "Broad method-of-use defense"
- "Late-stage polymorph evergreening"
- "Extensive formulation/device patents"

Also provide a COMBINED characterization that reflects the overall IP barrier landscape created by ALL innovators together.
If strategies differ, the combined characterization should reflect the most aggressive / highest-barrier strategy (since generics must clear all thickets).

Return as JSON:

{{
  "innovators": [
    {{
      "company": "",
      "characterization": "",
      "confidence": 0.0,
      "rationale": ""
    }}
  ],
  "combined_characterization": "",
  "combined_confidence": 0.0,
  "combined_rationale": ""
}}

IMPORTANT:
- If only one innovator exists, "innovators" array will have one entry and combined_characterization = that entry's characterization.
- characterization: SHORT 1-LINE TITLE (max 5 words, ideally 3-4) - not a paragraph
- confidence per innovator: How well the evidence SUPPORTS THE SPECIFIC CHARACTERIZATION YOU CHOSE (not just how many patents you found).
  Score based on these 3 factors:

  HARD CEILING RULES (apply first, before scoring):
  - If you could NOT find specific patent numbers for this drug → max score is 0.5
  - If the drug is early-stage (Phase 1/2) with limited filing history → max score is 0.6
  - If you are inferring strategy from company reputation, not verified filings → max score is 0.4
  - If the drug is obscure or has sparse patent data online → max score is 0.5

  Then score these 3 factors (within the ceiling):

  1. PATTERN CONSISTENCY (0–0.4): Do the patents found all point to the same strategy, or are they mixed?
     - All patents clearly show the same pattern → 0.4
     - Mostly consistent with some outliers → 0.2–0.3
     - Mixed/conflicting signals → 0.1

  2. EVIDENCE DIRECTNESS (0–0.4): Did you find patents that directly confirm the strategy you named?
     - Found and verified specific patent numbers (e.g., US10XXXXXX) → 0.4
     - Found descriptions/summaries but not verified patent numbers → 0.2
     - Inferred from general company reputation only → 0.0–0.1

  3. PORTFOLIO BREADTH (0–0.2): Did the pattern hold across multiple products (not just {drug_name})?
     - Pattern confirmed across 3+ products with evidence → 0.2
     - Confirmed for {drug_name} only → 0.1
     - Could not verify across portfolio → 0.0

  Add the 3 factors, then apply the ceiling. 0.8+ requires: specific patent numbers found + consistent pattern + multi-product evidence.
  NEVER assign 1.0 — maximum allowed score is 0.9.

- combined_confidence: average of all innovator confidence scores, then apply the same ceiling rules above.

- rationale: AFTER calculating confidence using the rules above, add a plain English explanation of the IP strategy. Do NOT include patent numbers or scoring breakdowns.
"""

    print("  Analyzing filing patterns...")
    response = await _gemini_call_with_search(prompt)
    data = _parse_json_response(response)

    if not data:
        print("  Failed to analyze filing patterns")
        return {}

    return data

# ─────────────────────────────────────────────
# ✅ BIGQUERY SETUP
# ─────────────────────────────────────────────
def get_bq_client():
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        return bigquery.Client(credentials=creds, project=PROJECT_ID)
    return bigquery.Client(project=PROJECT_ID)


def create_dataset_if_not_exists(client):
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = BQ_LOCATION

    try:
        client.get_dataset(dataset_ref)
        print("[BQ] Dataset exists")
    except Exception:
        client.create_dataset(dataset)
        print("[BQ] Dataset created")


def create_table_if_not_exists(client):
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    schema = [
        bigquery.SchemaField("drug", "STRING"),
        bigquery.SchemaField("company", "STRING"),
        bigquery.SchemaField("characterization", "STRING"),
        bigquery.SchemaField("confidence", "STRING"),
        bigquery.SchemaField("rationale", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]

    table = bigquery.Table(table_ref, schema=schema)

    try:
        client.get_table(table_ref)
        print("[BQ] Table exists")
    except Exception:
        client.create_table(table)
        print("[BQ] Table created")


def insert_into_bq(client, drug_name, result):
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    rows = []
    for inv in result.get("innovators", []):
        rows.append({
            "drug": drug_name,
            "company": inv.get("company"),
            "characterization": inv.get("characterization"),
            "confidence": str(inv.get("confidence")),
            "rationale": inv.get("rationale"),
            "created_at": datetime.utcnow().isoformat()
        })

    if rows:
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            print(f"[BQ ERROR] {errors}")
        else:
            print(f"[BQ] Inserted {len(rows)} rows for {drug_name}")

# ─────────────────────────────────────────────
# GCS LIST
# ─────────────────────────────────────────────
def list_all_gcs_drugs():
    client = get_gcs_client()
    prefix = GCS_PATENTS_PREFIX.rstrip("/") + "/"

    blobs = list(client.list_blobs(GCS_BUCKET_NAME, prefix=prefix))

    drugs = sorted({b.name.split("/")[1] for b in blobs if "/" in b.name})
    print(f"[GCS] Found {len(drugs)} drugs")
    return drugs

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
def save_drug_csv(drug, result):
    """Write CSV to GCS instead of local disk."""
    import io as _io
    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Drug", "Company", "Characterization", "Confidence"])

    for inv in result.get("innovators", []):
        writer.writerow([
            drug,
            inv.get("company"),
            inv.get("characterization"),
            inv.get("confidence"),
        ])

    uri = gcs_cache.write_bytes(
        _STEP4_SUBFOLDER,
        f"{drug}.csv",
        buf.getvalue().encode("utf-8"),
        content_type="text/csv",
    )
    print(f"[STEP4] CSV saved to GCS: {uri}")
    return uri

# ─────────────────────────────────────────────
# PROCESS DRUG
# ─────────────────────────────────────────────
async def process_drug(i, total, drug, semaphore, bq_client, results, failed):

    print(f"\n[{i}/{total}] {drug}")

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                result = await asyncio.wait_for(
                    analyze_innovator_filing_patterns(drug),
                    timeout=TIMEOUT_PER_DRUG
                )

            if result:
                results[drug] = result

                save_drug_csv(drug, result)
                insert_into_bq(bq_client, drug, result)

                # Store in vector DB if the optional cog.strategy_vectordb
                # module is available. Failure here doesn't fail the drug —
                # the BQ/GCS writes already succeeded.
                if _vectordb_ingest is not None:
                    try:
                        _vectordb_ingest(drug, result)
                    except Exception as ve:
                        print(f"  [VECTORDB] Failed to store {drug}: {ve}")

                print(f"✅ Done: {drug}")
                return

            raise Exception("Empty result")

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + random.random()
                print(f"[Retry] {drug} in {wait:.2f}s")
                await asyncio.sleep(wait)
            else:
                print(f"[FAILED] {drug}: {e}")
                log_error(f"{drug}: {e}")
                failed.append((drug, str(e)))

# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────
async def run_all(drugs):

    # ✅ INIT BIGQUERY
    bq_client = get_bq_client()
    create_dataset_if_not_exists(bq_client)
    create_table_if_not_exists(bq_client)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DRUGS)

    results = {}
    failed = []

    tasks = [
        process_drug(i + 1, len(drugs), drug, semaphore, bq_client, results, failed)
        for i, drug in enumerate(drugs)
    ]

    await asyncio.gather(*tasks)

    print("\n======== SUMMARY ========")
    print(f"Success: {len(results)}")
    print(f"Failed : {len(failed)}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--drug")

    args = parser.parse_args()

    if args.drug:
        drugs = [args.drug]
    else:
        drugs = list_all_gcs_drugs()

    if args.limit:
        drugs = drugs[:args.limit]

    start = datetime.now()

    asyncio.run(run_all(drugs))

    print(f"\nTime taken: {(datetime.now() - start).total_seconds():.2f}s")
