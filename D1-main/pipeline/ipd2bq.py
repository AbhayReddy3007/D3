"""
Patent Legal Robustness Scorer
Sub-Factor 1 (40%): Novelty & Non-Obviousness of the Core Inventive Step
Sub-Factor 2 (30%): Obvious-to-Combine Risk
Sub-Factor 3 (20%): Prosecution History Vulnerability
Sub-Factor 4 (10%): Secondary Considerations (Evidence of Non-Obviousness)
+ Country-wise Weighted Score & Final Patent Score per drug

Input:  BigQuery table cognito_prod_datamart.Master_LOE
Output: BigQuery tables
          - patent_strength_table
          - patent_strength_country_score_table

Shortlists patents where Tag = 'Blocking' and Grant Date is not empty,
then queries AlloyDB for relevant chunks and uses Gemini to score each patent.
Uses the 'Source_File' column from the BQ table to directly match AlloyDB filenames.

fetch_relevant_chunks searches the drug-specific collection first, then falls back
to scanning ALL collections in AlloyDB if the file is not found in the primary one.
"""

import os
import re
import json
import time
import argparse
from datetime import datetime
import pandas as pd
# google.generativeai is deprecated. Use google.genai (new SDK) only.
from google import genai
from google.genai import types
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv(override=True)

# ── Import AlloyDB client ─────────────────────────────────────────────────────
# `alloydb_client.py` lives in `cog/` (a sibling package). Depending on how
# this script is deployed it may sit at any of:
#   <script_dir>/cog/alloydb_client.py            (running from pipeline/)
#   <script_dir>/../cog/alloydb_client.py         (container layout: /app/Pipeline/ + /app/cog/)
#   <script_dir>/alloydb_client.py                (rare — flat layout)
#
# We probe each candidate, add the first one that exists to sys.path, and
# fail with a clear, actionable message if none do. The previous version
# only checked one location, which produced a misleading
# `ModuleNotFoundError: No module named 'alloydb_client'` whenever the
# deployment layout differed.
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_alloydb_candidates = [
    os.path.join(SCRIPT_DIR, "cog"),                  # pipeline/cog/
    os.path.join(SCRIPT_DIR, os.pardir, "cog"),       # ../cog/
    SCRIPT_DIR,                                       # alongside this script
]
_alloydb_dir = next(
    (d for d in _alloydb_candidates
     if os.path.exists(os.path.join(d, "alloydb_client.py"))),
    None,
)
if _alloydb_dir is None:
    raise ModuleNotFoundError(
        "alloydb_client.py not found. Looked in:\n  "
        + "\n  ".join(os.path.abspath(d) for d in _alloydb_candidates)
        + "\nMake sure cog/alloydb_client.py is included in your container "
          "image and lives next to (or as a sibling of) this script."
    )
_alloydb_dir = os.path.abspath(_alloydb_dir)
if _alloydb_dir not in sys.path:
    sys.path.insert(0, _alloydb_dir)
print(f"[IPD2BQ] alloydb_client located at: {_alloydb_dir}/alloydb_client.py")

from alloydb_client import AlloyDBClient

# ── BigQuery Config (from .env) ───────────────────────────────────────────────
BQ_PROJECT_ID  = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID  = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_TABLE_ID    = os.getenv("BQ_LOE_TABLE_NAME", "Master_LOE")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
BQ_LOCATION    = os.getenv("BQ_LOCATION", "asia-south1")

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-flash"
API_KEY       = os.environ.get("GEMINI_API_KEY", "")

SCORE_COLORS = {
    1: "C6EFCE",  # Green  – Very Robust
    2: "92D050",  # Light green
    3: "FFEB9C",  # Yellow – Moderate
    4: "FFC7CE",  # Light red
    5: "FF0000",  # Red    – Exceptional Challenger Opportunity
}

# ── Country Weights ───────────────────────────────────────────────────────────
COUNTRY_WEIGHTS = {
    "US": 0.25,
    "EP": 0.25,
    "CN": 0.05,
    "CA": 0.05,
    "AU": 0.05,
    "KR": 0.05,
    "IN": 0.05,
    "BR": 0.05,
    "MX": 0.05,
    "TW": 0.05,
    "JP": 0.05,
    "RU": 0.05,
}

JURISDICTION_LABELS = {
    "US": "United States",
    "EP": "European Union (EP)",
    "CN": "China",
    "CA": "Canada",
    "AU": "Australia",
    "KR": "South Korea",
    "IN": "India",
    "BR": "Brazil",
    "MX": "Mexico",
    "TW": "Taiwan",
    "JP": "Japan",
    "RU": "Russia",
}

SF1_SCORE_LABELS = {
    1: "Truly Novel Concept",
    2: "Significant Modification in Known Technical Area",
    3: "Derivative but Non-Trivial",
    4: "Derivative of Known Class with Predictable Properties",
    5: "Minor Predictable Variation of Known Prior Art",
}

SF2_SCORE_LABELS = {
    1: "No Clear Motivation or Unexpected Result",
    2: "Limited Motivation Due to Technical Challenges",
    3: "Some Motivation but Outcome Uncertain",
    4: "Clear Motivation to Combine with Reasonable Expectation",
    5: "Combination Widely Known and Predictable",
}

SF3_SCORE_LABELS = {
    1: "Claims Granted Largely as Filed",
    2: "Minor Claim Amendments",
    3: "Moderate Amendments and Arguable Vulnerabilities",
    4: "Meaningful Claim Narrowing to Overcome Prior Art",
    5: "Significant Narrowing Amendments Creating Strong Estoppel Risk",
}

SF4_SCORE_LABELS = {
    1: "Clear Unexpected Results Supported by Data",
    2: "Strong Technical Improvement",
    3: "Moderate Improvement",
    4: "Weak or Unsupported Advantages",
    5: "No Evidence of Unexpected Results",
}

SF1_SECTIONS = [
    "independent claim", "claim 1", "background", "prior art",
    "summary of invention", "example", "field of invention",
]

SF2_SECTIONS = [
    "background", "state of the art", "references cited",
    "summary of invention", "technical problem", "prior art",
    "combination", "known", "disclosure", "independent claim",
    "claim 1", "motivation",
]

SF3_SECTIONS = [
    "independent claim", "claim 1", "claims", "amendment",
    "prosecution", "rejection", "office action", "response",
    "examiner", "prior art", "narrowing", "limitation",
    "background", "abstract",
]

SF4_SECTIONS = [
    "example", "experimental", "comparative", "data", "table",
    "result", "stability", "bioavailability", "efficacy",
    "toxicity", "improvement", "advantage", "unexpected",
    "synergy", "in vivo", "in vitro",
]

SF1_PROMPT = """
You are a pharmaceutical patent attorney evaluating the novelty and non-obviousness
of a patent's core inventive step.

Below are the most relevant excerpts from the patent document (independent claims,
background, prior art discussion, summary, and examples):

--- PATENT EXCERPTS ---
{chunks}
--- END EXCERPTS ---

Patent details:
- Drug: {drug}
- Patent Number: {patent_number}
- Filing Date: {filing_date}
- Grant Date: {grant_date}

Evaluate Sub-Factor 1: Novelty & Non-Obviousness of the Core Inventive Step (Weight: 40%).

Objective: Determine whether the claimed invention is structurally or conceptually close to prior art.

Evaluation Steps:
1. Parse the independent claim and identify the inventive element.
2. Summarize the core inventive step.
3. Extract cited prior art from the Background section.
4. Compare the invention with known approaches using the patent text and literature references.
5. Determine whether the modification appears incremental or novel.

Scoring rubric (assign exactly one integer score 1-5):

  Score 5 — Minor Predictable Variation of Known Prior Art:
    The claimed invention differs from prior art only by a routine modification that
    would normally be tried by a skilled person.
    Typical indicators:
    - Simple substitution (e.g., replacing one known excipient with another used for the same purpose)
    - Small parameter adjustment (e.g., pH range, dosage range, concentration)
    - Minor mechanical tweak to an existing device design
    - Same function achieved using the same mechanism
    Key signal: The modification is standard practice in the field.

  Score 4 — Derivative of Known Class with Predictable Properties:
    The invention belongs to a known class of technologies but introduces small modifications.
    Typical indicators:
    - New salt or polymorph of a known molecule
    - Use of a known stabilization technique in a closely related system
    - Device design derived from an existing platform
    Key signal: The modification is expected to work based on prior knowledge.

  Score 3 — Derivative but Non-Trivial:
    The invention is based on known technology but involves non-obvious adjustments or combinations.
    Typical indicators:
    - Combining technologies not previously used together
    - Optimization that required experimentation
    - Modification not directly suggested in prior art
    Key signal: The solution was not obvious but still builds on known technology.

  Score 2 — Significant Modification in Known Technical Area:
    The invention introduces a major technical change within an existing field.
    Typical indicators:
    - New mechanism for drug stabilization
    - New device architecture for the same therapy
    - Novel manufacturing approach not previously applied to this class
    Key signal: The approach is technically different but still within the same technological domain.

  Score 1 — Truly Novel Concept:
    The invention introduces a new technical principle or approach not previously seen in the prior art.
    Typical indicators:
    - No close structural or technical analogues
    - Prior art does not describe similar solutions
    - The invention solves a problem previously considered unsolved
    Key signal: The invention represents a new concept rather than an improvement of existing technology.

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "score": <integer 1-5>,
  "score_reason": "<1 sentence stating the primary reason this specific score was chosen over adjacent scores>",
  "reasoning": "<2-4 sentence justification referencing specific claims or background text>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"]
}}
"""

SF2_PROMPT = """
You are a pharmaceutical patent attorney evaluating the obvious-to-combine risk
of a patent — i.e., whether a person skilled in the art would have had a clear
motivation to combine the known elements with a reasonable expectation of success.

Below are the most relevant excerpts from the patent document (background, state of
the art, cited references, summary of invention, and technical problem description):

--- PATENT EXCERPTS ---
{chunks}
--- END EXCERPTS ---

Patent details:
- Drug: {drug}
- Patent Number: {patent_number}
- Filing Date: {filing_date}
- Grant Date: {grant_date}

Evaluate Sub-Factor 2: Obvious-to-Combine Risk (Weight: 30%).

Objective: Determine whether a person skilled in the art would have had a clear
motivation to combine known elements with a reasonable expectation of success.

Evaluation Steps:
1. Parse the independent claim and identify the elements being combined.
2. Determine whether the invention combines elements such as APIs, excipients,
   device components, manufacturing steps, or therapeutic strategies.
3. Identify the technical problem the patent claims to solve.
4. Review the Background section, Summary of the invention, and cited prior art.
5. Extract references mentioned in the Background section and reference list.
6. Assess whether prior patents or scientific literature already proposed similar combinations
   (e.g., [API name + formulation] or [technology + combination therapy]).
7. Assess whether a skilled person would have expected the combination to succeed.

Scoring rubric (assign exactly one integer score 1-5):

  Score 5 — Combination Widely Known and Predictable:
    The combination is explicitly discussed in prior patents or journal literature.
    Typical indicators:
    - Prior patents describing similar combinations
    - Scientific articles proposing the same strategy
    - Combination solves a widely recognized problem
    - Result appears purely additive
    Sources to confirm: Google Patents / Espacenet, PubMed / Google Scholar.
    Key signal: The field already expected this combination to work.

  Score 4 — Clear Motivation to Combine:
    Prior art strongly suggests combining the elements.
    Typical indicators:
    - Prior patents describe similar combinations
    - Scientific literature proposes the same approach
    - Known benefits from combining similar technologies
    Key signal: Combination was a logical next step in the field.

  Score 3 — Motivation Exists but Outcome Uncertain:
    Literature or patents suggest the combination but do not clearly demonstrate success.
    Typical indicators:
    - Limited experimental evidence in prior art
    - Combination discussed but not validated
    - Uncertain compatibility between components
    Key signal: Combination was plausible but not clearly predictable.

  Score 2 — Limited Motivation to Combine:
    Little prior art supports the combination.
    Typical indicators:
    - Few references describing similar approaches
    - Known incompatibilities or technical barriers
    - Limited discussion in scientific literature
    Key signal: Combination would not have been an obvious direction.

  Score 1 — No Clear Motivation or Unexpected Result:
    The combination appears entirely novel.
    Typical indicators:
    - No similar patents found in patent databases
    - No journal literature discussing the combination
    - Patent demonstrates unexpected synergy
    Key signal: The combination was not suggested by prior art.

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "score": <integer 1-5>,
  "score_reason": "<1 sentence stating the primary reason this specific score was chosen over adjacent scores>",
  "reasoning": "<2-4 sentence justification referencing specific background or cited references>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"]
}}
"""

SF3_PROMPT = """
You are a pharmaceutical patent attorney evaluating the prosecution history
vulnerability of a patent — i.e., whether the patent was granted only after
significant claim narrowing or limiting arguments that may weaken enforceability.

Below are the most relevant excerpts from the patent document (independent claims,
amendments, prosecution history references, examiner rejections, and applicant responses):

--- PATENT EXCERPTS ---
{chunks}
--- END EXCERPTS ---

Patent details:
- Drug: {drug}
- Patent Number: {patent_number}
- Filing Date: {filing_date}
- Grant Date: {grant_date}

Evaluate Sub-Factor 3: Prosecution History Vulnerability (Weight: 20%).

Objective: Determine whether the patent was granted only after significant claim
narrowing or limiting arguments, which may weaken enforceability.

Evaluation Steps:
1. Retrieve and review any prosecution history (file wrapper) information available in the excerpts.
2. Extract the originally filed independent claim (if discernible).
3. Extract the final granted independent claim.
4. Compare the two versions to identify narrowing amendments.
5. Review examiner rejections and cited prior art references.
6. Review applicant responses and arguments used to distinguish prior art.

Scoring rubric (assign exactly one integer score 1-5):

  Score 5 — Significant Narrowing Amendments:
    Broad claim scope was surrendered to obtain allowance, creating strong estoppel risk.
    Typical indicators:
    - Major structural limitations added
    - Specific excipients, ratios, or device features introduced
    - Narrowed chemical structures
    Key signal: Broad claim scope was surrendered to obtain allowance.

  Score 4 — Meaningful Claim Narrowing:
    Claims were modified to distinguish strong prior art, reducing scope during prosecution.
    Typical indicators:
    - Claims modified to distinguish strong prior art
    - Important technical limitations added
    Key signal: Scope reduced during prosecution.

  Score 3 — Moderate Amendments:
    Several amendments were made but they are not strongly limiting; some vulnerability exists.
    Typical indicators:
    - Several amendments but not strongly limiting
    Key signal: Some vulnerability exists.

  Score 2 — Minor Amendments:
    Only small clarifications or wording changes were made during prosecution.
    Typical indicators:
    - Small clarifications or wording changes
    Key signal: Prosecution history does not significantly weaken the patent.

  Score 1 — Claims Granted Largely as Filed:
    Few or no amendments were required; prosecution history is strong.
    Typical indicators:
    - Few or no amendments
    - Weak prior art cited
    Key signal: Strong prosecution history.

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "score": <integer 1-5>,
  "score_reason": "<1 sentence stating the primary reason this specific score was chosen over adjacent scores>",
  "reasoning": "<2-4 sentence justification referencing specific claim amendments or prosecution arguments>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"]
}}
"""

SF4_PROMPT = """
You are a pharmaceutical patent attorney evaluating secondary considerations
(objective evidence of non-obviousness) for a patent.

Below are the most relevant excerpts from the patent document (examples, experimental
data, comparative experiments, data tables, and claimed advantages):

--- PATENT EXCERPTS ---
{chunks}
--- END EXCERPTS ---

Patent details:
- Drug: {drug}
- Patent Number: {patent_number}
- Filing Date: {filing_date}
- Grant Date: {grant_date}

Evaluate Sub-Factor 4: Secondary Considerations — Evidence of Non-Obviousness (Weight: 10%).

Objective: Evaluate whether the patent provides objective evidence supporting
non-obviousness, such as unexpected results.

Evaluation Steps:
1. Review experimental data in the patent, focusing on:
   - Examples section
   - Data tables
   - Comparative experiments
2. Identify claimed advantages, such as:
   - Improved stability
   - Improved bioavailability
   - Enhanced efficacy
   - Reduced toxicity
3. Assess whether similar advantages were already known in scientific literature.
4. Assess whether similar results were already disclosed in patent databases.
5. Evaluate whether the improvement is truly unexpected.

Scoring rubric (assign exactly one integer score 1-5):

  Score 5 — No Evidence of Unexpected Results:
    No objective evidence supporting non-obviousness.
    Typical indicators:
    - No comparative data in the patent
    - Advantages asserted but not demonstrated
    - Similar results already known in literature
    Key signal: No objective evidence supporting non-obviousness.

  Score 4 — Weak or Unsupported Advantages:
    Advantages are weakly supported.
    Typical indicators:
    - Limited data
    - Benefits not clearly demonstrated
    Key signal: Advantages are weakly supported.

  Score 3 — Moderate Improvement:
    Improvement exists but may be predictable.
    Typical indicators:
    - Data shows improvement but similar effects already reported in literature
    Key signal: Improvement exists but may be predictable.

  Score 2 — Strong Technical Improvement:
    Significant improvement supported by experiments strengthens the patent.
    Typical indicators:
    - Significant improvement supported by experiments
    - Prior art does not show similar results
    Key signal: Improvement strengthens the patent.

  Score 1 — Clear Unexpected Results Supported by Data:
    Results strongly support non-obviousness.
    Typical indicators:
    - Dramatic improvement over prior art
    - Strong comparative data
    - Evidence of synergy
    Key signal: Results strongly support non-obviousness.

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "score": <integer 1-5>,
  "score_reason": "<1 sentence stating the primary reason this specific score was chosen over adjacent scores>",
  "reasoning": "<2-4 sentence justification referencing specific experimental data or claimed advantages>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"]
}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

_alloydb_client = AlloyDBClient()

def get_chroma_clients():
    """Return a list containing the AlloyDB client (backward-compatible name)."""
    return [_alloydb_client]

# Keep backward-compatible single-client helper
def get_chroma_client():
    return _alloydb_client

def collection_name(drug: str) -> str:
    return f"patents_{drug.strip().replace(' ', '_')}"

def infer_jurisdiction(patent_number: str) -> str:
    prefix_map = {
        "US": "US", "EP": "EP", "WO": "WO", "CN": "CN", "IN": "IN",
        "JP": "JP", "KR": "KR", "BR": "BR", "MX": "MX", "TW": "TW",
        "AU": "AU", "CA": "CA", "RU": "RU", "EA": "RU",
    }
    pn = str(patent_number).strip().upper()
    for prefix, label in prefix_map.items():
        if pn.startswith(prefix):
            return label
    return "UNKNOWN"


def get_country_weight(jurisdiction: str) -> float:
    return COUNTRY_WEIGHTS.get(jurisdiction.strip().upper(), 0.0)


# ── AlloyDB chunk fetcher (with all-collections fallback) ─────────────────────

def fetch_relevant_chunks(client, drug, patent_number, source_file, sections, top_k=12):
    """
    Fetch relevant chunks for a patent from AlloyDB.

    Strategy:
      1. Try the drug-specific collection (fast path): patents_{drug}
      2. If not found, scan ALL collections.

    Args:
        client:       AlloyDBClient instance (kept for API compat)
        drug:         Drug name string (used to build primary collection name)
        patent_number: Patent number string (used for logging only)
        source_file:  Exact filename to look up in metadata
        sections:     List of section keyword strings for relevance ranking
        top_k:        Maximum number of chunks to return

    Returns:
        List of document strings (chunks), ranked by section keyword relevance.
    """
    if not source_file.strip():
        print(f"\n    [WARN] No source file specified for '{patent_number}'")
        return []

    exact_filename = source_file.strip()

    def _query_collection(collection, filename):
        """Return all chunks from a collection whose metadata filename matches."""
        docs = []

        # Primary attempt: filter by both filename and valid chunk_index
        try:
            results = collection.get(
                where={"$and": [
                    {"filename": {"$eq": filename}},
                    {"chunk_index": {"$gte": 0}},
                ]},
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
        except Exception:
            pass

        # Fallback: drop chunk_index filter (handles older ingestion schemas)
        if not docs:
            try:
                results = collection.get(
                    where={"filename": {"$eq": filename}},
                    include=["documents", "metadatas"],
                )
                docs = [
                    d for d, m in zip(
                        results.get("documents", []),
                        results.get("metadatas", []),
                    )
                    if m.get("chunk_index", -1) >= 0
                ]
            except Exception:
                pass

        return docs

    def _rank_and_slice(docs):
        """Sort by section keyword relevance and return top_k."""
        def relevance_score(text):
            t = text.lower()
            return sum(1 for kw in sections if kw in t)
        return sorted(docs, key=relevance_score, reverse=True)[:top_k]

    # ── 1. Primary: drug-specific collection ─────────────────────────────────
    primary_coll_name = collection_name(drug)
    try:
        primary_coll = _alloydb_client.get_collection(primary_coll_name)
        docs = _query_collection(primary_coll, exact_filename)
        if docs:
            print(
                f"\n    [INFO] Found {len(docs)} chunks in collection "
                f"'{primary_coll_name}' for '{exact_filename}'",
                end=" ",
            )
            return _rank_and_slice(docs)
    except Exception:
        pass

    print(
        f"\n    [INFO] '{exact_filename}' not found in primary collection "
        f"'{primary_coll_name}' — scanning all collections...",
        end=" ",
    )

    # ── 2. Fallback: search every collection ─────────────────────────────────
    try:
        all_collections = _alloydb_client.list_collections()
    except Exception as e:
        print(f"\n    [ERROR] Cannot list collections: {e}")
        return []

    for coll_obj in all_collections:
        coll_name = coll_obj.name if hasattr(coll_obj, "name") else str(coll_obj)
        if coll_name == primary_coll_name:
            continue
        try:
            coll = _alloydb_client.get_collection(coll_name)
            docs = _query_collection(coll, exact_filename)
            if docs:
                print(
                    f"\n    [INFO] Found {len(docs)} chunks in fallback collection "
                    f"'{coll_name}' for '{exact_filename}'",
                    end=" ",
                )
                return _rank_and_slice(docs)
        except Exception as e:
            print(f"\n    [WARN] Could not search collection '{coll_name}': {e}")
            continue

    print(f"\n    [WARN] No chunks found for '{exact_filename}' in any collection")
    return []


# Lazy module-level client so we configure exactly once. The new SDK's
# Client is cheap to keep around and is thread-safe for the synchronous
# generate_content path we use here.
_genai_client = None

def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=API_KEY)
    return _genai_client


def call_gemini(prompt: str) -> dict:
    client = _get_genai_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    text = (response.text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT_FILE = "scoring_checkpoint111111.json"

def checkpoint_key(drug, patent_number):
    return f"{drug.strip()}|{patent_number.strip()}"

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        print(f"📂 Loaded checkpoint: {len(data)} patents already scored ({path})")
        return data
    return {}

def save_checkpoint(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


# ── Core Scoring ──────────────────────────────────────────────────────────────

import asyncio
import concurrent.futures

# Thread pool for running blocking Gemini calls concurrently
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Concurrency limit: how many patents scored at once
PATENT_CONCURRENCY = 4


def _run_gemini_score_sync(prompt, chunks, drug, patent_number, filing_date, grant_date):
    """Synchronous Gemini call — designed to run in a thread."""
    chunks_text = "\n\n---\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks))
    full_prompt = prompt.format(
        chunks=chunks_text, drug=drug, patent_number=patent_number,
        filing_date=filing_date, grant_date=grant_date,
    )
    try:
        result = call_gemini(full_prompt)
        result["chunks_used"] = len(chunks)
        return result
    except Exception as e:
        return {"score": None, "reasoning": f"Gemini error: {e}",
                "key_findings": [], "chunks_used": len(chunks)}


async def score_patent_async(client, row, cached, active_sfs):
    """
    Score a single patent — all 4 sub-factors run in PARALLEL via threads.
    """
    drug          = str(row.get("Drug Name", "")).strip()
    patent_number = str(row.get("Patent Number", "")).strip()
    filing_date   = str(row.get("Filing Date", "")).strip()
    grant_date    = str(row.get("Grant Date", "")).strip()
    source_file   = str(row.get("Source File", "")).strip()

    result = dict(cached)
    empty = {
        "score": None,
        "reasoning": "No patent chunks found in AlloyDB for this patent.",
        "key_findings": [],
        "chunks_used": 0,
    }

    SF_CONFIG = {
        "sf1": (SF1_PROMPT, SF1_SECTIONS),
        "sf2": (SF2_PROMPT, SF2_SECTIONS),
        "sf3": (SF3_PROMPT, SF3_SECTIONS),
        "sf4": (SF4_PROMPT, SF4_SECTIONS),
    }

    # Identify which sub-factors still need scoring
    sfs_to_score = [sf for sf in active_sfs if sf not in result]
    if not sfs_to_score:
        return result

    loop = asyncio.get_event_loop()

    # Launch all sub-factor Gemini calls in parallel
    async def _score_one_sf(sf_key):
        prompt_template, sections = SF_CONFIG[sf_key]
        chunks = fetch_relevant_chunks(client, drug, patent_number, source_file, sections)
        if not chunks:
            return sf_key, empty
        sf_result = await loop.run_in_executor(
            _executor,
            _run_gemini_score_sync,
            prompt_template, chunks, drug, patent_number, filing_date, grant_date,
        )
        return sf_key, sf_result

    sf_tasks = [_score_one_sf(sf) for sf in sfs_to_score]
    sf_results = await asyncio.gather(*sf_tasks)

    for sf_key, sf_result in sf_results:
        result[sf_key] = sf_result
        score = sf_result.get("score", "ERR")
        print(f"    [{sf_key.upper()}] score={score}", end="  ")

    # Also log cached ones
    for sf_key in active_sfs:
        if sf_key not in sfs_to_score:
            print(f"    [{sf_key.upper()}] cached ✓", end="  ")

    return result


# Keep sync wrapper for backward compat
def score_patent(client, row, cached, active_sfs):
    drug          = str(row.get("Drug Name", "")).strip()
    patent_number = str(row.get("Patent Number", "")).strip()
    filing_date   = str(row.get("Filing Date", "")).strip()
    grant_date    = str(row.get("Grant Date", "")).strip()
    source_file   = str(row.get("Source File", "")).strip()

    result = dict(cached)
    empty = {
        "score": None,
        "reasoning": "No patent chunks found in AlloyDB for this patent.",
        "key_findings": [],
        "chunks_used": 0,
    }

    SF_CONFIG = {
        "sf1": (SF1_PROMPT, SF1_SECTIONS),
        "sf2": (SF2_PROMPT, SF2_SECTIONS),
        "sf3": (SF3_PROMPT, SF3_SECTIONS),
        "sf4": (SF4_PROMPT, SF4_SECTIONS),
    }

    for sf_key in active_sfs:
        if sf_key in result:
            print(f"    [{sf_key.upper()}] cached ✓", end="  ")
            continue
        prompt_template, sections = SF_CONFIG[sf_key]
        chunks = fetch_relevant_chunks(client, drug, patent_number, source_file, sections)
        if chunks:
            result[sf_key] = _run_gemini_score_sync(
                prompt_template, chunks, drug, patent_number, filing_date, grant_date
            )
        else:
            result[sf_key] = empty
        score = result[sf_key].get("score", "ERR")
        print(f"    [{sf_key.upper()}] score={score}", end="  ")

    return result


# ── Final Summary Generation ──────────────────────────────────────────────────

FINAL_SUMMARY_PROMPT = """
You are a pharmaceutical patent attorney. Based on the sub-factor scores and reasoning
below, generate a concise summary for this patent.

Patent: {patent_number}
Drug: {drug}
Patent Type: {patent_type}

Sub-Factor Scores and Reasoning:
- SF1 (Novelty & Non-Obviousness, 40%): Score {sf1_score}/5 — {sf1_reasoning}
- SF2 (Obvious-to-Combine Risk, 30%): Score {sf2_score}/5 — {sf2_reasoning}
- SF3 (Prosecution History Vulnerability, 20%): Score {sf3_score}/5 — {sf3_reasoning}
- SF4 (Secondary Considerations, 10%): Score {sf4_score}/5 — {sf4_reasoning}
- Weighted Final Score: {weighted_score}/5

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "core_inventive_step": "<1 sentence summarizing the patent's core inventive step>",
  "key_vulnerabilities": "<1-2 sentences on the main legal weaknesses based on the scores>",
  "key_strengths": "<1-2 sentences on the main legal strengths based on the scores>"
}}
"""


def generate_final_summary(row, result):
    drug = str(row.get("Drug Name", "")).strip()
    patent_number = str(row.get("Patent Number", "")).strip()
    patent_type = str(row.get("Step 1 Claim Category", "N/A")).strip()

    sf1 = result.get("sf1", {})
    sf2 = result.get("sf2", {})
    sf3 = result.get("sf3", {})
    sf4 = result.get("sf4", {})

    scores = [sf1.get("score"), sf2.get("score"), sf3.get("score"), sf4.get("score")]
    weights = [0.4, 0.3, 0.2, 0.1]

    valid = [(s, w) for s, w in zip(scores, weights) if s is not None]
    weighted_score = round(sum(s * w for s, w in valid) / sum(w for _, w in valid), 2) if valid else "N/A"

    prompt = FINAL_SUMMARY_PROMPT.format(
        patent_number=patent_number, drug=drug, patent_type=patent_type,
        sf1_score=scores[0] or "N/A", sf1_reasoning=sf1.get("reasoning", "Not scored"),
        sf2_score=scores[1] or "N/A", sf2_reasoning=sf2.get("reasoning", "Not scored"),
        sf3_score=scores[2] or "N/A", sf3_reasoning=sf3.get("reasoning", "Not scored"),
        sf4_score=scores[3] or "N/A", sf4_reasoning=sf4.get("reasoning", "Not scored"),
        weighted_score=weighted_score,
    )
    try:
        return call_gemini(prompt)
    except Exception as e:
        return {
            "core_inventive_step": f"Error generating summary: {e}",
            "key_vulnerabilities": "",
            "key_strengths": "",
        }


# ── Country-Wise Scoring ──────────────────────────────────────────────────────

def compute_weighted_score(result: dict) -> float | None:
    sf1_score = result.get("sf1", {}).get("score")
    sf2_score = result.get("sf2", {}).get("score")
    sf3_score = result.get("sf3", {}).get("score")
    sf4_score = result.get("sf4", {}).get("score")

    scores_weights = [
        (sf1_score, 0.4), (sf2_score, 0.3),
        (sf3_score, 0.2), (sf4_score, 0.1),
    ]
    valid = [(s, w) for s, w in scores_weights if s is not None]
    if not valid:
        return None
    return round(sum(s * w for s, w in valid) / sum(w for _, w in valid), 2)


def compute_country_wise_scores(df_shortlist: pd.DataFrame, results: list[dict]) -> dict:
    drug_data = {}
    patent_lookup = {}

    for i, (_, row) in enumerate(df_shortlist.iterrows()):
        drug = str(row.get("Drug Name", "")).strip()
        pn = str(row.get("Patent Number", "")).strip()
        r = results[i]

        jurisdiction = infer_jurisdiction(pn)
        cw = get_country_weight(jurisdiction)
        ws = compute_weighted_score(r)

        patent_lookup[f"{drug}|{pn}"] = {
            "jurisdiction": jurisdiction,
            "country_weight": cw,
            "weighted_score": ws,
        }

        if drug not in drug_data:
            drug_data[drug] = {}
        if jurisdiction not in drug_data[drug]:
            drug_data[drug][jurisdiction] = []
        drug_data[drug][jurisdiction].append({"patent_number": pn, "weighted_score": ws})

    output = {"drug_details": {}, "patent_lookup": patent_lookup}

    for drug, jurisdictions in drug_data.items():
        drug_entry = {"jurisdictions": {}, "final_patent_score": 0.0}

        for jur, patents in jurisdictions.items():
            cw = get_country_weight(jur)
            valid_scores = [p["weighted_score"] for p in patents if p["weighted_score"] is not None]

            if valid_scores:
                avg_ws = round(sum(valid_scores) / len(valid_scores), 2)
            else:
                avg_ws = None

            country_weighted_score = round(avg_ws * cw, 4) if (avg_ws is not None and cw > 0) else 0.0

            drug_entry["jurisdictions"][jur] = {
                "label": JURISDICTION_LABELS.get(jur, jur),
                "country_weight": cw,
                "patents": patents,
                "patent_count": len(patents),
                "avg_weighted_score": avg_ws,
                "country_weighted_score": country_weighted_score,
            }

            drug_entry["final_patent_score"] += country_weighted_score

        drug_entry["final_patent_score"] = round(drug_entry["final_patent_score"], 4)
        output["drug_details"][drug] = drug_entry

    return output


# ── BigQuery Output ───────────────────────────────────────────────────────────

def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None  # Use ADC (Application Default Credentials)

def _bq_client():
    """Return an authenticated BigQuery client."""
    credentials = _get_credentials()
    return bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )


def _write_bq_table(rows: list[dict], table_id: str, schema: list) -> None:
    """Truncate-and-load rows into a BigQuery table."""
    client = _bq_client()
    full_table = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_APPEND",
    )
    df_out = pd.DataFrame(rows)
    df_out = df_out.drop_duplicates()
    job = client.load_table_from_dataframe(df_out, full_table, job_config=job_config)
    job.result()
    print(f"  ✅ {len(rows)} rows written to {full_table}")


# ── BQ schema definitions ─────────────────────────────────────────────────────

PATENT_STRENGTH_SCHEMA = [
    bigquery.SchemaField("drug_name",             "STRING"),
    bigquery.SchemaField("patent_number",          "STRING"),
    bigquery.SchemaField("patent_type",            "STRING"),
    bigquery.SchemaField("jurisdiction",           "STRING"),
    bigquery.SchemaField("country_weight",         "FLOAT64"),
    bigquery.SchemaField("core_inventive_step",    "STRING"),
    bigquery.SchemaField("sf1_score",              "INTEGER"),
    bigquery.SchemaField("sf1_label",              "STRING"),
    bigquery.SchemaField("sf1_score_reason",       "STRING"),
    bigquery.SchemaField("sf1_reasoning",          "STRING"),
    bigquery.SchemaField("sf1_key_finding_1",      "STRING"),
    bigquery.SchemaField("sf1_key_finding_2",      "STRING"),
    bigquery.SchemaField("sf1_key_finding_3",      "STRING"),
    bigquery.SchemaField("sf1_chunks_used",        "INTEGER"),
    bigquery.SchemaField("sf2_score",              "INTEGER"),
    bigquery.SchemaField("sf2_label",              "STRING"),
    bigquery.SchemaField("sf2_score_reason",       "STRING"),
    bigquery.SchemaField("sf2_reasoning",          "STRING"),
    bigquery.SchemaField("sf2_key_finding_1",      "STRING"),
    bigquery.SchemaField("sf2_key_finding_2",      "STRING"),
    bigquery.SchemaField("sf2_key_finding_3",      "STRING"),
    bigquery.SchemaField("sf2_chunks_used",        "INTEGER"),
    bigquery.SchemaField("sf3_score",              "INTEGER"),
    bigquery.SchemaField("sf3_label",              "STRING"),
    bigquery.SchemaField("sf3_score_reason",       "STRING"),
    bigquery.SchemaField("sf3_reasoning",          "STRING"),
    bigquery.SchemaField("sf3_key_finding_1",      "STRING"),
    bigquery.SchemaField("sf3_key_finding_2",      "STRING"),
    bigquery.SchemaField("sf3_key_finding_3",      "STRING"),
    bigquery.SchemaField("sf3_chunks_used",        "INTEGER"),
    bigquery.SchemaField("sf4_score",              "INTEGER"),
    bigquery.SchemaField("sf4_label",              "STRING"),
    bigquery.SchemaField("sf4_score_reason",       "STRING"),
    bigquery.SchemaField("sf4_reasoning",          "STRING"),
    bigquery.SchemaField("sf4_key_finding_1",      "STRING"),
    bigquery.SchemaField("sf4_key_finding_2",      "STRING"),
    bigquery.SchemaField("sf4_key_finding_3",      "STRING"),
    bigquery.SchemaField("sf4_chunks_used",        "INTEGER"),
    bigquery.SchemaField("weighted_score",         "FLOAT64"),
    bigquery.SchemaField("key_vulnerabilities",    "STRING"),
    bigquery.SchemaField("key_strengths",          "STRING"),
    bigquery.SchemaField("filing_date",            "STRING"),
    bigquery.SchemaField("grant_date",             "STRING"),
]

COUNTRY_SCORE_SCHEMA = [
    bigquery.SchemaField("drug_name",                  "STRING"),
    bigquery.SchemaField("jurisdiction",               "STRING"),
    bigquery.SchemaField("country_name",               "STRING"),
    bigquery.SchemaField("country_weight",             "FLOAT64"),
    bigquery.SchemaField("patent_count",               "INTEGER"),
    bigquery.SchemaField("avg_weighted_score",         "FLOAT64"),
    bigquery.SchemaField("country_weighted_score",     "FLOAT64"),
    bigquery.SchemaField("final_patent_score",         "FLOAT64"),
]


def _sf_fields(result: dict, sf_key: str, score_labels: dict) -> dict:
    """Extract all sub-factor fields for a given sf_key into a flat dict."""
    sf = result.get(sf_key, {})
    score = sf.get("score")
    findings = sf.get("key_findings", [])
    return {
        f"{sf_key}_score":         score,
        f"{sf_key}_label":         score_labels.get(score, "") if score else "",
        f"{sf_key}_score_reason":  sf.get("score_reason", ""),
        f"{sf_key}_reasoning":     sf.get("reasoning", ""),
        f"{sf_key}_key_finding_1": findings[0] if len(findings) > 0 else "",
        f"{sf_key}_key_finding_2": findings[1] if len(findings) > 1 else "",
        f"{sf_key}_key_finding_3": findings[2] if len(findings) > 2 else "",
        f"{sf_key}_chunks_used":   sf.get("chunks_used", 0),
    }


def build_output_bigquery(df_shortlist, results, country_scores=None):
    """Write patent_strength_table and patent_strength_country_score_table to BigQuery."""

    # ── 1. patent_strength_table ──────────────────────────────────────────────
    print("\nGenerating final summaries and building patent_strength_table rows...")
    strength_rows = []
    for i, (_, row_data) in enumerate(df_shortlist.iterrows()):
        r = results[i]
        drug          = str(row_data.get("Drug Name", "")).strip()
        patent_number = str(row_data.get("Patent Number", "")).strip()
        patent_type   = str(row_data.get("Step 1 Claim Category", "")).strip()
        jurisdiction  = infer_jurisdiction(patent_number)
        cw            = get_country_weight(jurisdiction)
        weighted      = compute_weighted_score(r)

        print(f"  [{i+1}/{len(df_shortlist)}] {patent_number} — generating summary...", end=" ")
        summary = generate_final_summary(row_data, r)
        print("done")

        row_out = {
            "drug_name":          drug,
            "patent_number":      patent_number,
            "patent_type":        patent_type,
            "jurisdiction":       jurisdiction,
            "country_weight":     cw,
            "core_inventive_step": summary.get("core_inventive_step", ""),
            "weighted_score":     weighted,
            "key_vulnerabilities": summary.get("key_vulnerabilities", ""),
            "key_strengths":      summary.get("key_strengths", ""),
            "filing_date":        str(row_data.get("Filing Date", "")),
            "grant_date":         str(row_data.get("Grant Date", "")),
        }
        row_out.update(_sf_fields(r, "sf1", SF1_SCORE_LABELS))
        row_out.update(_sf_fields(r, "sf2", SF2_SCORE_LABELS))
        row_out.update(_sf_fields(r, "sf3", SF3_SCORE_LABELS))
        row_out.update(_sf_fields(r, "sf4", SF4_SCORE_LABELS))
        strength_rows.append(row_out)

    print("\nWriting patent_strength_table → BigQuery...")
    _write_bq_table(strength_rows, "patent_strength_table", PATENT_STRENGTH_SCHEMA)

    # ── 2. patent_strength_country_score_table ────────────────────────────────
    if country_scores:
        print("\nBuilding patent_strength_country_score_table rows...")
        country_rows = []
        for drug, drug_entry in country_scores["drug_details"].items():
            final_ps = drug_entry["final_patent_score"]
            for jur, jdata in sorted(
                drug_entry["jurisdictions"].items(),
                key=lambda x: (-x[1]["country_weight"], x[0])
            ):
                country_rows.append({
                    "drug_name":              drug,
                    "jurisdiction":           jur,
                    "country_name":           JURISDICTION_LABELS.get(jur, jur),
                    "country_weight":         jdata["country_weight"],
                    "patent_count":           jdata["patent_count"],
                    "avg_weighted_score":     jdata["avg_weighted_score"],
                    "country_weighted_score": round(jdata["country_weighted_score"], 4),
                    "final_patent_score":     round(final_ps, 4),
                })

        print("Writing patent_strength_country_score_table → BigQuery...")
        _write_bq_table(country_rows, "patent_strength_country_score_table", COUNTRY_SCORE_SCHEMA)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_from_bigquery() -> pd.DataFrame:
    """Load patent data from BigQuery Master_LOE table."""
    print(f"Connecting to BigQuery: {BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}")

    credentials = _get_credentials()
    client = bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )

    query = f"""
        SELECT
            Drug_Name,
            Patent_Number,
            Jurisdiction,
            Tag,
            Blocking_Category,
            Reason,
            Step_1_Claim_Category,
            Step_2_Matched_Elements,
            S2_Active_Ingredient__Form,
            S2_Formulation_Details,
            S2_Route_of_Administration,
            S2_Device_Description,
            S2_Combination_TechProcess,
            Step_3_Technical_Barrier,
            Step_3_Confidence,
            Step_3_Evidence_Type,
            Step_3_Evidence_Summary,
            Step_4_Blocking_Indicator,
            Step_4_Confidence,
            Step_4_Regulatory_Failure_if_Removed,
            Step_4_Bridging_Studies_Required,
            Step_4_Formulation_Consistent_Across_Phases,
            Step_4_Reason,
            Step_5_Novel__Difficult,
            Step_5_Novelty_Signal,
            Step_5_FirstinClass,
            Step_5_Prior_Failed_Attempts,
            Step_5_Complex_Implementation,
            Step_5_Confidence,
            Step_5_Reason,
            Filing_Date,
            Grant_Date,
            PTE_months,
            Pediatric_Exclusivity,
            Phase,
            Launch_Date,
            Approval_Date,
            Approval_Date_Source,
            Est_Approval_Year,
            Exclusivity_Year,
            Controlling_Patent_Expiry_Year,
            Years_to_Entry,
            Avg_Years_to_Entry,
            Score,
            Avg_Years_to_Entry_US__EP,
            IP_Dimension_1_Score,
            Source_File,
            Type,
            No_Of_Forecasted_Patents
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`
    """

    print("Executing BigQuery query...")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} total rows from BigQuery.")

    # Normalise column names to match the rest of the script (spaces, title-case)
    # BQ column names use underscores; map them to the human-readable names the
    # rest of the script expects (e.g. "Drug_Name" → "Drug Name").
    col_rename = {
        "Drug_Name": "Drug Name",
        "Patent_Number": "Patent Number",
        "Filing_Date": "Filing Date",
        "Grant_Date": "Grant Date",
        "Source_File": "Source File",
        "Blocking_Category": "Blocking Category",
        "Step_1_Claim_Category": "Step 1 Claim Category",
        "Step_2_Matched_Elements": "Step 2 Matched Elements",
        "S2_Active_Ingredient__Form": "S2 Active Ingredient Form",
        "S2_Formulation_Details": "S2 Formulation Details",
        "S2_Route_of_Administration": "S2 Route of Administration",
        "S2_Device_Description": "S2 Device Description",
        "S2_Combination_TechProcess": "S2 Combination TechProcess",
        "Step_3_Technical_Barrier": "Step 3 Technical Barrier",
        "Step_3_Confidence": "Step 3 Confidence",
        "Step_3_Evidence_Type": "Step 3 Evidence Type",
        "Step_3_Evidence_Summary": "Step 3 Evidence Summary",
        "Step_4_Blocking_Indicator": "Step 4 Blocking Indicator",
        "Step_4_Confidence": "Step 4 Confidence",
        "Step_4_Regulatory_Failure_if_Removed": "Step 4 Regulatory Failure if Removed",
        "Step_4_Bridging_Studies_Required": "Step 4 Bridging Studies Required",
        "Step_4_Formulation_Consistent_Across_Phases": "Step 4 Formulation Consistent Across Phases",
        "Step_4_Reason": "Step 4 Reason",
        "Step_5_Novel__Difficult": "Step 5 Novel Difficult",
        "Step_5_Novelty_Signal": "Step 5 Novelty Signal",
        "Step_5_FirstinClass": "Step 5 First in Class",
        "Step_5_Prior_Failed_Attempts": "Step 5 Prior Failed Attempts",
        "Step_5_Complex_Implementation": "Step 5 Complex Implementation",
        "Step_5_Confidence": "Step 5 Confidence",
        "Step_5_Reason": "Step 5 Reason",
        "PTE_months": "PTE months",
        "Pediatric_Exclusivity": "Pediatric Exclusivity",
        "Launch_Date": "Launch Date",
        "Approval_Date": "Approval Date",
        "Approval_Date_Source": "Approval Date Source",
        "Est_Approval_Year": "Est Approval Year",
        "Exclusivity_Year": "Exclusivity Year",
        "Controlling_Patent_Expiry_Year": "Controlling Patent Expiry Year",
        "Years_to_Entry": "Years to Entry",
        "Avg_Years_to_Entry": "Avg Years to Entry",
        "Avg_Years_to_Entry_US__EP": "Avg Years to Entry US EP",
        "IP_Dimension_1_Score": "IP Dimension 1 Score",
        "No_Of_Forecasted_Patents": "No Of Forecasted Patents",
    }
    df = df.rename(columns=col_rename)
    return df


def main():
    parser = argparse.ArgumentParser(description="Score patents on legal robustness sub-factors")
    parser.add_argument("drug", nargs="?", default=None,
                        help="Optional: single drug name to process (e.g. Semaglutide). Omit to process all drugs.")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between Gemini calls (default 1.5)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_FILE,
                        help=f"Checkpoint file path (default: {DEFAULT_CHECKPOINT_FILE})")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore existing checkpoint and re-score everything")
    args = parser.parse_args()

    if not API_KEY:
        raise EnvironmentError("Set GEMINI_API_KEY environment variable before running.")

    # ── 1. Load & shortlist from BigQuery ────────────────────────────────────
    df = load_from_bigquery()
    df.columns = [c.strip() for c in df.columns]

    if "Source File" not in df.columns:
        raise ValueError(
            "BigQuery table is missing a 'Source_File' column. "
            "Please ensure the column exists in the Master_LOE table."
        )

    mask_blocking = df["Tag"].astype(str).str.strip().str.lower() == "blocking"
    mask_granted = df["Grant Date"].notna() & (df["Grant Date"].astype(str).str.strip() != "")
    mask_not_forecasted = df["Type"].astype(str).str.strip().str.lower() != "forecasted"
    df_shortlist = df[mask_blocking & mask_granted & mask_not_forecasted].drop_duplicates().reset_index(drop=True)
    print(f"Shortlisted {len(df_shortlist)} patents from BigQuery (Tag=Blocking AND Grant Date present AND Type\u2260Forecasted).")

    if args.drug:
        df_shortlist = df_shortlist[
            df_shortlist["Drug Name"].str.lower() == args.drug.lower()
        ].reset_index(drop=True)
        print(f"[Filter] Filtered to drug '{args.drug}': {len(df_shortlist)} patents remaining.")
        if df_shortlist.empty:
            print("No patents match this drug name. Exiting.")
            return

    if df_shortlist.empty:
        print("No patents match the shortlist criteria. Exiting.")
        return

    missing_sf = df_shortlist[df_shortlist["Source File"].astype(str).str.strip() == ""]
    if not missing_sf.empty:
        print(f"\n\u26a0\ufe0f  WARNING: {len(missing_sf)} shortlisted patent(s) have no Source File value:")
        for _, row in missing_sf.iterrows():
            print(f"   - {row.get('Patent Number', 'N/A')} ({row.get('Drug Name', 'N/A')})")
        print("   These patents will return empty chunks and score as N/A.\n")

    # ── 2. Load checkpoint ────────────────────────────────────────────────────
    checkpoint_path = args.checkpoint
    cache = {} if args.reset else load_checkpoint(checkpoint_path)
    active_sfs = ["sf1", "sf2", "sf3", "sf4"]

    # ── 3. Connect to AlloyDB ─────────────────────────────────────────────────
    client = get_chroma_client()
    print(f"Connected to AlloyDB\n")

    # ── 4. Score patents concurrently (4 at a time, 4 sub-factors in parallel) ─
    checkpoint_path = args.checkpoint

    async def _score_all():
        semaphore = asyncio.Semaphore(PATENT_CONCURRENCY)
        results_indexed = {}

        async def _process_one(idx, row_data):
            pn = str(row_data.get("Patent Number", "N/A")).strip()
            drug = str(row_data.get("Drug Name", "N/A")).strip()
            sf = str(row_data.get("Source File", "")).strip()
            key = checkpoint_key(drug, pn)
            cached_entry = cache.get(key, {})
            needs_scoring = any(sf_key not in cached_entry for sf_key in active_sfs)

            print(f"[{idx+1}/{len(df_shortlist)}] {pn} ({drug}) — source: '{sf}'")

            if not needs_scoring:
                print(f"    All sub-factors cached — skipping Gemini calls ✓")
                results_indexed[idx] = cached_entry
                return

            async with semaphore:
                result = await score_patent_async(client, row_data, cached_entry, active_sfs)
                print()

            cache[key] = result
            save_checkpoint(cache, checkpoint_path)
            results_indexed[idx] = result

        tasks = []
        for i, (_, row) in enumerate(df_shortlist.iterrows()):
            tasks.append(_process_one(i, row))

        await asyncio.gather(*tasks)

        # Return results in original order
        return [results_indexed[i] for i in range(len(df_shortlist))]

    results = asyncio.run(_score_all())

    # ── 5. Compute country-wise scores ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing country-wise weighted scores...")
    print(f"{'='*60}")
    country_scores = compute_country_wise_scores(df_shortlist, results)

    for drug, drug_entry in country_scores["drug_details"].items():
        print(f"\n  Drug: {drug}")
        print(f"  {'Jurisdiction':<12} {'Country':<26} {'Weight':<8} {'#Pat':<6} {'Avg WS':<10} {'Country WS'}")
        print(f"  {'-'*80}")
        for jur, jdata in sorted(drug_entry["jurisdictions"].items(),
                                  key=lambda x: (-x[1]["country_weight"], x[0])):
            avg_ws = jdata["avg_weighted_score"]
            avg_ws_str = f"{avg_ws:.2f}" if avg_ws is not None else "N/A"
            print(f"  {jur:<12} {jdata['label']:<26} {jdata['country_weight']:<8} "
                  f"{jdata['patent_count']:<6} {avg_ws_str:<10} {jdata['country_weighted_score']:.4f}")
        print(f"  {chr(9472)*80}")
        print(f"  FINAL PATENT SCORE: {drug_entry['final_patent_score']:.4f}")

    # ── 6. Write output to BigQuery ───────────────────────────────────────────
    build_output_bigquery(df_shortlist, results, country_scores)

    # ── 7. Console summary ────────────────────────────────────────────────────
    for sf_key, sf_label, score_labels in [
        ("sf1", "SF1 (Novelty & Non-Obviousness)", SF1_SCORE_LABELS),
        ("sf2", "SF2 (Obvious-to-Combine Risk)", SF2_SCORE_LABELS),
        ("sf3", "SF3 (Prosecution History Vulnerability)", SF3_SCORE_LABELS),
        ("sf4", "SF4 (Secondary Considerations)", SF4_SCORE_LABELS),
    ]:
        scored = [r[sf_key]["score"] for r in results if r.get(sf_key, {}).get("score") is not None]
        if scored:
            print(f"\n{sf_label}:")
            print(f"  Patents scored : {len(scored)} / {len(df_shortlist)}")
            print(f"  Average score  : {sum(scored)/len(scored):.2f}")
            for s in range(1, 6):
                print(f"  Score {s} ({score_labels[s][:45]:<45}): {scored.count(s)}")

    print(f"\n{'='*60}")
    print("FINAL PATENT SCORES (Country-Weighted)")
    print(f"{'='*60}")
    for drug, drug_entry in country_scores["drug_details"].items():
        print(f"  {drug:<30} \u2192 {drug_entry['final_patent_score']:.4f}")

    print(f"\n{'='*60}")
    print("BigQuery output complete:")
    print(f"  \u2192 {BQ_PROJECT_ID}.{BQ_DATASET_ID}.patent_strength_table")
    print(f"  \u2192 {BQ_PROJECT_ID}.{BQ_DATASET_ID}.patent_strength_country_score_table")
    print(f"{'='*60}")
if __name__ == "__main__":
    main()
