"""
blocking_analyser.py
─────────────────────
Multi-step blocking analysis pipeline.

Step 1 (implemented):
  Classify the patent's primary claim into one of 7 categories.
  If Composition of Matter → BLOCKING immediately.
  Otherwise → Step 2.

Step 2 (implemented):
  Read the drug's real-world formulation data from a local Excel file.
  The Excel has one row per trial/source, multiple rows per drug.
  Check if the patent's claims cover any of these 5 elements across all rows:
    - Active Ingredient & Form
    - Formulation Details
    - Route of Administration
    - Device Description
    - Combination Tech/Process
  If ANY element is present in ANY row → continue to Step 3.
  If NONE are present across ALL rows → NON-BLOCKING.

  Excel is loaded once at startup and cached in memory.
  Drug rows are matched by Molecule column (case-insensitive).

Steps 3–5: pending implementation.
"""

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from google.genai import types

from .indexer import (
    gemini_client,
    generate_embeddings,
    get_dates_from_chromadb,
)

# ── EMA EPAR helpers (inlined from ema_epar_extractor to avoid import conflicts) ──

import io
from urllib.parse import urljoin
from bs4 import BeautifulSoup

_EMA_BASE_URL    = "https://www.ema.europa.eu"
_EMA_EXCEL_PAGE  = "https://www.ema.europa.eu/en/medicines/download-medicine-data"
_EMA_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
_EMPTY_EPAR = {
    "overview_pdf"                    : None,
    "public_summary_pdf"              : None,
    "risk_management_plan_summary_pdf": None,
    "product_information_pdf"         : None,
}
_ema_df_cache: Optional[pd.DataFrame] = None


def _load_ema_excel() -> Optional[pd.DataFrame]:
    global _ema_df_cache
    if _ema_df_cache is not None:
        return _ema_df_cache
    try:
        page_resp = requests.get(_EMA_EXCEL_PAGE, headers=_EMA_HTTP_HEADERS, timeout=60)
        if page_resp.status_code == 404:
            return None
        page_resp.raise_for_status()
    except Exception as e:
        print(f"[EMA EXCEL] Failed to fetch page: {e}")
        return None

    soup   = BeautifulSoup(page_resp.text, "html.parser")
    anchor = None
    for a in soup.find_all("a"):
        if "download medicines data table" in (a.get_text(strip=True) or "").lower():
            anchor = a
            break
    if anchor is None:
        anchor = soup.find("a", href=re.compile(
            r"/en/documents/report/medicines-output-medicines-report_en\.xlsx$", re.I
        ))
    if not anchor or not anchor.get("href"):
        print("[EMA EXCEL] Could not find download link")
        return None

    try:
        xls_resp = requests.get(
            urljoin(_EMA_BASE_URL, anchor["href"]),
            headers=_EMA_HTTP_HEADERS, timeout=120,
        )
        xls_resp.raise_for_status()
    except Exception as e:
        print(f"[EMA EXCEL] Download failed: {e}")
        return None

    df = pd.read_excel(io.BytesIO(xls_resp.content), engine="openpyxl")
    df = df.dropna(how="all").reset_index(drop=True)
    header_row_idx = None
    for i, row in df.iterrows():
        if "Name of medicine" in row.values:
            header_row_idx = i
            break
    if header_row_idx is None:
        print("[EMA EXCEL] Could not find header row")
        return None
    df.columns = df.iloc[header_row_idx]
    df         = df[header_row_idx + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    _ema_df_cache = df
    print(f"[EMA EXCEL] Loaded and cached ({len(df)} medicines)")
    return df


def _resolve_all_ema_brands(generic_name: str) -> List[str]:
    """Resolve INN/generic name to all EMA brand names via INN column + FDA fallback."""
    df = _load_ema_excel()
    seen: set = set()
    brands: List[str] = []

    if df is not None:
        inn_col = next(
            (c for c in df.columns if any(x in str(c).lower()
             for x in ["international non-proprietary", "inn", "common name"])),
            None
        )
        if inn_col:
            target = generic_name.lower().strip()
            mask   = df[inn_col].astype(str).str.lower().str.strip() == target
            if not mask.any():
                mask = df[inn_col].astype(str).str.lower().str.contains(target, na=False, regex=False)
            for brand in df.loc[mask, "Name of medicine"].astype(str).str.strip():
                if brand.lower() not in seen and brand.lower() not in ("nan", ""):
                    seen.add(brand.lower())
                    brands.append(brand)

    # FDA fallback
    try:
        r = requests.get(
            _OPEN_FDA_BASE,
            params={"search": f'products.active_ingredients.name:"{generic_name}"', "limit": 5},
            timeout=20, headers=_EMA_HTTP_HEADERS,
        )
        if r.status_code == 200:
            fda_brands = [
                p.get("brand_name", "").strip().lower()
                for res in r.json().get("results", []) or []
                for p in res.get("products", []) or []
                if p.get("brand_name")
            ]
            if df is not None and "Name of medicine" in df.columns:
                ema_names = df["Name of medicine"].astype(str).str.lower().str.strip()
                for fb in fda_brands:
                    for matched in df.loc[ema_names == fb, "Name of medicine"].astype(str).str.strip():
                        if matched.lower() not in seen and matched.lower() not in ("nan", ""):
                            seen.add(matched.lower())
                            brands.append(matched)
    except Exception as e:
        print(f"[EMA BRANDS] FDA fallback failed: {e}")

    if not brands:
        print(f"[EMA BRANDS] No brands found for '{generic_name}' — trying name directly")
        brands = [generic_name]
    else:
        print(f"[EMA BRANDS] '{generic_name}' → {brands}")
    return brands


def _get_ema_epar_documents(drug_name: str) -> dict:
    """Fetch EPAR PDF links for a given EMA brand name."""
    df = _load_ema_excel()
    if df is None or "Name of medicine" not in df.columns or "Medicine URL" not in df.columns:
        return _EMPTY_EPAR.copy()

    result = df.loc[
        df["Name of medicine"].astype(str).str.lower().str.strip() == drug_name.lower().strip(),
        "Medicine URL"
    ]
    if result.empty:
        return _EMPTY_EPAR.copy()

    drug_page_url = str(result.iloc[0]).strip()
    try:
        resp = requests.get(drug_page_url, headers=_EMA_HTTP_HEADERS, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[EMA EPAR] Failed to fetch page for '{drug_name}': {e}")
        return _EMPTY_EPAR.copy()

    soup     = BeautifulSoup(resp.text, "html.parser")
    patterns = {
        "overview_pdf"                    : re.compile(r"/en/documents/.*epar-medicine-overview.*_en\.pdf",   re.I),
        "public_summary_pdf"              : re.compile(r"/en/documents/.*epar-summary.*_en\.pdf",             re.I),
        "risk_management_plan_summary_pdf": re.compile(r"/en/documents/.*epar-risk-management.*_en\.pdf",     re.I),
        "product_information_pdf"         : re.compile(r"/en/documents/.*epar-product-information.*_en\.pdf", re.I),
    }
    found = _EMPTY_EPAR.copy()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/en/documents/" not in href or not href.lower().endswith(".pdf"):
            continue
        for key, pattern in patterns.items():
            if found[key] is None and pattern.search(href):
                found[key] = urljoin(_EMA_BASE_URL, href)
        if all(found.values()):
            break
    return found

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

_ANALYSIS_CONCURRENCY = asyncio.Semaphore(5)
_MAX_ANALYSIS_RETRIES = 3

# All 7 claim categories used across all steps
CLAIM_CATEGORIES = [
    "Composition of Matter",
    "Salt/Polymorph",
    "Formulation",
    "Manufacturing Process",
    "Method of Treatment",
    "Device",
    "Dosage Regimen",
]

# The 5 Step 2 columns we care about from the Excel
_STEP2_COLUMNS = [
    "Active Ingredient & Form",
    "Formulation Details",
    "Route of Administration",
    "Device Description",
    "Combination Tech/Process",
]

# ─────────────────────────────────────────────
# Excel loader (loaded once, cached in memory)
# ─────────────────────────────────────────────

_FORMULATION_DF: Optional[pd.DataFrame] = None


def load_formulation_excel(path: Optional[str] = None) -> None:
    """
    Load the formulation Excel file into memory once at startup.
    Call this from agent.py before running any analysis.

    Args:
        path: Absolute path to the Excel file.
              Falls back to the FORMULATION_EXCEL_PATH environment variable.
    """
    global _FORMULATION_DF

    excel_path = path or os.environ.get("FORMULATION_EXCEL_PATH", "")
    if not excel_path:
        print("[FORMULATION EXCEL] WARNING: No path provided and "
              "FORMULATION_EXCEL_PATH not set — Step 2 will run without Excel data.")
        return

    try:
        _FORMULATION_DF = pd.read_excel(excel_path, dtype=str)
        # Normalise column names: strip whitespace
        _FORMULATION_DF.columns = _FORMULATION_DF.columns.str.strip()
        print(f"[FORMULATION EXCEL] Loaded {len(_FORMULATION_DF)} rows from: {excel_path}")
        print(f"[FORMULATION EXCEL] Columns: {list(_FORMULATION_DF.columns)}")
    except Exception as e:
        print(f"[FORMULATION EXCEL] ERROR loading '{excel_path}': {e}")
        _FORMULATION_DF = None


def get_drug_rows(drug_name: str) -> List[Dict]:
    """
    Return all rows from the cached Excel where Molecule matches drug_name
    (case-insensitive, strips whitespace).

    Only returns the 5 Step 2 columns plus Trial ID and Phase for context.
    Empty / NaN cells are dropped from each row dict.

    Returns an empty list if the Excel was not loaded or no rows match.
    """
    if _FORMULATION_DF is None:
        return []

    if "Molecule" not in _FORMULATION_DF.columns:
        print("[FORMULATION EXCEL] WARNING: 'Molecule' column not found in Excel.")
        return []

    mask = _FORMULATION_DF["Molecule"].str.strip().str.lower() == drug_name.strip().lower()
    matched = _FORMULATION_DF[mask]

    if matched.empty:
        print(f"[FORMULATION EXCEL] No rows found for drug: '{drug_name}'")
        return []

    print(f"[FORMULATION EXCEL] Found {len(matched)} row(s) for '{drug_name}'")

    # Keep context columns + the 5 Step 2 columns
    keep_cols = ["Trial ID", "Phase"] + _STEP2_COLUMNS
    available = [c for c in keep_cols if c in matched.columns]
    subset = matched[available]

    rows = []
    for _, row in subset.iterrows():
        # Drop empty / NaN cells from each row
        row_dict = {
            k: v for k, v in row.items()
            if pd.notna(v) and str(v).strip().lower() not in ("", "nan", "none", "n/a")
        }
        if row_dict:
            rows.append(row_dict)

    return rows


# ─────────────────────────────────────────────
# Jurisdiction helpers
# ─────────────────────────────────────────────

def extract_jurisdiction(patent_number_hint: str) -> str:
    match = re.search(r"\b([A-Z]{2})\d", patent_number_hint.upper())
    if match:
        return match.group(1)
    fallback = re.match(r"^([A-Z]{2})", patent_number_hint.upper())
    return fallback.group(1) if fallback else "UN"


def is_non_analysable_patent(filename: str) -> bool:
    """Returns True only for patents that cannot be analysed (e.g. unknown jurisdiction)."""
    jurisdiction = extract_jurisdiction(Path(filename).stem)
    return jurisdiction == "UN"  # only skip truly unknown jurisdictions


# ─────────────────────────────────────────────
# Standardised result helpers
# ─────────────────────────────────────────────

def error_result(filename: str) -> Dict:
    stem = Path(filename).stem
    return {
        "patent_number":                  stem,
        "jurisdiction":                   extract_jurisdiction(stem),
        "filing_date":                    None,
        "grant_date":                     None,
        "claim_category":                 None,
        "tag":                            None,
        "blocking_category":              None,
        "reason":                         None,
        "pte":                            None,
        "pediatric_exclusivity":          False,
        "estimated_approval_year":        None,
        "exclusivity_year":               None,
        "controlling_patent_expiry_year": None,
        "years_to_entry":                 None,
        "avg_years_to_entry":             None,
        "score":                          None,
        "approval_date_us":               None,
        "approval_date_eu":               None,
        "approval_date_us_source":        None,
        "approval_date_eu_source":        None,
        "source_file":                    filename,
    }


def skipped_result(filename: str) -> Dict:
    stem         = Path(filename).stem
    jurisdiction = extract_jurisdiction(stem)
    return {
        "patent_number":                  stem,
        "jurisdiction":                   jurisdiction,
        "filing_date":                    None,
        "grant_date":                     None,
        "claim_category":                 None,
        "tag":                            "SKIPPED",
        "blocking_category":              None,
        "reason":                         f"{jurisdiction} patent — indexed for future use, not analysed.",
        "pte":                            None,
        "pediatric_exclusivity":          False,
        "estimated_approval_year":        None,
        "exclusivity_year":               None,
        "controlling_patent_expiry_year": None,
        "years_to_entry":                 None,
        "avg_years_to_entry":             None,
        "score":                          None,
        "approval_date_us":               None,
        "approval_date_eu":               None,
        "approval_date_us_source":        None,
        "approval_date_eu_source":        None,
        "source_file":                    filename,
    }


# ─────────────────────────────────────────────
# RAG retrieval
# ─────────────────────────────────────────────

async def rag_query(
    query: str, collection, filename: str, top_k: int = 6
) -> List[str]:
    emb = await generate_embeddings([query])
    if not emb:
        return []
    try:
        results = collection.query(
            query_embeddings=[emb[0]],
            n_results=top_k,
            where={
                "$and": [
                    {"filename":    {"$eq": filename}},
                    {"chunk_index": {"$gte": 0}},
                ]
            },
            include=["documents", "metadatas", "distances"],
        )
        return results["documents"][0]
    except Exception as e:
        print(f"[RAG] Query failed: {e}")
        return []


def get_all_chunks(collection, filename: str) -> List[str]:
    """
    Returns ALL chunks for a patent in document order (by chunk_index).
    This ensures the full patent text — abstract, description, examples,
    AND claims — is available for analysis, with no sections omitted.
    """
    try:
        results = collection.get(
            where={
                "$and": [
                    {"filename":    {"$eq": filename}},
                    {"chunk_index": {"$gte": 0}},
                ]
            },
            include=["documents", "metadatas"],
        )
        combined = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1].get("chunk_index", 0),
        )
        chunks = [doc for doc, _ in combined]
        print(f"[FULL DOC] {filename} -> {len(chunks)} chunk(s) retrieved")
        return chunks
    except Exception as e:
        print(f"[FULL DOC] Failed to retrieve chunks for {filename}: {e}")
        return []


async def build_rag_context(collection, filename: str) -> str:
    """
    Builds the full patent context by concatenating ALL stored chunks in
    document order. The entire patent — cover page, abstract, description,
    examples, and claims — is passed to Gemini for analysis.

    No selective retrieval, no semantic filtering, no sections omitted.
    """
    chunks = get_all_chunks(collection, filename)
    if not chunks:
        print(f"[FULL DOC] No chunks found for {filename}")
        return ""

    context = "\n\n---\n\n".join(chunks)
    print(f"[FULL DOC] Built context: {len(chunks)} chunks | {len(context):,} chars for {filename}")
    return context


# ─────────────────────────────────────────────
# Gemini call helper (shared by all steps)
# ─────────────────────────────────────────────

async def _call_gemini_json(prompt: str, filename: str, step: str) -> Optional[Dict]:
    """
    Calls Gemini 2.5 Flash with JSON response mode.
    Retries up to _MAX_ANALYSIS_RETRIES times on truncation or empty response.
    On truncation retry, appends a concise-output instruction to reduce token usage.
    Used by every step in the pipeline.
    """
    _CONCISE_SUFFIX = (
        "\n\nIMPORTANT: Your previous response was truncated. "
        "Return ONLY the JSON object. Keep every string field under 60 words. "
        "Do NOT include any explanation outside the JSON."
    )

    for attempt in range(1, _MAX_ANALYSIS_RETRIES + 1):
        try:
            # On retry after truncation, ask for a more concise response
            current_prompt = prompt if attempt == 1 else prompt + _CONCISE_SUFFIX

            response = await gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )

            raw_text = response.text if response.text else ""

            try:
                finish_str = str(response.candidates[0].finish_reason)
                print(f"[{step}] finish_reason for {filename} (attempt {attempt}): {finish_str}")
                if finish_str in ("MAX_TOKENS", "2"):
                    print(f"[WARNING] Response truncated for {filename} (attempt {attempt}) — retrying with concise prompt")
                    if attempt < _MAX_ANALYSIS_RETRIES:
                        await asyncio.sleep(2 + random.uniform(0, 1))
                        continue
                elif finish_str in ("SAFETY", "3"):
                    print(f"[ERROR] Safety filter blocked response for {filename}.")
                    return None
            except (IndexError, AttributeError):
                pass

            if not raw_text.strip():
                print(f"[ERROR] Empty response for {filename} (attempt {attempt})")
                if attempt < _MAX_ANALYSIS_RETRIES:
                    await asyncio.sleep(2)
                    continue
                return None

            clean = raw_text.strip()
            if "```" in clean:
                clean = re.sub(r"```(?:json)?", "", clean).replace("```", "").strip()

            # Catch unterminated JSON before trying to parse — sign of silent truncation
            if clean and not clean.rstrip().endswith("}"):
                print(f"[WARNING] Response appears truncated (no closing brace) for {filename} (attempt {attempt})")
                if attempt < _MAX_ANALYSIS_RETRIES:
                    await asyncio.sleep(2 + random.uniform(0, 1))
                    continue

            print(f"[{step}] Raw response for {filename}: {clean[:300]!r}")
            return json.loads(clean)

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parse failed for {filename} (attempt {attempt}): {e}")
            if attempt < _MAX_ANALYSIS_RETRIES:
                await asyncio.sleep(2)
                continue
            return None

        except Exception as e:
            print(f"[ERROR] Gemini call failed for {filename} (attempt {attempt}): {e}")
            if attempt < _MAX_ANALYSIS_RETRIES:
                await asyncio.sleep(2)
                continue
            return None

    return None


# ─────────────────────────────────────────────
# STEP 1 — Claim classification
# ─────────────────────────────────────────────
#
# Classifies the patent's primary claim into one of 7 categories.
# If Composition of Matter → BLOCKING immediately.
# Otherwise → pass to Step 2 (pending).

STEP1_PROMPT = """You are a pharmaceutical patent expert.

SOURCE FILE: {filename}
PATENT NUMBER: {patent_number_hint}
JURISDICTION: {jurisdiction_hint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL PATENT DOCUMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────
TASK — STEP 1: CLASSIFY THE PRIMARY CLAIM
─────────────────────────────────────────────

Read the ENTIRE patent document thoroughly — every section including the title,
abstract, technical field, background, summary, detailed description, examples,
figures, tables, sequences, and any numbered or unnumbered paragraphs.
Do NOT search only for a "Claims" section. The document may not have one, or it
may be formatted differently. Instead, understand what the patent is PROTECTING
by reading all of the technical content — what compound, formulation, method,
device, or process is described as the invention throughout the document.
Identify the SINGLE best category that describes what this patent primarily protects.

CATEGORY DEFINITIONS:

1. Composition of Matter  ← STRICT DEFINITION — read carefully
   - The claim covers THE BASE MOLECULE ITSELF — its exact chemical structure,
     molecular formula, or the specific compound as a pure chemical entity
   - There can only be ONE true Composition of Matter patent per drug per
     jurisdiction — it is the foundational patent on the molecule itself
   - A claim is CoM ONLY if a competitor cannot make the molecule at all
     without infringing — because the patent owns the molecule itself
   - Examples: "A compound of formula [structure]...",
     "The peptide having the amino acid sequence...",
     "A GLP-1 receptor agonist having the structure..."
   - NOT CoM: salts of the molecule → use Salt/Polymorph
   - NOT CoM: polymorphs or crystalline forms → use Salt/Polymorph
   - NOT CoM: formulations containing the molecule → use Formulation
   - NOT CoM: methods of using the molecule → use Method of Treatment
   - NOT CoM: esters, prodrugs, or derivatives unless they ARE the drug itself
   - RULE: If the claim adds ANY qualifier beyond the bare molecule
     (a specific salt, a specific crystal form, a composition, a method),
     it is NOT Composition of Matter.

2. Salt/Polymorph
   - The claim covers a specific salt form, ester, prodrug, polymorph,
     crystalline form, amorphous form, or hydrate of the molecule
   - The molecule itself is NOT claimed — only a specific physical/chemical
     variant of it
   - Examples: "The sodium salt of...", "Crystalline Form A of...",
     "A polymorph of X characterised by X-ray diffraction peaks at...",
     "The acetate ester of...", "A co-crystal comprising..."

3. Formulation
   - The claim covers a pharmaceutical composition, formulation, or combination
   - Examples: "A pharmaceutical composition comprising X and excipient Y...",
     "A fixed-dose combination of X and Z...",
     "A sustained-release formulation comprising..."

4. Manufacturing Process
   - The claim covers a process or method of making or synthesizing the compound
   - Examples: "A process for preparing compound X comprising the steps of...",
     "A method of synthesizing..."

5. Method of Treatment
   - The claim covers a method of using the drug to treat a disease or condition
   - Examples: "A method of treating type 2 diabetes comprising administering...",
     "Use of compound X for treatment of obesity..."

6. Device
   - The claim covers a delivery device, drug-device combination, or administration system
   - Examples: "An injection pen for administering...", "An inhaler device comprising...",
     "A transdermal patch system..."

7. Dosage Regimen
   - The claim covers a specific dosing schedule, dose amount, or frequency of administration
   - Examples: "A method comprising administering X at a dose of Y mg once weekly...",
     "A dosing regimen comprising an initial dose of..."

─────────────────────────────────────────────
CLASSIFICATION RULES:
─────────────────────────────────────────────
- Determine what the patent PRIMARILY protects by reading the full document.
- Use the title, abstract, summary, detailed description, and examples together
  to understand the core invention — do not rely on any single section.
- Choose the category that best describes the BROADEST protection the patent
  appears to offer based on all technical content in the document.
- If the document primarily describes a bare molecule by its chemical structure,
  molecular formula, or amino acid sequence — classify as "Composition of Matter".
- CoM TEST: "Does the document primarily describe and protect a compound defined
  by its chemical structure, molecular formula, or amino acid sequence, without
  restricting it to a specific salt form, polymorph, or formulation?"
  If YES → Composition of Matter.
- Salt/Polymorph ONLY if the document focuses on a specific physical/chemical
  variant (named salt, crystal form, hydrate) rather than the bare molecule.
- Formulation ONLY if the document centres on a pharmaceutical composition or
  mixture, not the molecule itself.
- Do NOT downgrade a genuine compound patent to Salt/Polymorph or Formulation
  just because the document also describes such variants.

─────────────────────────────────────────────
ALSO EXTRACT:
─────────────────────────────────────────────
- patent_number: read from the document cover page or header
- jurisdiction:  two-letter office code (US, EP, WO, GB, JP, CN, AU, CA)
- pte:           Patent Term Extension in months (look for "Patent Term Extension",
                 "PTE", "35 U.S.C. 156", "SPC"). Convert years to months if needed. null if absent.
- pediatric_exclusivity: true ONLY if the document explicitly states pediatric
                 exclusivity is granted (look for "BPCA", "6-month exclusivity",
                 "pediatric extension"). false otherwise.

─────────────────────────────────────────────
OUTPUT — return ONLY valid JSON, no markdown:
─────────────────────────────────────────────
{{
  "patent_number":            "read from document, or '{patent_number_hint}' as fallback",
  "jurisdiction":             "two-letter code, or '{jurisdiction_hint}' as fallback",
  "claim_category":           "exactly one of the 7 categories below",
  "is_composition_of_matter": true ONLY if claim_category is "Composition of Matter" AND the claim covers the bare molecule itself with no salt/polymorph/formulation qualifiers — false otherwise,
  "reason":                   "1-2 sentences: what the primary claim covers and why you chose this category",
  "pte":                      number of months as integer or null,
  "pediatric_exclusivity":    true or false
}}

claim_category must be EXACTLY one of:
  "Composition of Matter"
  "Salt/Polymorph"
  "Formulation"
  "Manufacturing Process"
  "Method of Treatment"
  "Device"
  "Dosage Regimen"
"""


async def _run_step1(
    filename:           str,
    context:            str,
    patent_number_hint: str,
    jurisdiction_hint:  str,
) -> Optional[Dict]:
    """
    Step 1: Classify the patent's primary claim into one of 7 categories.
    Returns the parsed JSON dict, or None on failure.
    """
    print(f"[STEP 1] Classifying claim for {filename}...")

    safe_context = context.replace("{", "{{").replace("}", "}}")
    prompt = STEP1_PROMPT.format(
        filename           = filename,
        patent_number_hint = patent_number_hint,
        jurisdiction_hint  = jurisdiction_hint,
        context            = safe_context,
    )

    result = await _call_gemini_json(prompt, filename, "STEP1")
    if result is None:
        return None

    # Normalise claim_category to exact spelling
    raw_category = (result.get("claim_category") or "").strip()
    matched = next(
        (c for c in CLAIM_CATEGORIES if c.lower() == raw_category.lower()),
        None,
    )
    if not matched:
        print(f"[STEP 1] Unrecognised category '{raw_category}' for {filename} — defaulting to None")
    result["claim_category"] = matched

    # Enforce consistency: category and flag must agree
    if result["claim_category"] == "Composition of Matter":
        result["is_composition_of_matter"] = True
    else:
        result["is_composition_of_matter"] = False

    print(
        f"[STEP 1] {filename} → "
        f"Category: {result['claim_category']} | "
        f"CoM: {result['is_composition_of_matter']} | "
        f"Reason: {(result.get('reason') or '')[:80]}"
    )
    return result





# ─────────────────────────────────────────────
# STEP 2 — Claim element matching
# ─────────────────────────────────────────────
#
# Checks if the patent's claims cover any of these 5 elements:
#   1. Active ingredient and form
#   2. Formulation details
#   3. Route of administration
#   4. Device description
#   5. Combination tech/process
#
# Reference: the drug's known real-world profile from its FDA label.
# If the label is unavailable, Gemini checks the patent text alone.
#
# ANY element present → continue to Step 3.
# NONE present       → NON-BLOCKING.

STEP2_PROMPT = """You are a pharmaceutical patent expert performing Freedom-to-Operate analysis.
You are a STRICT REVIEWER. Your default position is NO MATCH unless the evidence is specific and direct.

PATENT FILE    : {filename}
PATENT NUMBER  : {patent_number}
JURISDICTION   : {jurisdiction}
CLAIM CATEGORY : {claim_category}  (classified in Step 1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS DRUG'S REAL-WORLD FORMULATION DATA
(Sourced from clinical trials and published sources — {row_count} record(s))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{formulation_rows}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL PATENT DOCUMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────
TASK — STEP 2: MATCH PATENT CLAIMS TO DRUG PROFILE
─────────────────────────────────────────────

For each of the 5 elements below, determine whether this patent specifically
and directly covers what this drug actually uses.
Read the ENTIRE document — title, abstract, background, detailed description,
examples, figures, sequences, and all numbered or unnumbered paragraphs.
Do NOT limit your reading to any section labelled "Claims". Understand what
the patent protects by reading ALL technical content in the document and
assess whether that protection covers what this drug actually uses.

─────────────────────────────────────────────
STRICT MATCHING RULES — apply to every element:
─────────────────────────────────────────────
- Read the ENTIRE document. Use the title, abstract, technical description,
  examples, and all other content to understand what the patent protects.
- Any section of the document — description, summary, examples, or numbered
  paragraphs — can inform your understanding of what is being protected.
- Background context helps you understand scope but does not itself define
  what is protected.
- The match must be SPECIFIC and DIRECT. The patent claim must explicitly
  describe the feature in a way that maps to this drug's actual profile.
- BROAD or GENERIC language does NOT match:
    "pharmaceutically acceptable salt"  → NOT a match (applies to all drugs)
    "aqueous solution"                  → NOT a match (too generic)
    "a therapeutically effective amount"→ NOT a match (applies to all drugs)
    "oral administration"               → NOT a match unless the claim specifies
                                          a feature of oral delivery unique to this drug
- The patent must claim something SPECIFIC to this drug's formulation —
  not something that would apply to any drug in the same class.
- If the patent claim is so broad it would read on thousands of drugs,
  it does NOT match for this drug specifically.
- Missing/empty fields → do not assume a match; treat as no match for that element.

─────────────────────────────────────────────
THE 5 ELEMENTS — STRICT CRITERIA:
─────────────────────────────────────────────

1. Active Ingredient & Form
   The patent must claim a SPECIFIC chemical form (a named salt, a specific
   polymorph, a specific ester or derivative) that matches exactly what this
   drug uses. Generic claims covering "any salt" or "any form" → NOT a match.

2. Formulation Details
   The patent must claim a SPECIFIC formulation feature (a named excipient,
   a specific concentration range, a specific release mechanism technology)
   that this drug demonstrably uses. Generic dosage form claims ("tablet",
   "solution", "injection") with no specifics → NOT a match.

3. Route of Administration
   Route alone is almost never a match. The patent must claim a SPECIFIC
   technical feature OF the route (a specific absorption mechanism, a specific
   device-route combination, a specific tissue target) that is unique to this
   drug's delivery. Simply claiming "subcutaneous" or "oral" → NOT a match.

4. Device Description
   The patent must claim a SPECIFIC device feature (a specific mechanism,
   a specific needle configuration, a specific reservoir design) that matches
   the device this drug actually uses. Generic claims for "a pen injector" or
   "a prefilled syringe" → NOT a match.

5. Combination Tech/Process
   The patent must claim a SPECIFIC named technology or process (e.g. SNAC
   co-formulation, a specific encapsulation process, a named absorption
   enhancer) that appears explicitly in this drug's records. General
   manufacturing process claims → NOT a match.

─────────────────────────────────────────────
PASS / FAIL DECISION:
─────────────────────────────────────────────
- any_element_present = true if AT LEAST ONE element matches under the
  strict criteria above.
- any_element_present = false ONLY if there is genuinely zero specific overlap
  between the patent's claims and this drug's actual profile across all 5 elements.
- When in doubt on any element → mark it false.

─────────────────────────────────────────────
OUTPUT — return ONLY valid JSON, no markdown:
─────────────────────────────────────────────
{{
  "elements_present": {{
    "active_ingredient_and_form": true or false,
    "formulation_details":        true or false,
    "route_of_administration":    true or false,
    "device_description":         true or false,
    "combination_tech_process":   true or false
  }},
  "any_element_present": true or false,
  "matched_elements":    ["list of element names that matched under strict criteria"],
  "reason": "1-2 sentences: specifically what matched or why nothing matched"
}}
"""


def _format_rows_for_prompt(rows: List[Dict]) -> str:
    """
    Format a list of Excel row dicts into a readable block for the Gemini prompt.
    Each row is numbered and only non-empty fields are shown.
    """
    if not rows:
        return "No formulation records available — assess from patent claims alone."

    lines = []
    for i, row in enumerate(rows, start=1):
        lines.append(f"Record {i}:")
        for key, val in row.items():
            lines.append(f"  {key}: {val}")
        lines.append("")  # blank line between records

    return "\n".join(lines).strip()


async def _run_step2(
    filename:     str,
    context:      str,
    step1_result: Dict,
    drug_rows:    List[Dict],
) -> Optional[Dict]:
    """
    Step 2: Check if the patent's claims cover any of the 5 formulation elements.

    Args:
        filename:     Patent PDF filename
        context:      RAG context string (same as Step 1)
        step1_result: Output dict from _run_step1
        drug_rows:    All rows from the formulation Excel for this drug.
                      Empty list if no Excel data available.

    Returns:
        Parsed JSON dict with elements_present, any_element_present, matched_elements, reason.
        Returns None on Gemini failure.
    """
    print(f"[STEP 2] Checking claim elements for {filename} "
          f"({len(drug_rows)} formulation record(s))...")

    formatted_rows = _format_rows_for_prompt(drug_rows)
    safe_context   = context.replace("{", "{{").replace("}", "}}")

    prompt = STEP2_PROMPT.format(
        filename       = filename,
        patent_number  = step1_result.get("patent_number", Path(filename).stem),
        jurisdiction   = step1_result.get("jurisdiction", ""),
        claim_category = step1_result.get("claim_category", ""),
        row_count      = len(drug_rows),
        formulation_rows = formatted_rows,
        context        = safe_context,
    )

    result = await _call_gemini_json(prompt, filename, "STEP2")
    if result is None:
        return None

    # Normalise
    elements = result.get("elements_present", {})
    matched  = result.get("matched_elements") or [k for k, v in elements.items() if v]
    result["matched_elements"] = matched

    # ── 1-element minimum gate ────────────────────────────────────────────────
    match_count = len(matched)
    result["any_element_present"] = match_count >= 1

    print(
        f"[STEP 2] {filename}\n"
        f"  Matched ({match_count}) : {matched}\n"
        f"  Pass gate  : {result['any_element_present']}\n"
        f"  Reason     : {(result.get('reason') or '')[:150]}"
    )
    return result


# ─────────────────────────────────────────────
# STEP 3 — Scientific barrier analysis
# ─────────────────────────────────────────────

# Maximum characters of evidence passed into any single prompt.
# Each source is trimmed proportionally so all sources contribute equally.
# This prevents Gemini output truncation regardless of how much evidence was gathered.
_MAX_EVIDENCE_CHARS = 12_000


def _cap_evidence(evidence_block: str, max_chars: int = _MAX_EVIDENCE_CHARS) -> str:
    """
    Caps the evidence block to max_chars characters.
    Splits on source section boundaries (lines starting with "[") so each
    source is trimmed proportionally rather than cutting mid-sentence.
    Appends a note if truncation occurred.
    """
    if len(evidence_block) <= max_chars:
        return evidence_block

    # Split into named sections — each starts with a "[Source]" header line
    sections = re.split(r'(?=^\[)', evidence_block, flags=re.MULTILINE)
    if not sections:
        return evidence_block[:max_chars] + "\n\n[Evidence truncated to fit context window]"

    per_section = max(max_chars // max(len(sections), 1), 500)
    capped = []
    total  = 0
    for section in sections:
        if not section.strip():
            continue
        allowed = min(per_section, max_chars - total)
        if allowed <= 0:
            break
        chunk = section[:allowed]
        if len(section) > allowed:
            # Try to cut at a sentence boundary
            cut = max(chunk.rfind(". "), chunk.rfind("\n"))
            if cut > allowed // 2:
                chunk = chunk[:cut + 1]
        capped.append(chunk.strip())
        total += len(chunk)

    result = "\n\n".join(capped)
    if len(evidence_block) > len(result):
        result += "\n\n[Evidence capped to fit context window — full details available in logs]"
    print(f"[EVIDENCE CAP] {len(evidence_block):,} chars -> {len(result):,} chars "
          f"({len(sections)} section(s))")
    return result

#
# Determines whether the claimed feature solves a REAL technical barrier
# (stability, bioavailability, safety) vs. being optional/incremental.
#
# Marketed drugs  → FDA Medical/CMC Review PDFs + EMA Assessment Report
# Clinical drugs  → PubMed abstracts (6 keyword combos) + Gemini journal search
#                   + completed trial rows from Excel
#
# is_technical_barrier = True  → continue to Step 4
# is_technical_barrier = False → NON-BLOCKING

_PUBMED_ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_EFETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PUBMED_EMAIL    = os.getenv("PUBMED_EMAIL", "patent_analysis@example.com")
_PUBMED_API_KEY  = os.getenv("NCBI_API_KEY")  # free at ncbi.nlm.nih.gov/account
# With API key: 10 req/s. Without: 3 req/s. Use conservative delay either way.
_PUBMED_DELAY    = 0.15 if _PUBMED_API_KEY else 0.4   # seconds between requests
_PUBMED_MAX_RETRIES = 3
_HTTP_HEADERS    = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
_OPEN_FDA_BASE   = "https://api.fda.gov/drug/drugsfda.json"

_KEYWORD_TEMPLATES = [
    "{molecule} formulation stability",
    "{molecule} pharmacokinetics",
    "{molecule} bioavailability",
    "{molecule} phase clinical trial",
    "{molecule} degradation",
    "{molecule} delivery system",
]


# ── PubMed helpers ────────────────────────────────────────────────────────────

def _pubmed_search(query: str, max_results: int = 5) -> List[str]:
    """Search PubMed and return a list of PMIDs. Retries on 429."""
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "email":   _PUBMED_EMAIL,
    }
    if _PUBMED_API_KEY:
        params["api_key"] = _PUBMED_API_KEY

    for attempt in range(1, _PUBMED_MAX_RETRIES + 1):
        try:
            r = requests.get(
                _PUBMED_ESEARCH,
                params=params,
                timeout=15,
                headers=_HTTP_HEADERS,
            )
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[PUBMED] 429 on '{query}' (attempt {attempt}) — waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            print(f"[PUBMED] '{query}' → {len(ids)} result(s)")
            return ids
        except Exception as e:
            if attempt == _PUBMED_MAX_RETRIES:
                print(f"[PUBMED] Search failed for '{query}' after {_PUBMED_MAX_RETRIES} attempts: {e}")
                return []
            time.sleep(1)
    return []


def _pubmed_fetch_abstracts(pmids: List[str]) -> str:
    """Fetch abstracts for a list of PMIDs. Returns concatenated plain text."""
    if not pmids:
        return ""

    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
        "email":   _PUBMED_EMAIL,
    }
    if _PUBMED_API_KEY:
        params["api_key"] = _PUBMED_API_KEY

    for attempt in range(1, _PUBMED_MAX_RETRIES + 1):
        try:
            r = requests.get(
                _PUBMED_EFETCH,
                params=params,
                timeout=20,
                headers=_HTTP_HEADERS,
            )
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[PUBMED] 429 on fetch (attempt {attempt}) — waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text.strip()
        except Exception as e:
            if attempt == _PUBMED_MAX_RETRIES:
                print(f"[PUBMED] Fetch failed for PMIDs {pmids}: {e}")
                return ""
            time.sleep(1)
    return ""


async def _gather_pubmed_evidence(drug_name: str) -> str:
    """
    Run all 6 keyword searches against PubMed SEQUENTIALLY with rate-limit delays.
    Parallel requests cause 429s — PubMed allows max 3 req/s without API key.
    Returns a single string of deduplicated abstracts.
    """
    loop    = asyncio.get_event_loop()
    queries = [t.format(molecule=drug_name) for t in _KEYWORD_TEMPLATES]

    seen_pmids: set = set()
    all_pmids:  List[str] = []

    for query in queries:
        pmids = await loop.run_in_executor(None, _pubmed_search, query, 5)
        for pmid in pmids:
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_pmids.append(pmid)
        # Respect PubMed rate limit between searches
        await asyncio.sleep(_PUBMED_DELAY)

    if not all_pmids:
        return ""

    print(f"[PUBMED] Fetching {len(all_pmids)} unique abstract(s) for '{drug_name}'")
    await asyncio.sleep(_PUBMED_DELAY)
    abstracts = await loop.run_in_executor(None, _pubmed_fetch_abstracts, all_pmids[:20])
    return abstracts


# ── Gemini open evidence search (Google Search grounding) ────────────────────

_SEARCH_ANGLES = [
    {
        "label": "Regulatory",
        "prompt": (
            "Search for regulatory and scientific evidence about whether this specific "
            "feature of '{drug_name}' is technically necessary for the drug to work:\n\n"
            "Patent claim: {claim_category} — {claim_reason}\n\n"
            "Look across ALL available sources: FDA reviews, EMA assessment reports, "
            "WHO reports, ICH guidelines, regulatory agency publications, health authority "
            "submissions, drug dossiers, post-market surveillance reports, and any other "
            "regulatory or government source that addresses whether this feature was required "
            "for stability, bioavailability, safety, or regulatory approval.\n\n"
            "Do NOT restrict yourself to any specific source. Find the most authoritative "
            "evidence available anywhere on the web.\n\n"
            "Summarise the key findings in 250 words. Cite the source name and year."
        ),
    },
    {
        "label": "Scientific Literature",
        "prompt": (
            "Search for peer-reviewed scientific evidence about whether this specific "
            "feature of '{drug_name}' solves a real technical barrier:\n\n"
            "Patent claim: {claim_category} — {claim_reason}\n\n"
            "Look across ALL scientific sources: journals, conference proceedings, "
            "preprints (bioRxiv, medRxiv), dissertations, technical reports, patent "
            "literature, pharmacopoeia monographs, and any scientific publication that "
            "addresses whether this formulation or delivery feature is scientifically "
            "necessary vs optional.\n\n"
            "Focus on: stability data, bioavailability studies, pharmacokinetics, "
            "degradation mechanisms, solubility challenges, absorption barriers.\n\n"
            "Do NOT restrict yourself to specific journals. Find the most relevant "
            "scientific evidence anywhere.\n\n"
            "Summarise the key findings in 250 words. Cite source names and years."
        ),
    },
    {
        "label": "Clinical & Industry",
        "prompt": (
            "Search for clinical and industry evidence about '{drug_name}' related to "
            "this patent claim:\n\n"
            "Patent claim: {claim_category} — {claim_reason}\n\n"
            "Look across ALL relevant sources: ClinicalTrials.gov records, clinical study "
            "reports, pharmaceutical company technical documents, industry white papers, "
            "patent filings by competitors (which may acknowledge the technical problem), "
            "drug product information sheets, pharmacist references, hospital formulary "
            "documents, and any clinical or industry source that discusses whether this "
            "feature was technically essential.\n\n"
            "Do NOT restrict yourself to any specific database. Find the most relevant "
            "evidence available.\n\n"
            "Summarise the key findings in 250 words. Cite source names and years."
        ),
    },
]


async def _gather_gemini_evidence_angle(
    drug_name:      str,
    claim_category: str,
    claim_reason:   str,
    angle:          dict,
) -> str:
    """Run a single search angle via Gemini with Google Search grounding."""
    prompt = angle["prompt"].format(
        drug_name      = drug_name,
        claim_category = claim_category,
        claim_reason   = claim_reason,
    )
    label = angle["label"]

    try:
        response = await gemini_client.aio.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = prompt,
            config   = types.GenerateContentConfig(
                tools       = [types.Tool(google_search=types.GoogleSearch())],
                temperature = 0.1,
            ),
        )
        text = (response.text or "").strip()
        if text:
            print(f"[GEMINI SEARCH] {label}: {len(text)} chars for '{drug_name}'")
            return f"[{label} Evidence — Open Web Search]\n{text}"
        return ""
    except Exception as e:
        print(f"[GEMINI SEARCH] {label} failed for '{drug_name}': {e}")
        return ""


async def _gather_all_gemini_evidence(
    drug_name:      str,
    claim_category: str,
    claim_reason:   str,
) -> str:
    """
    Run all 3 search angles in parallel via Gemini Google Search grounding.
    Each angle searches the open web with no source restrictions.
    Returns combined evidence string.
    """
    results = await asyncio.gather(
        *[
            _gather_gemini_evidence_angle(drug_name, claim_category, claim_reason, angle)
            for angle in _SEARCH_ANGLES
        ],
        return_exceptions=True,
    )

    parts = []
    for angle, result in zip(_SEARCH_ANGLES, results):
        if isinstance(result, Exception):
            print(f"[GEMINI SEARCH] {angle['label']} error: {result}")
        elif result and str(result).strip():
            parts.append(str(result).strip())

    return "\n\n".join(parts)


# ── FDA Medical/CMC Review fetcher ────────────────────────────────────────────

def _get_fda_review_urls(drug_name: str) -> List[Tuple[str, str]]:
    """
    Query Drugs@FDA and return the single latest Label URL and single latest
    Review URL — same logic as fda_label_extractor.get_brand_nda_details_with_dates.

    Tracks the latest doc by date across ALL NDA applications.
    Returns at most 2 entries: (Label, url) and/or (Review, url).
    """
    try:
        r = requests.get(
            _OPEN_FDA_BASE,
            params={
                "search": f'products.active_ingredients.name:"{drug_name}"',
                "limit":  10,
            },
            timeout=20,
            headers=_HTTP_HEADERS,
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        results = r.json().get("results", []) or []
    except Exception as e:
        print(f"[FDA REVIEW] API request failed for '{drug_name}': {e}")
        return []

    # Track single latest Label and single latest Review across all NDAs
    latest_docs: Dict[str, Dict] = {
        "Label":  {"date": "0", "url": None},
        "Review": {"date": "0", "url": None},
    }

    for app in results:
        for submission in app.get("submissions", []) or []:
            for doc in submission.get("application_docs", []) or []:
                doc_type = doc.get("type", "")
                doc_date = doc.get("date", "") or "0"
                doc_url  = doc.get("url", "")

                # Map all review variants to single "Review" key
                if doc_type in ("Review", "Medical Review", "Chemistry Review",
                                "Pharmacology Review", "Clinical Pharmacology Review"):
                    key = "Review"
                elif doc_type == "Label":
                    key = "Label"
                else:
                    continue

                if doc_url and doc_date > latest_docs[key]["date"]:
                    latest_docs[key] = {"date": doc_date, "url": doc_url}
                    print(f"[FDA REVIEW] Latest {key} so far: {doc_date} | {doc_url[:80]}")

    result = []
    for key in ("Label", "Review"):
        url = latest_docs[key]["url"]
        if url:
            print(f"[FDA REVIEW] Using latest {key}: {url[:80]}")
            result.append((key, url))

    if not result:
        print(f"[FDA REVIEW] No Label or Review PDFs found for '{drug_name}'")

    return result


async def _analyse_fda_review_pdf(pdf_url: str, doc_type: str, drug_name: str, claim_reason: str) -> str:
    """Send an FDA review PDF to Gemini and extract scientific necessity evidence."""
    prompt = (
        f"You are analysing an FDA {doc_type} for '{drug_name}'.\n\n"
        f"The patent in question claims: {claim_reason}\n\n"
        f"From this regulatory review document, extract ONLY content relevant to:\n"
        f"1. Whether the specific formulation feature, delivery system, or process was "
        f"considered NECESSARY by the FDA for approval (stability, bioavailability, safety)\n"
        f"2. Any statements indicating the feature solved a technical problem\n"
        f"3. Any CMC (Chemistry, Manufacturing, Controls) requirements related to the claim\n\n"
        f"If the document does not address technical necessity of the claimed feature, "
        f"state 'Not addressed in this review'.\n\n"
        f"Be concise — 200 words maximum."
    )

    try:
        response = await gemini_client.aio.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = [
                types.Part.from_uri(file_uri=pdf_url, mime_type="application/pdf"),
                prompt,
            ],
            config = types.GenerateContentConfig(
                temperature       = 0.1,
                max_output_tokens = 1024,
            ),
        )
        text = response.text or ""
        print(f"[FDA REVIEW] Extracted {len(text)} chars from {doc_type}")
        return f"[FDA {doc_type}]\n{text.strip()}"
    except Exception as e:
        print(f"[FDA REVIEW] Gemini analysis failed for {doc_type}: {e}")
        return ""


async def _gather_fda_evidence(drug_name: str, claim_reason: str) -> str:
    """Fetch and analyse the latest FDA Label + Review PDFs for marketed drugs."""
    loop        = asyncio.get_event_loop()
    review_urls = await loop.run_in_executor(None, _get_fda_review_urls, drug_name)

    if not review_urls:
        print(f"[FDA REVIEW] No review PDFs found for '{drug_name}'")
        return ""

    results = await asyncio.gather(
        *[_analyse_fda_review_pdf(url, doc_type, drug_name, claim_reason)
          for doc_type, url in review_urls],
        return_exceptions=True,
    )

    parts = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[FDA REVIEW] Error: {r}")
        elif r:
            parts.append(r)

    return "\n\n".join(parts)


# ── EMA Assessment Report fetcher ────────────────────────────────────────────

_EMA_EPAR_PREFERENCE = [
    "product_information_pdf",
    "public_summary_pdf",
    "overview_pdf",
    "risk_management_plan_summary_pdf",
]

_EMA_STEP3_PROMPT = """You are analysing an EMA EPAR document for '{drug_name}'.

The patent in question claims: {claim_reason}

From this EMA regulatory document, extract ONLY content relevant to:
1. Whether the specific formulation feature, delivery system, or process was
   considered NECESSARY by EMA for marketing authorisation
2. Any scientific assessment of technical necessity (stability, bioavailability, safety)
3. Any pharmaceutical development (CMC) findings related to the claim
4. Statements about why specific excipients, devices, or processes were required

If the document does not address technical necessity of the claimed feature,
state 'Not addressed in this EMA document'.

Be concise — 200 words maximum."""


async def _analyse_ema_pdf(pdf_url: str, doc_type: str, drug_name: str, claim_reason: str) -> str:
    """Send a single EMA EPAR PDF to Gemini and extract Step 3 evidence."""
    prompt = _EMA_STEP3_PROMPT.format(
        drug_name    = drug_name,
        claim_reason = claim_reason,
    )
    try:
        response = await gemini_client.aio.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = [
                types.Part.from_uri(file_uri=pdf_url, mime_type="application/pdf"),
                prompt,
            ],
            config = types.GenerateContentConfig(
                temperature       = 0.1,
                max_output_tokens = 1024,
            ),
        )
        text = (response.text or "").strip()
        print(f"[EMA EPAR] Extracted {len(text)} chars from {doc_type} for '{drug_name}'")
        return f"[EMA EPAR — {doc_type}]\n{text}"
    except Exception as e:
        print(f"[EMA EPAR] Gemini failed for {doc_type} ({pdf_url[:60]}): {e}")
        return ""


async def _gather_ema_evidence(drug_name: str, claim_reason: str) -> str:
    """
    Resolve all EMA brand names for the drug, fetch their EPAR PDFs,
    and extract Step 3 scientific barrier evidence from each.

    Uses ema_epar_extractor._resolve_all_ema_brands + get_ema_epar_documents
    for brand resolution and PDF link finding.
    Deduplicates PDFs so the same file is never sent to Gemini twice.
    """
    loop = asyncio.get_event_loop()

    # Step 1 — resolve all EMA brand names
    print(f"[EMA EPAR] Resolving EMA brands for '{drug_name}'...")
    brands = await loop.run_in_executor(None, _resolve_all_ema_brands, drug_name)

    if not brands:
        print(f"[EMA EPAR] No EMA brands found for '{drug_name}' — skipping")
        return ""

    print(f"[EMA EPAR] Found {len(brands)} brand(s): {brands}")

    # Step 2 — collect EPAR PDF URLs (deduplicated)
    seen_urls: set = set()
    pdf_tasks: List[Tuple[str, str]] = []  # (doc_type, url)

    for brand in brands:
        try:
            links = await loop.run_in_executor(None, _get_ema_epar_documents, brand)
        except Exception as e:
            print(f"[EMA EPAR] Failed to get EPAR docs for '{brand}': {e}")
            continue

        for key in _EMA_EPAR_PREFERENCE:
            url = links.get(key)
            if url and url not in seen_urls:
                seen_urls.add(url)
                pdf_tasks.append((key, url))
                print(f"[EMA EPAR] '{brand}' → {key}: {url[:80]}")
                break  # one PDF per brand

    if not pdf_tasks:
        print(f"[EMA EPAR] No EPAR PDFs found for any brand of '{drug_name}'")
        return ""

    # Step 3 — send each PDF to Gemini in parallel
    print(f"[EMA EPAR] Analysing {len(pdf_tasks)} EPAR PDF(s) for '{drug_name}'...")
    results = await asyncio.gather(
        *[_analyse_ema_pdf(url, doc_type, drug_name, claim_reason)
          for doc_type, url in pdf_tasks],
        return_exceptions=True,
    )

    parts = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[EMA EPAR] Error: {r}")
        elif r and str(r).strip():
            parts.append(str(r).strip())

    if not parts:
        return ""

    return "[EMA Assessment Reports]\n\n" + "\n\n".join(parts)


# ── Europe PMC ────────────────────────────────────────────────────────────────

_EUROPEPMC_BASE    = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_EUROPEPMC_DELAY   = 0.5   # seconds between requests — EBI requests polite crawling
_EUROPEPMC_RETRIES = 3


def _europepmc_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search Europe PMC and return article dicts with title + abstract.
    Sorted by citation count (most cited first).
    Retries on 429.
    """
    params = {
        "query":      query,
        "resultType": "core",
        "pageSize":   max_results,
        "format":     "json",
        "sort":       "CITED desc",
    }

    for attempt in range(1, _EUROPEPMC_RETRIES + 1):
        try:
            r = requests.get(
                _EUROPEPMC_BASE,
                params=params,
                timeout=15,
                headers=_HTTP_HEADERS,
            )
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[EUROPEPMC] 429 on '{query}' (attempt {attempt}) — waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            articles = r.json().get("resultList", {}).get("result", []) or []
            print(f"[EUROPEPMC] '{query}' → {len(articles)} result(s)")
            return articles
        except Exception as e:
            if attempt == _EUROPEPMC_RETRIES:
                print(f"[EUROPEPMC] Search failed for '{query}': {e}")
                return []
            time.sleep(1)
    return []


def _europepmc_format_results(articles: List[Dict]) -> str:
    """Format Europe PMC results into a readable evidence block."""
    if not articles:
        return ""
    lines = []
    for a in articles:
        title    = a.get("title", "").strip()
        abstract = (a.get("abstractText") or "").strip()
        journal  = a.get("journalTitle", "").strip()
        year     = a.get("pubYear", "")
        cited_by = a.get("citedByCount", 0)
        pmid     = a.get("pmid", "")

        if not abstract:
            continue

        lines.append(
            f"Title   : {title}\n"
            f"Journal : {journal} ({year}) | Cited by: {cited_by} | PMID: {pmid}\n"
            f"Abstract: {abstract[:600]}{'...' if len(abstract) > 600 else ''}\n"
        )

    return "\n---\n".join(lines)


async def _gather_europepmc_evidence(drug_name: str) -> str:
    """
    Run all 6 keyword searches against Europe PMC sequentially with delays.
    Returns deduplicated formatted evidence block.
    """
    loop    = asyncio.get_event_loop()
    queries = [t.format(molecule=drug_name) for t in _KEYWORD_TEMPLATES]

    seen_ids:     set       = set()
    all_articles: List[Dict] = []

    for query in queries:
        articles = await loop.run_in_executor(None, _europepmc_search, query, 5)
        for article in articles:
            uid = article.get("id") or article.get("pmid") or article.get("doi", "")
            if uid and uid not in seen_ids:
                seen_ids.add(uid)
                all_articles.append(article)
        await asyncio.sleep(_EUROPEPMC_DELAY)

    if not all_articles:
        print(f"[EUROPEPMC] No articles found for '{drug_name}'")
        return ""

    print(f"[EUROPEPMC] {len(all_articles)} unique article(s) for '{drug_name}'")
    formatted = _europepmc_format_results(all_articles[:15])
    return f"[Europe PMC Evidence]\n{formatted}" if formatted else ""


# ── Completed trial rows from Excel ───────────────────────────────────────────

def _get_completed_rows(drug_rows: List[Dict]) -> List[Dict]:
    """Filter Excel rows to Status = Completed only."""
    completed = [
        r for r in drug_rows
        if str(r.get("Status", "")).strip().lower() == "completed"
    ]
    print(f"[STEP 3] {len(completed)} completed trial row(s) out of {len(drug_rows)} total")
    return completed


# ── Step 3 Gemini prompt ──────────────────────────────────────────────────────

STEP3_PROMPT = """You are a pharmaceutical patent expert performing scientific barrier analysis.
You are a SCEPTICAL REVIEWER. Your default position is NON-BLOCKING unless the evidence
explicitly and unambiguously proves otherwise.

DRUG NAME      : {drug_name}
PATENT NUMBER  : {patent_number}
JURISDICTION   : {jurisdiction}
CLAIM CATEGORY : {claim_category}  (Step 1)
CLAIM DETAILS  : {step1_reason}
MATCHED ELEMENTS (Step 2): {matched_elements}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCIENTIFIC EVIDENCE GATHERED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{evidence_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATENT DOCUMENT CHUNKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────
TASK — STEP 3: SCIENTIFIC BARRIER ANALYSIS
─────────────────────────────────────────────

Determine whether the claimed feature solves a REAL TECHNICAL BARRIER.
The standard is HIGH. Most patents will NOT meet it.

A REAL TECHNICAL BARRIER requires ALL of the following to be true:
1. Without this SPECIFIC feature (not just some formulation), the drug
   demonstrably fails — on stability, bioavailability, absorption, or safety
2. The evidence is EXPLICIT and DIRECT — a regulatory review (FDA/EMA) or
   peer-reviewed study states this specific feature was necessary, not just
   that it was used or that it produced improvements
3. No scientifically equivalent alternative existed at the time of filing
   that could achieve the same outcome

DOES NOT qualify as a real technical barrier:
- The feature improves convenience, dosing frequency, or patient compliance
- The feature improves manufacturing efficiency or cost
- Regulators accepted or approved the drug WITH this feature (accepted ≠ required)
- The drug works without this feature but less conveniently
- The evidence only shows the feature is common practice or industry standard
- General statements that formulation is important for this drug class
- The patent itself claims the feature is necessary (self-serving, not independent evidence)
- Improvements in PK/PD that are incremental, not enabling
- The claim category alone implies technical need — category is NOT evidence

─────────────────────────────────────────────
STRICT DECISION RULES — follow in order:
─────────────────────────────────────────────
1. If no independent external evidence (FDA/EMA/peer-reviewed) was found
   → is_technical_barrier = false, confidence = low

2. If evidence exists but only shows the feature is beneficial or preferred,
   not that the drug fails without it
   → is_technical_barrier = false, confidence = medium

3. If evidence is present but indirect, inferred, or from manufacturer sources only
   → is_technical_barrier = false, confidence = medium

4. Only if EXPLICIT independent evidence directly states the feature was
   technically necessary for the drug to function safely and effectively
   → is_technical_barrier = true, confidence = high

5. If you are uncertain after applying rules 1–4
   → is_technical_barrier = false, confidence = low

Confidence levels:
  high   → explicit, direct, independent statement of technical necessity
  medium → evidence exists but does not explicitly confirm necessity
  low    → no relevant evidence, or evidence is indirect/manufacturer-sourced

─────────────────────────────────────────────
OUTPUT — return ONLY valid JSON, no markdown:
─────────────────────────────────────────────
{{
  "is_technical_barrier": true or false,
  "confidence":           "high" or "medium" or "low",
  "evidence_type":        "FDA Review" | "EMA Assessment" | "Peer-reviewed Journal" | "Multiple Sources" | "Insufficient Evidence",
  "evidence_summary":     "2-3 sentences summarising the key scientific evidence found and why it does or does not confirm technical necessity",
  "reason":               "1-2 sentences: specific reason why this feature is/is not a real technical barrier — cite the evidence"
}}
"""


async def _run_step3(
    filename:       str,
    context:        str,
    step1_result:   Dict,
    step2_result:   Dict,
    drug_name:      str,
    drug_phase:     Dict[str, Optional[str]],
    drug_rows:      List[Dict],
) -> Optional[Dict]:
    """
    Step 3: Scientific barrier analysis.

    Determines whether the claimed feature solves a real technical barrier
    by consulting regulatory reviews (marketed) or journal literature (clinical).

    Args:
        filename:     Patent PDF filename
        context:      RAG context string
        step1_result: Output from _run_step1
        step2_result: Output from _run_step2
        drug_name:    Drug name string
        drug_phase:   {"US": phase_or_None, "EP": phase_or_None}
        drug_rows:    All Excel rows for this drug

    Returns:
        Parsed JSON dict or None on Gemini failure.
    """
    print(f"[STEP 3] Scientific barrier analysis for {filename}...")

    jurisdiction  = (step1_result.get("jurisdiction") or "").upper()
    phase         = drug_phase.get(jurisdiction) or drug_phase.get("US") or drug_phase.get("EP")
    claim_reason  = step1_result.get("reason") or ""
    claim_cat     = step1_result.get("claim_category") or ""
    matched_elems = step2_result.get("matched_elements") or []
    is_marketed   = (phase or "").lower() == "marketed"

    print(f"[STEP 3] {filename} — Phase: {phase} | Marketed: {is_marketed}")

    # ── Gather evidence in parallel ───────────────────────────────────────────
    evidence_parts: List[str] = []

    if is_marketed:
        print(f"[STEP 3] Marketed drug — fetching FDA reviews, EMA assessment, Europe PMC, open web search...")
        fda_ev, ema_ev, epmc_ev, gem_ev = await asyncio.gather(
            _gather_fda_evidence(drug_name, claim_reason),
            _gather_ema_evidence(drug_name, claim_reason),
            _gather_europepmc_evidence(drug_name),
            _gather_all_gemini_evidence(drug_name, claim_cat, claim_reason),
            return_exceptions=True,
        )

        for label, ev in [
            ("FDA",              fda_ev),
            ("EMA",              ema_ev),
            ("Europe PMC",       epmc_ev),
            ("Open Web Search",  gem_ev),
        ]:
            if isinstance(ev, Exception):
                print(f"[STEP 3] {label} evidence error: {ev}")
            elif ev and str(ev).strip():
                evidence_parts.append(str(ev).strip())

    else:
        print(f"[STEP 3] Clinical drug — fetching PubMed, Europe PMC, open web search...")
        pubmed_ev, epmc_ev, gem_ev = await asyncio.gather(
            _gather_pubmed_evidence(drug_name),
            _gather_europepmc_evidence(drug_name),
            _gather_all_gemini_evidence(drug_name, claim_cat, claim_reason),
            return_exceptions=True,
        )

        for label, ev in [
            ("PubMed",           pubmed_ev),
            ("Europe PMC",       epmc_ev),
            ("Open Web Search",  gem_ev),
        ]:
            if isinstance(ev, Exception):
                print(f"[STEP 3] {label} evidence error: {ev}")
            elif ev and str(ev).strip():
                evidence_parts.append(f"[{label} Evidence]\n{str(ev).strip()}")

        # Add completed trial rows from Excel as additional context
        completed_rows = _get_completed_rows(drug_rows)
        if completed_rows:
            completed_block = _format_rows_for_prompt(completed_rows)
            evidence_parts.append(
                f"[Completed Clinical Trial Formulation Data]\n{completed_block}"
            )

    if not evidence_parts:
        evidence_block = "No scientific evidence could be retrieved. Base assessment on patent claims and drug context."
    else:
        evidence_block = "\n\n" + "─" * 40 + "\n\n".join(evidence_parts)

    # ── Call Gemini for final determination ───────────────────────────────────
    safe_context = context.replace("{", "{{").replace("}", "}}")
    safe_evidence = _cap_evidence(evidence_block).replace("{", "{{").replace("}", "}}")

    prompt = STEP3_PROMPT.format(
        drug_name       = drug_name,
        patent_number   = step1_result.get("patent_number", Path(filename).stem),
        jurisdiction    = jurisdiction,
        claim_category  = claim_cat,
        step1_reason    = claim_reason,
        matched_elements = ", ".join(matched_elems) if matched_elems else "None",
        evidence_block  = safe_evidence,
        context         = safe_context,
    )

    result = await _call_gemini_json(prompt, filename, "STEP3")
    if result is None:
        return None

    # Normalise
    result["is_technical_barrier"] = bool(result.get("is_technical_barrier", False))
    result["confidence"]           = result.get("confidence", "low")
    result["evidence_type"]        = result.get("evidence_type", "Insufficient Evidence")
    result["evidence_summary"]     = result.get("evidence_summary", "")
    result["reason"]               = result.get("reason", "")

    # ── Confidence gate ───────────────────────────────────────────────────────
    # Only high confidence passes as a real technical barrier.
    # medium/low → force NON-BLOCKING regardless of is_technical_barrier value.
    if result["is_technical_barrier"] and result["confidence"] != "high":
        print(
            f"[STEP 3] {filename} → Confidence gate: "
            f"is_technical_barrier=True but confidence={result['confidence']} "
            f"→ overriding to False (insufficient evidence strength)"
        )
        result["is_technical_barrier"] = False
        result["reason"] = (
            f"[Confidence gate: {result['confidence']} confidence is insufficient] "
            + (result.get("reason") or "")
        )

    verdict = "CONTINUE → Step 4" if result["is_technical_barrier"] else "NON-BLOCKING"
    print(
        f"[STEP 3] {filename}\n"
        f"  Barrier     : {result['is_technical_barrier']} ({result['confidence']} confidence)\n"
        f"  Evidence    : {result['evidence_type']}\n"
        f"  Summary     : {result['evidence_summary'][:120]}\n"
        f"  Verdict     : {verdict}"
    )

    # Pass the raw evidence block through so Step 4 can reuse it without re-fetching
    result["_evidence_block"] = evidence_block
    return result


# ─────────────────────────────────────────────
# STEP 4 — Can development proceed without this feature?
# ─────────────────────────────────────────────

STEP4_PROMPT_MARKETED = """You are a pharmaceutical regulatory expert performing Freedom-to-Operate analysis.
You are a STRICT REVIEWER. Your default position is NON-BLOCKING unless evidence is explicit and direct.

DRUG NAME      : {drug_name}
PATENT NUMBER  : {patent_number}
JURISDICTION   : {jurisdiction}
CLAIM CATEGORY : {claim_category}
CLAIM DETAILS  : {step1_reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCIENTIFIC EVIDENCE (gathered in Step 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{evidence_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATENT DOCUMENT CHUNKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────
TASK — STEP 4: CAN DEVELOPMENT PROCEED WITHOUT THIS FEATURE?
─────────────────────────────────────────────

This drug is COMMERCIALLY MARKETED. A generic/biosimilar developer needs to know
whether they can omit or substitute the patented feature and still obtain regulatory approval.

CENTRAL QUESTION:
Would removing or substituting the patented feature cause failure to meet
FDA/EMA regulatory requirements for approval of a generic or biosimilar?

BLOCKING INDICATOR — answer YES only if ALL of the following are true:
1. The FDA or EMA review documents explicitly state this specific feature was
   required for approval (not merely used or preferred)
2. Removing the feature would cause the product to fail a specific regulatory
   standard (stability, bioavailability specification, safety requirement)
3. No approved alternative approach exists that achieves the same regulatory outcome

NOT a blocking indicator if:
- The feature was used in the approved product but not explicitly mandated
- An alternative formulation/device/process could achieve the same regulatory outcome
- The requirement is based on labelling preference rather than regulatory standard
- The evidence only shows the feature improves quality without being required

─────────────────────────────────────────────
STRICT DECISION RULES:
─────────────────────────────────────────────
1. Only explicit regulatory language stating the feature is REQUIRED → YES
2. Implied necessity, strong preference, or common practice → NO
3. If uncertain → NO

OUTPUT — return ONLY valid JSON, no markdown:
{{
  "is_blocking_indicator": true or false,
  "regulatory_failure_if_removed": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "1-2 sentences: specifically what regulatory requirement would fail and why, or why removal is feasible"
}}
"""


STEP4_PROMPT_CLINICAL = """You are a pharmaceutical development expert performing Freedom-to-Operate analysis.
You are a STRICT REVIEWER. Your default position is NON-BLOCKING unless evidence is explicit and direct.

DRUG NAME      : {drug_name}
PATENT NUMBER  : {patent_number}
JURISDICTION   : {jurisdiction}
CLAIM CATEGORY : {claim_category}
CLAIM DETAILS  : {step1_reason}
CLINICAL PHASE : {drug_phase}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCIENTIFIC EVIDENCE (gathered in Step 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{evidence_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATENT DOCUMENT CHUNKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────
TASK — STEP 4: CAN DEVELOPMENT PROCEED WITHOUT THIS FEATURE?
─────────────────────────────────────────────

This drug is in CLINICAL DEVELOPMENT ({drug_phase}). A competing developer needs to know
whether they can omit or substitute the patented feature and proceed without
major redevelopment.

CENTRAL QUESTION:
Would removing or substituting the patented feature require the competitor to
conduct new Phase I studies or PK/PD bridging studies before continuing development?

EVIDENCE TO EXAMINE:
1. Consistency of formulation across clinical phases — has this feature been
   present from Phase 1 through current phase without change? (suggests it is
   integral to the development programme, not just optimised later)
2. Any published statements that this configuration is optimised, required, or
   cannot be substituted without affecting pharmacokinetics or safety
3. Whether substituting the feature would produce a meaningfully different
   PK/PD profile requiring new bridging data

BLOCKING INDICATOR — answer YES only if:
- Published evidence shows the feature has been consistent across ALL phases
  (indicating it is integral, not a late optimisation)
  AND
- Removing it would produce a different PK/PD profile that would require new
  Phase I or bridging studies before Phase 2/3 could proceed

NOT a blocking indicator if:
- The feature was introduced after Phase 1 (optimisation, not necessity)
- An alternative could be substituted with only formulation development work
  (no new clinical studies required)
- The evidence only shows the feature is preferred or convenient
- Publications describe it as one of several viable approaches

─────────────────────────────────────────────
STRICT DECISION RULES:
─────────────────────────────────────────────
1. Consistent across all phases + new bridging studies required → YES
2. Late-stage optimisation, or bridging not required → NO
3. If uncertain → NO

OUTPUT — return ONLY valid JSON, no markdown:
{{
  "is_blocking_indicator": true or false,
  "bridging_studies_required": true or false,
  "formulation_consistent_across_phases": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "1-2 sentences: specifically what evidence supports or refutes the need for new clinical studies"
}}
"""


async def _run_step4(
    filename:     str,
    context:      str,
    step1_result: Dict,
    step2_result: Dict,
    step3_result: Dict,
    drug_name:    str,
    drug_phase:   Dict[str, Optional[str]],
) -> Optional[Dict]:
    """
    Step 4: Can development proceed without this feature?

    Commercial drugs — checks whether removing the feature would cause
    regulatory failure (FDA/EMA requirement not met).

    Clinical drugs — checks whether removing the feature would require
    new Phase I or PK/PD bridging studies.

    Reuses the evidence block already gathered in Step 3 — no new fetching.
    """
    patent_number  = step1_result.get("patent_number", Path(filename).stem)
    jurisdiction   = step1_result.get("jurisdiction", "")
    claim_cat      = step1_result.get("claim_category", "")
    claim_reason   = step1_result.get("reason", "")
    evidence_block = step3_result.get("_evidence_block", "No evidence available from Step 3.")

    # Determine phase for this patent's jurisdiction
    jur_upper = jurisdiction.upper()
    if jur_upper == "US":
        phase = drug_phase.get("US")
    elif jur_upper in ("EP", "EU"):
        phase = drug_phase.get("EP")
    else:
        phase = drug_phase.get("US") or drug_phase.get("EP")

    is_marketed = (phase or "").lower() == "marketed"
    print(f"[STEP 4] {filename} — Phase: {phase} | Marketed: {is_marketed}")

    safe_context  = context.replace("{", "{{").replace("}", "}}")
    safe_evidence = _cap_evidence(evidence_block).replace("{", "{{").replace("}", "}}")

    if is_marketed:
        prompt = STEP4_PROMPT_MARKETED.format(
            drug_name      = drug_name,
            patent_number  = patent_number,
            jurisdiction   = jurisdiction,
            claim_category = claim_cat,
            step1_reason   = claim_reason,
            evidence_block = safe_evidence,
            context        = safe_context,
        )
    else:
        prompt = STEP4_PROMPT_CLINICAL.format(
            drug_name      = drug_name,
            patent_number  = patent_number,
            jurisdiction   = jurisdiction,
            claim_category = claim_cat,
            step1_reason   = claim_reason,
            drug_phase     = phase or "Unknown",
            evidence_block = safe_evidence,
            context        = safe_context,
        )

    result = await _call_gemini_json(prompt, filename, "STEP4")
    if result is None:
        return None

    # Normalise
    result["is_blocking_indicator"] = bool(result.get("is_blocking_indicator", False))
    result["confidence"]            = result.get("confidence", "low")
    result["reason"]                = result.get("reason", "")

    # Confidence gate — only high confidence passes as blocking indicator
    if result["is_blocking_indicator"] and result["confidence"] != "high":
        print(
            f"[STEP 4] {filename} → Confidence gate: "
            f"is_blocking_indicator=True but confidence={result['confidence']} "
            f"→ overriding to False"
        )
        result["is_blocking_indicator"] = False
        result["reason"] = (
            f"[Confidence gate: {result['confidence']} confidence insufficient] "
            + (result.get("reason") or "")
        )

    verdict = "BLOCKING INDICATOR" if result["is_blocking_indicator"] else "NON-BLOCKING"
    print(
        f"[STEP 4] {filename}\n"
        f"  Blocking Indicator : {result['is_blocking_indicator']} ({result['confidence']} confidence)\n"
        f"  Reason             : {result.get('reason', '')[:120]}\n"
        f"  Verdict            : {verdict}"
    )
    return result


async def _run_step5(
    filename:     str,
    context:      str,
    step1_result: Dict,
    step2_result: Dict,
    step3_result: Dict,
    step4_result: Dict,
    drug_name:    str,
) -> Optional[Dict]:
    """
    Step 5: Is the claimed design novel and technically difficult?

    Assesses whether the patented feature is a genuinely innovative and
    technically demanding solution vs. a routine improvement.

    Sources used:
    - Patent specification (RAG context)
    - Scientific/regulatory evidence gathered in Step 3 (reused)

    Final classification:
    - BLOCKING if all four conditions met: product practices claim,
      feature solves necessary technical problem, removal prevents approval
      or requires restarting development, AND is highly novel/difficult.
    - NON-BLOCKING otherwise.
    """
    patent_number  = step1_result.get("patent_number", Path(filename).stem)
    jurisdiction   = step1_result.get("jurisdiction", "")
    claim_cat      = step1_result.get("claim_category", "")
    claim_reason   = step1_result.get("reason", "")
    evidence_block = step3_result.get("_evidence_block", "No evidence available from Step 3.")

    safe_context  = context.replace("{", "{{").replace("}", "}}")
    safe_evidence = _cap_evidence(evidence_block).replace("{", "{{").replace("}", "}}")

    prompt = STEP5_PROMPT.format(
        drug_name      = drug_name,
        patent_number  = patent_number,
        jurisdiction   = jurisdiction,
        claim_category = claim_cat,
        step1_reason   = claim_reason,
        step3_summary  = step3_result.get("evidence_summary", "N/A"),
        step4_reason   = step4_result.get("reason", "N/A"),
        evidence_block = safe_evidence,
        context        = safe_context,
    )

    result = await _call_gemini_json(prompt, filename, "STEP5")
    if result is None:
        return None

    # Normalise
    result["is_novel_and_difficult"] = bool(result.get("is_novel_and_difficult", False))
    result["final_tag"]              = result.get("final_tag", "NON-BLOCKING")
    result["confidence"]             = result.get("confidence", "low")
    result["reason"]                 = result.get("reason", "")

    # Confidence gate — only high confidence can yield BLOCKING
    if result["final_tag"] == "BLOCKING" and result["confidence"] != "high":
        print(
            f"[STEP 5] {filename} → Confidence gate: "
            f"final_tag=BLOCKING but confidence={result['confidence']} "
            f"→ overriding to NON-BLOCKING"
        )
        result["final_tag"] = "NON-BLOCKING"
        result["reason"] = (
            f"[Confidence gate: {result['confidence']} confidence insufficient for BLOCKING] "
            + (result.get("reason") or "")
        )

    print(
        f"[STEP 5] {filename}\n"
        f"  Novel & Difficult : {result['is_novel_and_difficult']}\n"
        f"  Novelty Signal    : {result.get('novelty_signal', '')}\n"
        f"  Confidence        : {result['confidence']}\n"
        f"  Final Tag         : {result['final_tag']}\n"
        f"  Reason            : {result.get('reason', '')[:120]}"
    )
    return result


STEP5_PROMPT = """You are a senior pharmaceutical patent expert making a FINAL blocking classification.
You are a STRICT REVIEWER. Default to NON-BLOCKING unless all four conditions are explicitly met.

DRUG NAME      : {drug_name}
PATENT NUMBER  : {patent_number}
JURISDICTION   : {jurisdiction}
CLAIM CATEGORY : {claim_category}
CLAIM DETAILS  : {step1_reason}

PRIOR STEP FINDINGS:
  Step 3 (Technical Barrier) : {step3_summary}
  Step 4 (Dev. Proceed?)     : {step4_reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCIENTIFIC & REGULATORY EVIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{evidence_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL PATENT DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────
TASK — STEP 5: NOVELTY & TECHNICAL DIFFICULTY ASSESSMENT
─────────────────────────────────────────────

Assess whether the patented feature is a genuinely innovative and technically
demanding solution, then make the FINAL blocking classification.

─────────────────────────────────────────────
PART A — NOVELTY & TECHNICAL DIFFICULTY
─────────────────────────────────────────────

Using the ENTIRE patent document — read every section including the title,
abstract, technical field, background, summary, detailed description, examples,
figures, and all numbered or unnumbered paragraphs — AND the scientific/regulatory
evidence, assess what the patent protects and whether it meets the criteria below.
Do NOT limit your reading to any section labelled "Claims". Understand the full
scope of the invention from all technical content in the document.

1. Does the patent solve a previously UNRESOLVED technical problem?
   (Not just an improvement — a problem that had no prior working solution)

2. Does prior art or the patent's background section describe FAILED ATTEMPTS
   before this solution was found?

3. Is the feature described as proprietary, strategically critical, or protected
   as a core innovation in SEC filings (10-K/20-F) or company disclosures?

4. Does implementation require COMPLEX formulation science, manufacturing
   process controls, or device engineering that is not standard industry practice?

INDICATORS OF HIGH NOVELTY / DIFFICULTY (strengthens BLOCKING):
  ✓ First-in-class solution with no prior working equivalent
  ✓ Addresses core stability, PK/PD, or manufacturability barrier
  ✓ Narrow technical pathway — very few viable alternatives
  ✓ Complex process controls or validation requirements
  ✓ Significant documented R&D investment (SEC filings, publications)
  ✓ Prior art shows failed attempts at solving the same problem

INDICATORS OF LOW NOVELTY / DIFFICULTY (strengthens NON-BLOCKING):
  ✗ Routine optimisation of known parameters
  ✗ Standard excipient selection or device use common in the field
  ✗ Broad, forgiving formulation ranges that others could replicate easily
  ✗ Common industry approach applied to a new context
  ✗ No evidence of prior failed attempts or technical struggle

─────────────────────────────────────────────
PART B — FINAL CLASSIFICATION
─────────────────────────────────────────────

A patent is BLOCKING only if ALL FOUR of the following are true:
  1. The product practices the claim (confirmed in Step 2)
  2. The feature solves a necessary technical problem (confirmed in Step 3)
  3. Removing it would prevent approval or require restarting development (confirmed in Step 4)
  4. The solution is genuinely novel and technically difficult (assessed here in Step 5)

If ANY of these four conditions is not met → NON-BLOCKING.

─────────────────────────────────────────────
STRICT DECISION RULES:
─────────────────────────────────────────────
- Steps 2, 3, 4 are already confirmed for patents reaching Step 5.
  Your job is to assess condition 4 and make the final call.
- High novelty + technical difficulty → BLOCKING
- Routine optimisation, standard practice, or low difficulty → NON-BLOCKING
- If evidence of novelty/difficulty is ambiguous or absent → NON-BLOCKING
- Confidence must be HIGH for a BLOCKING verdict. If uncertain → NON-BLOCKING.

─────────────────────────────────────────────
OUTPUT — return ONLY valid JSON, no markdown:
─────────────────────────────────────────────
{{
  "is_novel_and_difficult":   true or false,
  "novelty_signal":           "high" or "medium" or "low",
  "first_in_class":           true or false,
  "prior_failed_attempts":    true or false,
  "complex_implementation":   true or false,
  "final_tag":                "BLOCKING" or "NON-BLOCKING",
  "blocking_category":        "Composition of Matter" or "Co-formulation/formulation" or "Delivery device required for use" or "Method of treatment claimed broadly" or null,
  "confidence":               "high" or "medium" or "low",
  "reason":                   "2-3 sentences: what makes this novel/not novel, and why the final classification follows"
}}

blocking_category rules:
  - Must be one of the four exact strings above, or null
  - Set to null if final_tag is NON-BLOCKING
  - If BLOCKING, use the claim_category from Step 1 unless the evidence warrants a different category
"""


# ─────────────────────────────────────────────
# Per-file analysis orchestrator
# ─────────────────────────────────────────────

async def _run_step1_only(
    filename:   str,
    collection,
) -> Optional[Dict]:
    """
    Phase 1 helper — runs Step 1 only and returns a dict with everything
    needed to decide CoM routing and then continue to Steps 2+.

    Returns:
        {
          "filename":          str,
          "step1":             dict,       # raw Step 1 Gemini output
          "context":           str,        # RAG context
          "dates":             dict,       # filing_date, grant_date
          "patent_number":     str,        # resolved
          "jurisdiction":      str,        # resolved
          "is_com":            bool,
          "filing_date":       str | None,
        }
        or None on failure.
    """
    async with _ANALYSIS_CONCURRENCY:
        patent_number_hint = Path(filename).stem
        jurisdiction_hint  = extract_jurisdiction(patent_number_hint)

        context = await build_rag_context(collection, filename)
        dates   = get_dates_from_chromadb(collection, filename)

        if not context.strip():
            print(f"[PHASE 1] No RAG chunks for {filename} — skipping")
            return None

        step1 = await _run_step1(filename, context, patent_number_hint, jurisdiction_hint)
        if step1 is None:
            print(f"[PHASE 1] Step 1 failed for {filename}")
            return None

        patent_number = re.sub(r"[\s,]", "", (step1.get("patent_number") or "").strip())
        if not patent_number:
            patent_number = patent_number_hint
        jurisdiction = (step1.get("jurisdiction") or "").strip().upper() or jurisdiction_hint

        step1["patent_number"] = patent_number
        step1["jurisdiction"]  = jurisdiction

        print(
            f"[PHASE 1] {filename} → {step1['claim_category']} | "
            f"{patent_number} | {jurisdiction}"
        )

        return {
            "filename":      filename,
            "step1":         step1,
            "context":       context,
            "dates":         dates,
            "patent_number": patent_number,
            "jurisdiction":  jurisdiction,
            "is_com":        bool(step1.get("is_composition_of_matter")),
            "filing_date":   dates.get("filing_date"),
        }


def _build_com_blocking_result(phase1: Dict) -> Dict:
    """Build the final BLOCKING result for the primary CoM patent."""
    step1 = phase1["step1"]
    dates = phase1["dates"]
    return {
        "patent_number":                  phase1["patent_number"],
        "jurisdiction":                   phase1["jurisdiction"],
        "filing_date":                    dates.get("filing_date"),
        "grant_date":                     dates.get("grant_date"),
        "claim_category":                 "Composition of Matter",
        "tag":                            "BLOCKING",
        "blocking_category":              "Composition of Matter",
        "reason":                         step1.get("reason"),
        "pte":                            step1.get("pte"),
        "pediatric_exclusivity":          bool(step1.get("pediatric_exclusivity", False)),
        "step2_elements_present":                     None,
        "step3_is_technical_barrier":                 None,
        "step3_confidence":                           None,
        "step3_evidence_type":                        None,
        "step3_evidence_summary":                     None,
        "step4_is_blocking_indicator":                None,
        "step4_confidence":                           None,
        "step4_regulatory_failure_if_removed":        None,
        "step4_bridging_studies_required":            None,
        "step4_formulation_consistent_across_phases": None,
        "step4_reason":                               None,
        "estimated_approval_year":        None,
        "exclusivity_year":               None,
        "controlling_patent_expiry_year": None,
        "years_to_entry":                 None,
        "avg_years_to_entry":             None,
        "score":                          None,
        "approval_date_us":               None,
        "approval_date_eu":               None,
        "approval_date_us_source":        None,
        "approval_date_eu_source":        None,
        "source_file":                    phase1["filename"],
    }


async def _run_steps2_plus(
    phase1:     Dict,
    drug_name:  str,
    drug_rows:  List[Dict],
    drug_phase: Dict[str, Optional[str]],
) -> Dict:
    """
    Phase 2 helper — receives Step 1 output and runs Steps 2+ to completion.
    Used for all patents that are NOT the primary CoM for their jurisdiction.
    """
    async with _ANALYSIS_CONCURRENCY:
        filename     = phase1["filename"]
        step1        = phase1["step1"]
        context      = phase1["context"]
        dates        = phase1["dates"]
        patent_number = phase1["patent_number"]
        jurisdiction  = phase1["jurisdiction"]

        print(f"\n[PHASE 2] ── {filename} → Steps 2+ ──")
        print(f"[STEP 1] {filename} → {step1['claim_category']} — proceeding to Step 2...")

        # ── Step 2 ──────────────────────────────────────────────────────────
        step2 = await _run_step2(filename, context, step1, drug_rows)

        if step2 is None:
            print(f"[ANALYSIS] Step 2 failed for {filename}")
            r = error_result(filename)
            r.update(dates)
            return r

        if not step2["any_element_present"]:
            print(
                f"[STEP 2] {filename} → No claim elements present → NON-BLOCKING\n"
                f"         Reason: {(step2.get('reason') or '')[:120]}"
            )
            return {
                "patent_number":                  patent_number,
                "jurisdiction":                   jurisdiction,
                "filing_date":                    dates.get("filing_date"),
                "grant_date":                     dates.get("grant_date"),
                "claim_category":                 step1["claim_category"],
                "tag":                            "NON-BLOCKING",
                "blocking_category":              None,
                "reason":                         step2.get("reason"),
                "pte":                            step1.get("pte"),
                "pediatric_exclusivity":          bool(step1.get("pediatric_exclusivity", False)),
                "step2_elements_present":                     step2.get("elements_present", {}),
                "step3_is_technical_barrier":                 None,
                "step3_confidence":                           None,
                "step3_evidence_type":                        None,
                "step3_evidence_summary":                     None,
                "step4_is_blocking_indicator":                None,
                "step4_confidence":                           None,
                "step4_regulatory_failure_if_removed":        None,
                "step4_bridging_studies_required":            None,
                "step4_formulation_consistent_across_phases": None,
                "step4_reason":                               None,
            "step5_is_novel_and_difficult":               None,
            "step5_novelty_signal":                           None,
            "step5_first_in_class":                           None,
            "step5_prior_failed_attempts":                None,
            "step5_complex_implementation":               None,
            "step5_confidence":                               None,
            "step5_reason":                                       None,
                "estimated_approval_year":        None,
                "exclusivity_year":               None,
                "controlling_patent_expiry_year": None,
                "years_to_entry":                 None,
                "avg_years_to_entry":             None,
                "score":                          None,
                "approval_date_us":               None,
                "approval_date_eu":               None,
                "approval_date_us_source":        None,
                "approval_date_eu_source":        None,
                "source_file":                    filename,
            }

        # ── Step 3 ──────────────────────────────────────────────────────────
        print(
            f"[STEP 2] {filename} → Elements present: {step2['matched_elements']} "
            f"— continuing to Step 3..."
        )

        step3 = await _run_step3(
            filename     = filename,
            context      = context,
            step1_result = step1,
            step2_result = step2,
            drug_name    = drug_name,
            drug_phase   = drug_phase,
            drug_rows    = drug_rows,
        )

        if step3 is None:
            print(f"[ANALYSIS] Step 3 failed for {filename}")
            r = error_result(filename)
            r.update(dates)
            return r

        if not step3["is_technical_barrier"]:
            print(
                f"[STEP 3] {filename} → Not a technical barrier → NON-BLOCKING\n"
                f"         Reason: {(step3.get('reason') or '')[:120]}"
            )
            return {
                "patent_number":                  patent_number,
                "jurisdiction":                   jurisdiction,
                "filing_date":                    dates.get("filing_date"),
                "grant_date":                     dates.get("grant_date"),
                "claim_category":                 step1["claim_category"],
                "tag":                            "NON-BLOCKING",
                "blocking_category":              None,
                "reason":                         step3.get("reason"),
                "pte":                            step1.get("pte"),
                "pediatric_exclusivity":          bool(step1.get("pediatric_exclusivity", False)),
                "step2_elements_present":                     step2.get("elements_present", {}),
                "step3_is_technical_barrier":                 False,
                "step3_confidence":                           step3.get("confidence"),
                "step3_evidence_type":                        step3.get("evidence_type"),
                "step3_evidence_summary":                     step3.get("evidence_summary"),
                "step4_is_blocking_indicator":                None,
                "step4_confidence":                           None,
                "step4_regulatory_failure_if_removed":        None,
                "step4_bridging_studies_required":            None,
                "step4_formulation_consistent_across_phases": None,
                "step4_reason":                               None,
            "step5_is_novel_and_difficult":               None,
            "step5_novelty_signal":                           None,
            "step5_first_in_class":                           None,
            "step5_prior_failed_attempts":                None,
            "step5_complex_implementation":               None,
            "step5_confidence":                               None,
            "step5_reason":                                       None,
                "estimated_approval_year":        None,
                "exclusivity_year":               None,
                "controlling_patent_expiry_year": None,
                "years_to_entry":                 None,
                "avg_years_to_entry":             None,
                "score":                          None,
                "approval_date_us":               None,
                "approval_date_eu":               None,
                "approval_date_us_source":        None,
                "approval_date_eu_source":        None,
                "source_file":                    filename,
            }

        # ── Step 3 pass → Step 4 ─────────────────────────────────────────────
        print(
            f"[STEP 3] {filename} → Real technical barrier confirmed "
            f"({step3['confidence']} confidence) — continuing to Step 4..."
        )

        step4 = await _run_step4(
            filename     = filename,
            context      = context,
            step1_result = step1,
            step2_result = step2,
            step3_result = step3,
            drug_name    = drug_name,
            drug_phase   = drug_phase,
        )

        if step4 is None:
            print(f"[STEP 4] {filename} → Step 4 failed — treating as NON-BLOCKING")
            step4 = {
                "is_blocking_indicator":              False,
                "regulatory_failure_if_removed":      None,
                "bridging_studies_required":          None,
                "formulation_consistent_across_phases": None,
                "confidence":                         "low",
                "reason":                             "Step 4 analysis failed.",
            }

        if not step4["is_blocking_indicator"]:
            print(
                f"[STEP 4] {filename} → Development can proceed without feature → NON-BLOCKING\n"
                f"         Reason: {(step4.get('reason') or '')[:120]}"
            )
            return {
                "patent_number":                          patent_number,
                "jurisdiction":                           jurisdiction,
                "filing_date":                            dates.get("filing_date"),
                "grant_date":                             dates.get("grant_date"),
                "claim_category":                         step1["claim_category"],
                "tag":                                    "NON-BLOCKING",
                "blocking_category":                      None,
                "reason":                                 step4.get("reason"),
                "pte":                                    step1.get("pte"),
                "pediatric_exclusivity":                  bool(step1.get("pediatric_exclusivity", False)),
                "step2_elements_present":                 step2.get("elements_present", {}),
                "step3_is_technical_barrier":             True,
                "step3_confidence":                       step3.get("confidence"),
                "step3_evidence_type":                    step3.get("evidence_type"),
                "step3_evidence_summary":                 step3.get("evidence_summary"),
                "step4_is_blocking_indicator":            False,
                "step4_confidence":                       step4.get("confidence"),
                "step4_regulatory_failure_if_removed":    step4.get("regulatory_failure_if_removed"),
                "step4_bridging_studies_required":        step4.get("bridging_studies_required"),
                "step4_formulation_consistent_across_phases": step4.get("formulation_consistent_across_phases"),
                "step4_reason":                           step4.get("reason"),
                "estimated_approval_year":                None,
                "exclusivity_year":                       None,
                "controlling_patent_expiry_year":         None,
                "years_to_entry":                         None,
                "avg_years_to_entry":                     None,
                "score":                                  None,
                "approval_date_us":                       None,
                "approval_date_eu":                       None,
                "approval_date_us_source":                None,
                "approval_date_eu_source":                None,
                "source_file":                            filename,
            }

        # ── Step 4 pass → blocking indicator confirmed, continue to Step 5 ────
        print(
            f"[STEP 4] {filename} → Blocking indicator confirmed "
            f"({step4['confidence']} confidence) — continuing to Step 5..."
        )

        step5 = await _run_step5(
            filename     = filename,
            context      = context,
            step1_result = step1,
            step2_result = step2,
            step3_result = step3,
            step4_result = step4,
            drug_name    = drug_name,
        )

        if step5 is None:
            print(f"[STEP 5] {filename} → Step 5 failed — treating as NON-BLOCKING")
            step5 = {
                "is_novel_and_difficult": False,
                "novelty_signal":         "low",
                "first_in_class":         False,
                "prior_failed_attempts":  False,
                "complex_implementation": False,
                "final_tag":              "NON-BLOCKING",
                "blocking_category":      None,
                "confidence":             "low",
                "reason":                 "Step 5 analysis failed.",
            }

        final_tag        = step5.get("final_tag", "NON-BLOCKING")
        blocking_cat     = step5.get("blocking_category") if final_tag == "BLOCKING" else None

        print(
            f"[STEP 5] {filename} → FINAL: {final_tag}"
            + (f" | Category: {blocking_cat}" if blocking_cat else "")
        )

        return {
            "patent_number":                          patent_number,
            "jurisdiction":                           jurisdiction,
            "filing_date":                            dates.get("filing_date"),
            "grant_date":                             dates.get("grant_date"),
            "claim_category":                         step1["claim_category"],
            "tag":                                    final_tag,
            "blocking_category":                      blocking_cat,
            "reason":                                 step5.get("reason"),
            "pte":                                    step1.get("pte"),
            "pediatric_exclusivity":                  bool(step1.get("pediatric_exclusivity", False)),
            "step2_elements_present":                 step2.get("elements_present", {}),
            "step3_is_technical_barrier":             True,
            "step3_confidence":                       step3.get("confidence"),
            "step3_evidence_type":                    step3.get("evidence_type"),
            "step3_evidence_summary":                 step3.get("evidence_summary"),
            "step4_is_blocking_indicator":            True,
            "step4_confidence":                       step4.get("confidence"),
            "step4_regulatory_failure_if_removed":    step4.get("regulatory_failure_if_removed"),
            "step4_bridging_studies_required":        step4.get("bridging_studies_required"),
            "step4_formulation_consistent_across_phases": step4.get("formulation_consistent_across_phases"),
            "step4_reason":                           step4.get("reason"),
            "step5_is_novel_and_difficult":           step5.get("is_novel_and_difficult"),
            "step5_novelty_signal":                   step5.get("novelty_signal"),
            "step5_first_in_class":                   step5.get("first_in_class"),
            "step5_prior_failed_attempts":            step5.get("prior_failed_attempts"),
            "step5_complex_implementation":           step5.get("complex_implementation"),
            "step5_confidence":                       step5.get("confidence"),
            "step5_reason":                           step5.get("reason"),
            "estimated_approval_year":                None,
            "exclusivity_year":                       None,
            "controlling_patent_expiry_year":         None,
            "years_to_entry":                         None,
            "avg_years_to_entry":                     None,
            "score":                                  None,
            "approval_date_us":                       None,
            "approval_date_eu":                       None,
            "approval_date_us_source":                None,
            "approval_date_eu_source":                None,
            "source_file":                            filename,
        }


# ─────────────────────────────────────────────
# Per-patent analysis cache (JSON file)
# ─────────────────────────────────────────────
# Stores completed blocking analysis results per patent as JSON files
# so that re-runs only analyse NEW patents.
#
# Structure:
#   <CACHE_DIR>/<drug_name>/
#       <filename>.json        — one file per patent with the full result dict
#
# Simple, human-readable, no database dependency.

_ANALYSIS_CACHE_DIR = Path(
    os.getenv("ANALYSIS_CACHE_DIR", Path(__file__).parent / "analysis_cache")
)


def _drug_cache_dir(drug_name: str) -> Path:
    """Returns the cache directory for a specific drug, creating it if needed."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name.strip().lower())
    d = _ANALYSIS_CACHE_DIR / safe_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_filename(filename: str) -> str:
    """Converts a patent PDF filename to a safe JSON cache filename."""
    stem = Path(filename).stem
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", stem)
    return f"{safe}.json"


def store_patent_analysis(drug_name: str, filename: str, result: Dict) -> None:
    """Store a single patent's completed analysis result as a JSON file."""
    try:
        cache_dir  = _drug_cache_dir(drug_name)
        cache_file = cache_dir / _cache_filename(filename)
        cache_file.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        print(f"[ANALYSIS CACHE] Store failed for {filename}: {e}")


def load_cached_patent_analysis(drug_name: str, filename: str) -> Optional[Dict]:
    """Load a single patent's cached analysis result. Returns None if not cached."""
    try:
        cache_file = _drug_cache_dir(drug_name) / _cache_filename(filename)
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_cached_patents_bulk(drug_name: str, filenames: List[str]) -> Dict[str, Dict]:
    """
    Load cached analysis results for multiple filenames at once.
    Returns {filename: patent_dict} for files that have cached results.
    """
    cache_dir = _drug_cache_dir(drug_name)
    cached    = {}

    for filename in filenames:
        cache_file = cache_dir / _cache_filename(filename)
        try:
            if cache_file.exists():
                patent = json.loads(cache_file.read_text(encoding="utf-8"))
                cached[filename] = patent
        except (json.JSONDecodeError, OSError) as e:
            print(f"[ANALYSIS CACHE] Failed to read cache for {filename}: {e}")
            continue

    return cached


def invalidate_patent_cache(drug_name: str, filename: str) -> None:
    """Remove a single patent's cached analysis."""
    try:
        cache_file = _drug_cache_dir(drug_name) / _cache_filename(filename)
        if cache_file.exists():
            cache_file.unlink()
    except Exception:
        pass


def invalidate_drug_cache(drug_name: str) -> None:
    """Remove ALL cached analysis results for a drug."""
    import shutil as _shutil
    try:
        cache_dir = _drug_cache_dir(drug_name)
        if cache_dir.exists():
            count = len(list(cache_dir.glob("*.json")))
            _shutil.rmtree(cache_dir)
            print(f"[ANALYSIS CACHE] Invalidated {count} cached result(s) for '{drug_name}'")
    except Exception as e:
        print(f"[ANALYSIS CACHE] Invalidation failed for '{drug_name}': {e}")


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

async def run_blocking_analysis(
    drug_name:  str,
    pdf_refs:   List[dict],
    collection,
    drug_phase: Optional[Dict[str, Optional[str]]] = None,
    force_reanalyse: bool = False,
) -> List[Dict]:
    """
    Two-phase blocking analysis for all US/EP patents — with incremental caching.

    On first run: analyses all patents and caches each result.
    On subsequent runs: loads cached results for already-analysed patents,
    only sends NEW patents through the Gemini Steps 1-5 pipeline, then
    merges cached + fresh results for CoM routing and final output.

    Phase 1 — Run Step 1 on NEW patents in parallel to classify claim categories.
              Load cached Step 1 data for already-analysed patents.
    Between  — Identify the ONE primary CoM per jurisdiction (earliest filing date)
               across ALL patents (cached + new).
               Primary CoM → BLOCKING immediately, skips Steps 2+.
               All other patents (including secondary CoMs) → Phase 2.
    Phase 2  — Run Steps 2+ in parallel on NEW non-primary-CoM patents.
               Cached non-CoM patents use their stored results directly.

    Args:
        drug_name:        Drug name string (from GCS folder name)
        pdf_refs:         List of {"filename": str, ...} from gcs_lister
        collection:       AlloyDB collection
        drug_phase:       {"US": phase, "EP": phase} — from clinical timeline.
        force_reanalyse:  If True, ignore cache and re-analyse everything.
    """
    all_filenames  = [ref["filename"] for ref in pdf_refs]
    analysis_files = [f for f in all_filenames if not is_non_analysable_patent(f)]
    skipped_files  = [f for f in all_filenames if     is_non_analysable_patent(f)]

    for f in skipped_files:
        print(f"[SKIP ANALYSIS] {f} — non-US/EP patent")

    # ── Load cached results for already-analysed patents ──────────────────────
    cached_results: Dict[str, Dict] = {}
    new_files:      List[str]       = []

    if not force_reanalyse:
        cached_results = load_cached_patents_bulk(drug_name, analysis_files)
        new_files      = [f for f in analysis_files if f not in cached_results]

        if cached_results:
            print(
                f"\n[CACHE] {len(cached_results)} patent(s) loaded from cache, "
                f"{len(new_files)} new patent(s) to analyse"
            )
            for f in cached_results:
                tag = cached_results[f].get("tag", "?")
                cat = cached_results[f].get("claim_category", "?")
                print(f"  [CACHED] {f} → {tag} ({cat})")
        else:
            print(f"\n[CACHE] No cached results — analysing all {len(analysis_files)} patent(s)")
            new_files = analysis_files
    else:
        print(f"\n[CACHE] Force re-analyse — ignoring cache for all {len(analysis_files)} patent(s)")
        invalidate_drug_cache(drug_name)
        new_files = analysis_files

    # If everything is cached, skip the expensive Gemini calls entirely
    if not new_files:
        print(f"[CACHE] All patents cached — skipping Gemini analysis entirely")
        patents = list(cached_results.values())
        for filename in skipped_files:
            patents.append(skipped_result(filename))
        _print_summary_table(drug_name, patents)
        return patents

    drug_rows = get_drug_rows(drug_name)
    if drug_rows:
        print(f"[STEP 2] {len(drug_rows)} Excel record(s) ready for '{drug_name}'")
    else:
        print(f"[STEP 2] No Excel data for '{drug_name}' — Step 2 will assess from patent claims alone")

    if drug_phase:
        print(f"[STEP 3] Phase info ready for '{drug_name}': {drug_phase}")
    else:
        print(f"[STEP 3] No phase info for '{drug_name}' — defaulting to clinical path")
        drug_phase = {}

    # ── Phase 1: Step 1 on NEW patents in parallel ────────────────────────────
    print(f"\n[PHASE 1] Running Step 1 on {len(new_files)} NEW patent(s)...")

    new_phase1_results = await asyncio.gather(
        *[_run_step1_only(f, collection) for f in new_files],
        return_exceptions=True,
    )

    # Build a unified phase1 view: cached patents contribute their stored
    # step1 data (claim_category, jurisdiction, filing_date, is_com) so
    # CoM routing considers ALL patents, not just new ones.
    all_phase1: List[Dict] = []

    # Add cached patents as phase1-style dicts for CoM routing
    for filename, cached_patent in cached_results.items():
        all_phase1.append({
            "filename":      filename,
            "patent_number": cached_patent.get("patent_number", Path(filename).stem),
            "jurisdiction":  (cached_patent.get("jurisdiction") or "").upper(),
            "is_com":        cached_patent.get("claim_category") == "Composition of Matter"
                             and cached_patent.get("tag") == "BLOCKING",
            "filing_date":   cached_patent.get("filing_date"),
            "_from_cache":   True,
        })

    # Add new patents' phase1 results
    for filename, result in zip(new_files, new_phase1_results):
        if isinstance(result, Exception) or result is None:
            all_phase1.append({"filename": filename, "_failed": True})
        else:
            result["_from_cache"] = False
            all_phase1.append(result)

    # ── Identify primary CoM per jurisdiction (across ALL patents) ────────────
    primary_com_filenames: set = set()

    all_jurisdictions = sorted(set(
        r.get("jurisdiction") for r in all_phase1
        if isinstance(r, dict) and r.get("jurisdiction") and not r.get("_failed")
    ))
    print(f"[CoM ROUTING] Jurisdictions found: {all_jurisdictions}")

    for jurisdiction in all_jurisdictions:
        com_candidates = [
            r for r in all_phase1
            if isinstance(r, dict)
            and not r.get("_failed")
            and r.get("is_com")
            and r.get("jurisdiction") == jurisdiction
        ]
        if not com_candidates:
            continue

        com_candidates.sort(key=lambda r: r.get("filing_date") or "9999-99-99")
        primary = com_candidates[0]
        primary_com_filenames.add(primary["filename"])

        print(f"\n[CoM ROUTING] Primary CoM for {jurisdiction}: "
              f"{primary.get('patent_number', '?')} (filed: {primary.get('filing_date') or 'unknown'})"
              f" → BLOCKING (skips Steps 2+)")

        for secondary in com_candidates[1:]:
            print(f"[CoM ROUTING] Secondary CoM for {jurisdiction}: "
                  f"{secondary.get('patent_number', '?')} → sent to Steps 2+ as Formulation-class")

    # ── Build results: merge cached results + route new patents ───────────────
    patents: List[Dict] = []
    phase2_inputs: List[Dict] = []

    # Add cached results directly (they already went through full analysis)
    for filename, cached_patent in cached_results.items():
        # If a cached patent is now the primary CoM (e.g. a new patent changed routing),
        # we still use its cached result — CoM routing is stable for existing patents.
        patents.append(cached_patent)
        print(f"[RESULT] {filename} → from cache ({cached_patent.get('tag', '?')})")

    # Route new patents
    for filename, result in zip(new_files, new_phase1_results):
        if isinstance(result, Exception) or result is None:
            print(f"[ERROR] Phase 1 failed for {filename}: {result}")
            patents.append(error_result(filename))
            continue

        if filename in primary_com_filenames:
            com_result = _build_com_blocking_result(result)
            patents.append(com_result)
            store_patent_analysis(drug_name, filename, com_result)
            print(f"[RESULT] {filename} → NEW primary CoM BLOCKING (cached)")
        else:
            if result.get("step1", {}).get("claim_category") == "Composition of Matter":
                result["step1"]["claim_category"] = "Formulation"
                result["step1"]["is_composition_of_matter"] = False
                print(
                    f"[CoM ROUTING] {result.get('patent_number')} reclassified: "
                    f"Composition of Matter → Formulation"
                )
            phase2_inputs.append(result)

    # ── Phase 2: Steps 2+ on NEW non-primary-CoM patents in parallel ──────────
    if phase2_inputs:
        print(f"\n[PHASE 2] Running Steps 2+ on {len(phase2_inputs)} NEW patent(s)...")

        phase2_results = await asyncio.gather(
            *[_run_steps2_plus(p, drug_name, drug_rows, drug_phase)
              for p in phase2_inputs],
            return_exceptions=True,
        )

        for phase1_data, result in zip(phase2_inputs, phase2_results):
            if isinstance(result, Exception):
                print(f"[ERROR] Phase 2 failed for {phase1_data['filename']}: {result}")
                patents.append(error_result(phase1_data["filename"]))
            else:
                patents.append(result)
                store_patent_analysis(drug_name, phase1_data["filename"], result)
                print(f"[RESULT] {phase1_data['filename']} → NEW {result.get('tag', '?')} (cached)")

    # ── Add skipped patents ───────────────────────────────────────────────────
    for filename in skipped_files:
        patents.append(skipped_result(filename))

    _print_summary_table(drug_name, patents)
    return patents


def _print_summary_table(drug_name: str, patents: List[Dict]) -> None:
    """Prints the analysis summary table. Shared by cached and fresh paths."""
    col_pn = max((len(p.get("patent_number") or "") for p in patents), default=14)
    col_pn = max(col_pn, 14)
    col_s1 = 26
    col_s2 = 44
    col_s3 = 36

    header = (
        f"{'Patent Number':<{col_pn}}  "
        f"{'Step 1 Category':<{col_s1}}  "
        f"{'Step 2 Matched Elements':<{col_s2}}  "
        f"{'Step 3 Scientific Barrier':<{col_s3}}"
    )
    divider = "-" * len(header)

    print(f"\n[SUMMARY] ── {drug_name} ──")
    print(divider)
    print(header)
    print(divider)

    for p in patents:
        patent_num = p.get("patent_number") or ""
        tag        = p.get("tag") or ""

        if tag == "SKIPPED":
            s1, s2, s3 = "SKIPPED", "—", "—"
        else:
            s1 = p.get("claim_category") or "—"

            elements = p.get("step2_elements_present")
            if elements is None:
                s2 = "N/A (primary CoM → BLOCKING)"
            else:
                matched = [k for k, v in elements.items() if v]
                s2 = ", ".join(matched) if matched else "None matched"

            barrier = p.get("step3_is_technical_barrier")
            if barrier is None:
                s3 = "N/A (stopped earlier)"
            else:
                s3 = (
                    f"{'YES' if barrier else 'NO'} "
                    f"[{p.get('step3_confidence') or ''}] "
                    f"{p.get('step3_evidence_type') or ''}"
                )

        print(
            f"{patent_num:<{col_pn}}  "
            f"{s1:<{col_s1}}  "
            f"{s2:<{col_s2}}  "
            f"{s3:<{col_s3}}"
        )

    print(divider)
    print()
