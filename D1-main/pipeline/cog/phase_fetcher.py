"""
phase_fetcher.py
─────────────────
Handles:
  - Fetching clinical development stage from BigQuery (clinical_efficacy table)
  - Loading fallback phase data from a local Excel sheet
  - Merging BQ + fallback stages per jurisdiction (US / EP)
  - Assigning phase_at_filing to each patent dict
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

BQ_TABLE_NAME      = os.getenv("BQ_TABLE_NAME")
BQ_PROJECT_ID      = os.getenv("BQ_PROJECT_ID")
BQ_DATASET_ID      = os.getenv("BQ_DATASET_ID")
BQ_SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

PHASE_FALLBACK_EXCEL = Path(
    os.getenv("PHASE_FALLBACK_EXCEL", Path(__file__).parent / "phase_fallback.xlsx")
)

_TIMELINE_STAGES = [
    "Preclinical", "Phase 1", "Phase 2", "Phase 3", "Pre-registration", "Marketed"
]
_STAGE_RANK: Dict[str, int] = {s: i for i, s in enumerate(_TIMELINE_STAGES)}

if BQ_PROJECT_ID:
    print(f"[BQ] Config loaded: {BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_NAME}")
else:
    print("[BQ] No BigQuery config found — phase will be unavailable")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", name.lower().strip())


# ─────────────────────────────────────────────
# Drug alias map
# ─────────────────────────────────────────────
# Maps known salt forms / aliases to a single canonical INN name.
# Used by canonicalise_drug_name() which is called at the top of the
# pipeline so every downstream lookup uses the same canonical name.

_DRUG_ALIASES: Dict[str, str] = {
    "aleniglipron l-arginine": "aleniglipron",
    "aleniglipron l arginine": "aleniglipron",
    "aleniglipronlarginine":   "aleniglipron",   # normalised form
}


def canonicalise_drug_name(drug_name: str) -> str:
    """
    Returns the canonical INN for a drug name, resolving known salt forms
    and aliases to a single name.

    Steps:
      1. Exact lowercase match in alias map
      2. Normalised match (strip spaces/hyphens/underscores)
      3. No match -> return original name unchanged (preserving original casing)
    """
    lower = drug_name.strip().lower()

    # Exact lowercase match
    if lower in _DRUG_ALIASES:
        canonical = _DRUG_ALIASES[lower]
        print(f"[ALIAS] '{drug_name}' -> '{canonical}' (exact alias)")
        return canonical

    # Normalised match
    norm = _normalize(drug_name)
    for alias_key, canonical in _DRUG_ALIASES.items():
        if _normalize(alias_key) == norm:
            print(f"[ALIAS] '{drug_name}' -> '{canonical}' (normalised alias)")
            return canonical

    return drug_name


def _highest_phase(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Returns whichever phase is further along. None is lower than any real stage."""
    if a is None:
        return b
    if b is None:
        return a
    return a if _STAGE_RANK.get(a, -1) >= _STAGE_RANK.get(b, -1) else b


# ─────────────────────────────────────────────
# BigQuery fetch
# ─────────────────────────────────────────────

def import_from_gbq(
    drug_name:            str,
    table_name:           str,
    project_id:           str,
    dataset_id:           str,
    service_account_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches clinical phase data from the `clinical_efficacy` table.

    Column mapping:
        molecule_name   → cleaned_generic_name
        trial_location  → Drug_Geography  (comma-separated countries, split into rows)
        phase           → highest_development_stage

    Phase normalisation:
        1, 1a, 1b, Phase 1, Phase I  → 'Phase I'
        2, 2a, 2b, Phase 2, Phase II → 'Phase II'
        3, 3a, 3b, Phase 3, Phase III→ 'Phase III'
        4, Phase 4                   → 'Marketed'
    """
    try:
        if service_account_path:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            client = bigquery.Client(project=project_id)

        fq_table = f"{project_id}.{dataset_id}.{table_name}"

        query = f"""
        WITH filtered AS (
          SELECT
            molecule_name                                AS cleaned_generic_name,
            TRIM(trial_location_part)                    AS Drug_Geography,
            CASE
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^4')   THEN 'Marketed'
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^iii') THEN 'Phase III'
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^3')   THEN 'Phase III'
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^ii')  THEN 'Phase II'
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^2')   THEN 'Phase II'
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^i')   THEN 'Phase I'
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^1')   THEN 'Phase I'
              ELSE TRIM(phase)
            END                                          AS highest_development_stage,
            CASE
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^4')   THEN 5
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^iii') THEN 3
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^3')   THEN 3
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^ii')  THEN 2
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^2')   THEN 2
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^i')   THEN 1
              WHEN REGEXP_CONTAINS(LOWER(TRIM(phase)), r'^1')   THEN 1
              ELSE 0
            END                                          AS stage_rank
          FROM `{fq_table}`,
          UNNEST(SPLIT(trial_location, ',')) AS trial_location_part
          WHERE LOWER(REGEXP_REPLACE(
                  COALESCE(molecule_name, ''),
                  r'[\s\-_]+', ''
                )) = LOWER(REGEXP_REPLACE(@drug_name, r'[\s\-_]+', ''))
        ),
        ranked AS (
          SELECT
            cleaned_generic_name,
            Drug_Geography,
            highest_development_stage,
            ROW_NUMBER() OVER (
              PARTITION BY cleaned_generic_name, Drug_Geography
              ORDER BY stage_rank DESC
            ) AS rn
          FROM filtered
        )
        SELECT
          cleaned_generic_name,
          Drug_Geography,
          highest_development_stage
        FROM ranked
        WHERE rn = 1
        ORDER BY Drug_Geography
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name),
            ]
        )

        df = client.query(query, job_config=job_config).to_dataframe()
        print(f"[BQ] Fetched {len(df)} row(s) from BigQuery")
        for _, row in df.head(5).iterrows():
            print(
                f"[BQ]   {row.get('cleaned_generic_name')} | "
                f"{row.get('Drug_Geography')} | "
                f"{row.get('highest_development_stage')}"
            )
        return df

    except Exception as e:
        print(f"[BQ] Query failed: {e}")
        return pd.DataFrame()


def _match_drug_name_in_bq(drug_name: str, df: pd.DataFrame) -> Dict[str, str]:
    """Strict matching only — exact or normalised. No substring matching."""
    if df.empty or "cleaned_generic_name" not in df.columns:
        return {}

    drug_norm    = _normalize(drug_name)
    matched_rows = []

    for _, row in df.iterrows():
        bq_name = str(row["cleaned_generic_name"] or "")
        bq_norm = _normalize(bq_name)
        if bq_name.lower() == drug_name.lower():
            print(f"[BQ MATCH] Exact: '{drug_name}' → '{bq_name}'")
            matched_rows.append(row)
        elif bq_norm == drug_norm:
            print(f"[BQ MATCH] Normalised: '{drug_name}' → '{bq_name}'")
            matched_rows.append(row)

    if not matched_rows:
        print(f"[BQ MATCH] No match found for '{drug_name}'")
        return {}

    geography_stages: Dict[str, str] = {}
    _US_GEOS = {"united states", "us", "usa", "united states of america"}
    _EU_GEOS = {"eu", "europe", "european union"}

    for row in matched_rows:
        raw_geo = str(row.get("Drug_Geography") or "")
        stage   = str(row.get("highest_development_stage") or "")
        geos    = [g.strip() for g in re.split(r"[,;]", raw_geo) if g.strip()]
        for geo in geos:
            geo_lower = geo.lower()
            if geo_lower in _US_GEOS:
                canonical = "United States"
            elif geo_lower in _EU_GEOS:
                canonical = "EU"
            else:
                continue
            if canonical not in geography_stages:
                geography_stages[canonical] = stage
                print(f"[BQ MATCH] Geography: '{geo}' → '{canonical}' | Stage: {stage}")

    return geography_stages


def _bq_stage_to_timeline(geography_stages: Dict[str, str], drug_name: str) -> Dict:
    _BQ_STAGE_MAP = {
        "marketed":         "Marketed",
        "pre-registration": "Pre-registration",
        "phase iii":        "Phase 3",
        "phase ii":         "Phase 2",
        "phase i":          "Phase 1",
        "preclinical":      "Preclinical",
    }

    mapped: Dict[str, str] = {}
    for geo, raw_stage in geography_stages.items():
        internal = _BQ_STAGE_MAP.get(raw_stage.lower().strip(), "Preclinical")
        if internal not in _TIMELINE_STAGES:
            internal = "Preclinical"
        mapped[geo] = internal
        print(f"[BQ TIMELINE] {drug_name} | {geo}: '{raw_stage}' → '{internal}'")

    current_stage = (
        max(mapped.values(), key=lambda s: _STAGE_RANK.get(s, 0))
        if mapped else "Preclinical"
    )
    current_idx = _TIMELINE_STAGES.index(current_stage)
    stage_years = {s: None for s in _TIMELINE_STAGES}

    print(f"[BQ TIMELINE] {drug_name} → Overall: '{current_stage}' | Per-geography: {mapped}")

    return {
        "current_stage":    current_stage,
        "all_stages":       _TIMELINE_STAGES,
        "completed_stages": _TIMELINE_STAGES[: current_idx + 1],
        "stage_years":      stage_years,
        "geography_stages": mapped,
        "notes":            f"Stage sourced from BigQuery. Per-geography: {mapped}",
        "source":           "bigquery",
        "drug_name":        drug_name,
    }


async def fetch_clinical_timeline(
    drug_name:          str,
    bq_table_name:      Optional[str] = None,
    bq_project_id:      Optional[str] = None,
    bq_dataset_id:      Optional[str] = None,
    bq_service_account: Optional[str] = None,
) -> Dict:
    """
    Queries BigQuery for clinical stage data, then immediately merges
    the fallback Excel so geography_stages already reflects the highest
    phase per jurisdiction from both sources.

    Returns:
        Timeline dict with keys: current_stage, geography_stages, source, drug_name, …
        geography_stages is already the fully merged result (BQ + fallback Excel).
    """
    # ── Canonicalise alias -> INN before any lookup ───────────────────────────
    drug_name = canonicalise_drug_name(drug_name)

    # ─────────────────────────────────────────────────────────────────────────
    empty = {
        "current_stage":    None,
        "all_stages":       _TIMELINE_STAGES,
        "completed_stages": [],
        "stage_years":      {s: None for s in _TIMELINE_STAGES},
        "geography_stages": {},
        "notes":            None,
        "source":           "unavailable",
        "drug_name":        drug_name,
    }

    _bq_table   = bq_table_name      or BQ_TABLE_NAME
    _bq_project = bq_project_id      or BQ_PROJECT_ID
    _bq_dataset = bq_dataset_id      or BQ_DATASET_ID
    _bq_sa      = bq_service_account or BQ_SERVICE_ACCOUNT

    # ── Step 1: BigQuery ──────────────────────────────────────────────────────
    bq_geography: Dict[str, Optional[str]] = {"United States": None, "EU": None}

    if _bq_table and _bq_project and _bq_dataset:
        print(f"[TIMELINE] Querying BigQuery for '{drug_name}'...")
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(
                None,
                lambda: import_from_gbq(
                    drug_name            = drug_name,
                    table_name           = _bq_table,
                    project_id           = _bq_project,
                    dataset_id           = _bq_dataset,
                    service_account_path = _bq_sa,
                ),
            )
            matched = _match_drug_name_in_bq(drug_name, df)
            # matched keys are "United States" / "EU"
            for k, v in matched.items():
                bq_geography[k] = v
            print(f"[TIMELINE] BQ → {bq_geography}")
        except Exception as e:
            print(f"[TIMELINE] BigQuery error: {e}")
    else:
        print(f"[TIMELINE] No BigQuery config — skipping BQ lookup")

    # ── Step 2: Fallback Excel ────────────────────────────────────────────────
    fallback_geography: Dict[str, Optional[str]] = {"United States": None, "EU": None}

    fallback_df = load_fallback_phase_excel()
    if fallback_df is not None:
        raw = lookup_fallback_phase(drug_name, fallback_df)
        # lookup_fallback_phase returns {"US": ..., "EP": ...} — remap to canonical keys
        fallback_geography["United States"] = raw.get("US")
        fallback_geography["EU"]            = raw.get("EP")
        print(f"[TIMELINE] Fallback Excel → {fallback_geography}")
    else:
        print(f"[TIMELINE] Fallback Excel not available")

    # ── Step 3: Merge — highest phase per jurisdiction ────────────────────────
    merged: Dict[str, Optional[str]] = {}
    for geo in ("United States", "EU"):
        bq_val = bq_geography[geo]
        fb_val = fallback_geography[geo]
        merged[geo] = _highest_phase(bq_val, fb_val)

        if merged[geo] is None:
            winner = "None available"
        elif bq_val == fb_val:
            winner = "Both equal"
        elif merged[geo] == bq_val:
            winner = "BQ"
        else:
            winner = "Fallback (higher stage)"
        print(
            f"[TIMELINE MERGE] {geo} → BQ={bq_val!r} | Fallback={fb_val!r} "
            f"→ {merged[geo]!r} ({winner})"
        )

    # ── Build timeline dict from merged result ────────────────────────────────
    if not any(merged.values()):
        print(f"[TIMELINE] No phase data found from BQ or fallback for '{drug_name}'")
        return empty

    # Map to internal stage names
    _BQ_STAGE_MAP = {
        "marketed":         "Marketed",
        "pre-registration": "Pre-registration",
        "phase iii":        "Phase 3",
        "phase ii":         "Phase 2",
        "phase i":          "Phase 1",
        "preclinical":      "Preclinical",
    }
    _FALLBACK_STAGE_MAP = {
        "marketed": "Marketed", "pre-registration": "Pre-registration",
        "preregistration": "Pre-registration",
        "phase iii": "Phase 3", "phase 3": "Phase 3", "phase3": "Phase 3",
        "3": "Phase 3", "iii": "Phase 3",
        "phase ii": "Phase 2", "phase 2": "Phase 2", "phase2": "Phase 2",
        "2": "Phase 2", "ii": "Phase 2",
        "phase i": "Phase 1", "phase 1": "Phase 1", "phase1": "Phase 1",
        "1": "Phase 1", "i": "Phase 1",
        "preclinical": "Preclinical",
    }

    normalised: Dict[str, Optional[str]] = {}
    for geo, stage in merged.items():
        if stage is None:
            normalised[geo] = None
            continue
        s = stage.lower().strip()
        # Try BQ map first, then fallback map
        internal = _BQ_STAGE_MAP.get(s) or _FALLBACK_STAGE_MAP.get(s) or stage
        if internal not in _TIMELINE_STAGES:
            internal = stage  # keep as-is if already normalised
        normalised[geo] = internal
        print(f"[TIMELINE] {geo}: '{stage}' → '{internal}'")

    current_stage = (
        max(
            (v for v in normalised.values() if v),
            key=lambda s: _STAGE_RANK.get(s, 0),
        )
        if any(normalised.values()) else "Preclinical"
    )
    current_idx = _TIMELINE_STAGES.index(current_stage) if current_stage in _TIMELINE_STAGES else 0

    source = "bigquery+fallback" if any(bq_geography.values()) and any(fallback_geography.values()) \
             else "bigquery" if any(bq_geography.values()) \
             else "fallback"

    print(f"[TIMELINE] '{drug_name}' → Overall: '{current_stage}' | Per-geo: {normalised} | Source: {source}")

    return {
        "current_stage":    current_stage,
        "all_stages":       _TIMELINE_STAGES,
        "completed_stages": _TIMELINE_STAGES[: current_idx + 1],
        "stage_years":      {s: None for s in _TIMELINE_STAGES},
        "geography_stages": normalised,   # fully merged — United States / EU keys
        "notes":            f"Stage from {source}. Per-geography: {normalised}",
        "source":           source,
        "drug_name":        drug_name,
    }


# ─────────────────────────────────────────────
# Fallback Excel
# ─────────────────────────────────────────────

def load_fallback_phase_excel() -> Optional[pd.DataFrame]:
    if not PHASE_FALLBACK_EXCEL.exists():
        print(f"[PHASE FALLBACK] File not found: {PHASE_FALLBACK_EXCEL}")
        return None
    try:
        df = pd.read_excel(PHASE_FALLBACK_EXCEL)
        df.columns = [c.strip() for c in df.columns]
        print(f"[PHASE FALLBACK] Loaded {len(df)} rows from {PHASE_FALLBACK_EXCEL.name}")
        print(f"[PHASE FALLBACK] Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"[PHASE FALLBACK] Failed to load: {e}")
        return None


def lookup_fallback_phase(drug_name: str, df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Looks up US/EP phase from the fallback Excel sheet.

    Expected columns: Drug / Molecule, Phase, Primary Country 1 (USA or EU).

    Returns:
        {"US": phase_or_None, "EP": phase_or_None}
    """
    result  = {"US": None, "EP": None}
    col_map = {c.strip().lower(): c for c in df.columns}

    mol_col      = next((col_map[k] for k in col_map if k in ("drug", "molecule", "drug name")), None)
    phase_col    = next((col_map[k] for k in col_map if k == "phase"), None)
    country1_col = next((col_map[k] for k in col_map if k == "primary country 1"), None)
    country_col  = next((col_map[k] for k in col_map if k == "primary country"), None)
    geo_col      = country1_col or country_col

    if not mol_col or not phase_col or not geo_col:
        print(f"[PHASE FALLBACK] Missing required columns.")
        print(f"[PHASE FALLBACK] Need: Drug/Molecule, Phase, Primary Country 1 (or Primary Country)")
        print(f"[PHASE FALLBACK] Found: {list(df.columns)}")
        return result

    print(f"[PHASE FALLBACK] Using columns → Drug='{mol_col}' | Phase='{phase_col}' | Geo='{geo_col}'")

    drug_norm   = _normalize(drug_name)
    drug_simple = drug_name.strip().lower().replace("_", " ")

    _FALLBACK_STAGE_MAP = {
        "marketed": "Marketed", "pre-registration": "Pre-registration",
        "preregistration": "Pre-registration",
        "phase iii": "Phase 3", "phase 3": "Phase 3", "phase3": "Phase 3",
        "3": "Phase 3", "iii": "Phase 3",
        "phase ii": "Phase 2", "phase 2": "Phase 2", "phase2": "Phase 2",
        "2": "Phase 2", "ii": "Phase 2",
        "phase i": "Phase 1", "phase 1": "Phase 1", "phase1": "Phase 1",
        "1": "Phase 1", "i": "Phase 1",
        "preclinical": "Preclinical",
    }

    matched_rows = 0
    for _, row in df.iterrows():
        raw_mol    = str(row.get(mol_col) or "")
        mol_norm   = _normalize(raw_mol)
        mol_simple = raw_mol.strip().lower().replace("_", " ")

        if mol_norm != drug_norm and mol_simple != drug_simple:
            continue

        matched_rows += 1
        match_type = "normalised" if mol_norm == drug_norm else "simple"
        print(
            f"[PHASE FALLBACK] Matched ({match_type}): "
            f"Excel='{raw_mol.strip()}' ↔ Pipeline='{drug_name}'"
        )

        phase = str(row.get(phase_col) or "").strip()
        geo   = str(row.get(geo_col)   or "").strip().lower()

        if not phase or phase.lower() in ("nan", "none", ""):
            continue
        if not geo or geo in ("nan", "none", ""):
            continue

        phase = _FALLBACK_STAGE_MAP.get(phase.strip().lower(), phase)

        if geo in ("usa", "us", "united states"):
            if result["US"] is None:
                result["US"] = phase
                print(f"[PHASE FALLBACK] '{drug_name}' | US → {phase}")
        elif geo in ("eu", "europe", "european union"):
            if result["EP"] is None:
                result["EP"] = phase
                print(f"[PHASE FALLBACK] '{drug_name}' | EP → {phase}")
        else:
            print(f"[PHASE FALLBACK] '{drug_name}' | Ignoring geo: {row.get(geo_col)!r}")

    if matched_rows == 0:
        print(f"[PHASE FALLBACK] NO MATCH for '{drug_name}'")
        all_names = df[mol_col].dropna().str.strip().unique().tolist()
        print(f"[PHASE FALLBACK] All drug names in sheet: {all_names}")

    return result


# ─────────────────────────────────────────────
# Phase assignment (year-based from fallback Excel)
# ─────────────────────────────────────────────

def _parse_phase_number(phase_str: str) -> Optional[float]:
    """
    Parses a phase string into a numeric rank for comparison.
    Handles: 'Phase 1', 'Phase 2', 'Phase 3', '3a', '3b', 'Marketed',
             'Pre-registration', 'Preclinical', 'I', 'II', 'III', etc.
    Sub-phases like 3a/3b are treated as 3.
    Phase 4 is treated as Marketed (5.0).
    Returns None if unparseable.
    """
    s = str(phase_str).strip().lower()
    if not s or s in ("nan", "none", ""):
        return None

    if s in ("marketed", "launched", "approved"):
        return 5.0
    if s in ("pre-registration", "preregistration", "pre registration", "nda/bla", "nda", "bla"):
        return 4.0
    if s in ("preclinical", "pre-clinical", "discovery"):
        return 0.0

    # Roman numeral mapping
    _roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

    # Strip "phase" prefix
    cleaned = re.sub(r"^phase\s*", "", s).strip()

    # Phase 4 → Marketed
    if re.match(r"^4", cleaned):
        return 5.0

    # Try roman numerals (e.g. "III", "IIa")
    roman_match = re.match(r"^(i{1,3}v?)\s*[a-z]?\s*$", cleaned, re.IGNORECASE)
    if roman_match:
        roman = roman_match.group(1).lower()
        return float(_roman.get(roman, 0))

    # Try numeric (e.g. "3", "3a", "3b", "2/3")
    num_match = re.match(r"^(\d)", cleaned)
    if num_match:
        num = int(num_match.group(1))
        # Phase 4 → Marketed rank
        if num >= 4:
            return 5.0
        return float(num)

    return None


def _phase_number_to_label(num: float) -> str:
    """Converts a numeric phase rank back to a display label."""
    _map = {
        0.0: "Preclinical",
        1.0: "Phase 1",
        2.0: "Phase 2",
        3.0: "Phase 3",
        4.0: "Pre-registration",
        5.0: "Marketed",
    }
    return _map.get(num, f"Phase {int(num)}")


def _build_phase_year_lookup(
    drug_name: str,
    df: pd.DataFrame,
) -> Dict[str, Dict[int, str]]:
    """
    Builds a lookup: {jurisdiction: {year: highest_phase_label}} from the
    fallback Excel that has columns: Drug, Phase, Primary Country, Start Year.

    Primary Country may contain multiple countries comma-separated (e.g. "USA,EU,Canada").
    Jurisdiction mapping:
        USA / US / United States  → US
        EU / Europe / European Union / any EU country name  → EP

    Phase sub-phases (3a, 3b) are treated as their base phase (3).
    Phase 4 is treated as Marketed.
    For each jurisdiction + year, only the highest phase is kept.

    Returns:
        {"US": {2015: "Phase 2", 2018: "Phase 3", ...},
         "EP": {2016: "Phase 1", ...}}
    """
    result: Dict[str, Dict[int, float]] = {"US": {}, "EP": {}}

    col_map = {c.strip().lower(): c for c in df.columns}

    mol_col   = next((col_map[k] for k in col_map if k in ("drug", "molecule", "drug name")), None)
    phase_col = next((col_map[k] for k in col_map if k == "phase"), None)
    year_col  = next((col_map[k] for k in col_map if k in ("start year", "startyear", "year")), None)

    # Support both "Primary Country" and "Primary Country 1"
    country_col = next(
        (col_map[k] for k in col_map if k in ("primary country", "primary country 1", "country")),
        None,
    )

    if not all([mol_col, phase_col, year_col, country_col]):
        print(f"[PHASE YEAR] Missing required columns for year-based lookup.")
        print(f"[PHASE YEAR] Need: Drug, Phase, Start Year, Primary Country")
        print(f"[PHASE YEAR] Found: {list(df.columns)}")
        return {"US": {}, "EP": {}}

    print(f"[PHASE YEAR] Using columns → Drug='{mol_col}' | Phase='{phase_col}' | Year='{year_col}' | Country='{country_col}'")

    drug_norm = _normalize(drug_name)

    _US_COUNTRIES = {
        "usa", "us", "united states", "united states of america",
    }
    _EP_COUNTRIES = {
        "eu", "europe", "european union",
        # Common EU member states that may appear
        "germany", "france", "italy", "spain", "netherlands", "belgium",
        "austria", "sweden", "denmark", "finland", "ireland", "portugal",
        "greece", "poland", "czech republic", "hungary", "romania",
        "bulgaria", "croatia", "slovakia", "slovenia", "estonia",
        "latvia", "lithuania", "luxembourg", "malta", "cyprus",
        "uk", "united kingdom", "great britain", "switzerland", "norway",
    }

    matched = 0
    for _, row in df.iterrows():
        raw_mol = str(row.get(mol_col) or "").strip()
        if _normalize(raw_mol) != drug_norm and raw_mol.strip().lower() != drug_name.strip().lower():
            continue

        matched += 1
        phase_str = str(row.get(phase_col) or "").strip()
        year_raw  = row.get(year_col)
        country_raw = str(row.get(country_col) or "").strip()

        phase_num = _parse_phase_number(phase_str)
        if phase_num is None:
            print(f"[PHASE YEAR]   Skipping unparseable phase: '{phase_str}'")
            continue

        # Parse year
        try:
            year = int(float(year_raw))
        except (ValueError, TypeError):
            print(f"[PHASE YEAR]   Skipping unparseable year: '{year_raw}'")
            continue

        # Parse countries (comma-separated)
        countries = [c.strip().lower() for c in re.split(r"[,;/]", country_raw) if c.strip()]

        jurisdictions_hit = set()
        for c in countries:
            if c in _US_COUNTRIES:
                jurisdictions_hit.add("US")
            if c in _EP_COUNTRIES:
                jurisdictions_hit.add("EP")

        for jur in jurisdictions_hit:
            existing = result[jur].get(year, -1.0)
            if phase_num > existing:
                result[jur][year] = phase_num
                print(
                    f"[PHASE YEAR]   {drug_name} | {jur} | {year} → "
                    f"Phase {phase_str} (rank {phase_num})"
                    + (f" [upgraded from {existing}]" if existing >= 0 else "")
                )

    if matched == 0:
        print(f"[PHASE YEAR] NO MATCH for '{drug_name}' in fallback Excel")
        all_names = df[mol_col].dropna().str.strip().unique().tolist()
        print(f"[PHASE YEAR] Available drugs: {all_names}")

    # Convert numeric ranks to labels
    label_result: Dict[str, Dict[int, str]] = {"US": {}, "EP": {}}
    for jur in ("US", "EP"):
        for yr, num in sorted(result[jur].items()):
            label_result[jur][yr] = _phase_number_to_label(num)
        if label_result[jur]:
            print(f"[PHASE YEAR] {drug_name} | {jur} year→phase map: {label_result[jur]}")

    return label_result


def _lookup_phase_for_year(
    phase_year_map: Dict[str, Dict[int, str]],
    jurisdiction: str,
    filing_year: int,
) -> Optional[str]:
    """
    Looks up the phase for a given jurisdiction and filing year.

    Logic: find the highest phase from all entries where Start Year <= filing_year.
    This represents the most advanced phase the drug had reached by that year.
    """
    jur_map = phase_year_map.get(jurisdiction, {})
    if not jur_map:
        return None

    # Collect all phases for years up to and including the filing year
    applicable = {yr: phase for yr, phase in jur_map.items() if yr <= filing_year}

    if not applicable:
        return None

    # Return the highest phase among all applicable years
    best_phase = None
    best_rank  = -1.0
    for yr, phase in applicable.items():
        rank = _parse_phase_number(phase)
        if rank is not None and rank > best_rank:
            best_rank  = rank
            best_phase = phase

    return best_phase


def assign_patent_phases(patents: List[Dict], timeline: Dict) -> List[Dict]:
    """
    Assigns phase_at_filing to each patent based on:
      1. The patent's filing year (from filing_date)
      2. The patent's jurisdiction (US / EP)
      3. The fallback Excel's year-based phase data

    For each patent, looks up the highest phase the drug had reached
    in that jurisdiction by the patent's filing year.

    Falls back to the overall geography_stages (from BQ + fallback merge)
    if year-based lookup is unavailable or returns no match.

    Args:
        patents:  List of patent dicts (from blocking_analyser)
        timeline: Timeline dict (from fetch_clinical_timeline)

    Returns:
        patents list with phase_at_filing set on each entry.
    """
    source           = timeline.get("source", "unavailable")
    geography_stages = timeline.get("geography_stages", {})
    drug_name        = timeline.get("drug_name", "")

    print(f"[PHASE] ── assign_patent_phases called ──")
    print(f"[PHASE]   drug_name        = '{drug_name}'")
    print(f"[PHASE]   source           = '{source}'")
    print(f"[PHASE]   geography_stages = {geography_stages}")

    # Fallback: overall merged stages (non-year-based, from BQ + fallback)
    fallback_stages: Dict[str, Optional[str]] = {
        "US": geography_stages.get("United States"),
        "EP": geography_stages.get("EU"),
    }
    print(f"[PHASE] Fallback stages → US: {fallback_stages['US']} | EP: {fallback_stages['EP']}")

    # ── Build year-based phase lookup from fallback Excel ─────────────────
    phase_year_map: Dict[str, Dict[int, str]] = {"US": {}, "EP": {}}
    fallback_df = load_fallback_phase_excel()
    if fallback_df is not None:
        phase_year_map = _build_phase_year_lookup(drug_name, fallback_df)
    else:
        print(f"[PHASE] Fallback Excel not available — using overall phase only")

    has_year_data = any(phase_year_map[j] for j in ("US", "EP"))
    if has_year_data:
        print(f"[PHASE] Year-based phase data available — will match by filing year")
    else:
        print(f"[PHASE] No year-based data — falling back to overall phase for all patents")

    # ── Assign to each patent ──
    any_assigned = False
    for patent in patents:
        if patent.get("tag") == "SKIPPED":
            patent["phase_at_filing"] = None
            continue

        jurisdiction = (patent.get("jurisdiction") or "").upper()
        filing_date  = patent.get("filing_date")

        # Try to extract filing year
        filing_year = None
        if filing_date:
            try:
                filing_year = int(str(filing_date)[:4])
            except (ValueError, TypeError):
                pass

        stage = None

        # Year-based lookup (preferred)
        if filing_year and has_year_data:
            if jurisdiction in ("US", "EP"):
                stage = _lookup_phase_for_year(phase_year_map, jurisdiction, filing_year)
            else:
                # Unknown jurisdiction — try both, take highest
                us_phase = _lookup_phase_for_year(phase_year_map, "US", filing_year)
                ep_phase = _lookup_phase_for_year(phase_year_map, "EP", filing_year)
                stage = _highest_phase(us_phase, ep_phase)

            if stage:
                print(
                    f"[PHASE] {patent.get('patent_number')} | {jurisdiction} | "
                    f"Filed {filing_year} → {stage} (year-based)"
                )

        # Fallback to overall phase if year-based returned nothing
        if not stage:
            if jurisdiction == "US":
                stage = fallback_stages["US"]
            elif jurisdiction == "EP":
                stage = fallback_stages["EP"]
            else:
                stage = _highest_phase(fallback_stages["US"], fallback_stages["EP"])

            if stage:
                reason = "no filing year" if not filing_year else "no year-based data for this year"
                print(
                    f"[PHASE] {patent.get('patent_number')} | {jurisdiction} | "
                    f"→ {stage} (overall fallback — {reason})"
                )

        patent["phase_at_filing"] = stage
        if stage:
            any_assigned = True

        if not stage:
            print(f"[PHASE] {patent.get('patent_number')} | {jurisdiction} → None")

    if not any_assigned:
        print(f"[PHASE] No phase data available — phase_at_filing = None for all patents")

    return patents
