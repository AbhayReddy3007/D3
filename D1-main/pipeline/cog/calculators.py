"""
calculators.py
──────────────
All derived metric calculations applied to patent lists:

  1. Estimated approval year      — current year + phase offset (Phase 2 → +5, Phase 3 → +3)
  2. Exclusivity year             — approval year + jurisdiction offset (US +5, EP +10)
  3. Controlling patent expiry    — effective filing year + 20 years
  4. Years to entry               — max(expiry, exclusivity) - current year
  5. Pediatric exclusivity        — +0.5 years to US exclusivity if flag set
  6. Score                        — 1–5 based on avg years to entry across US + EP

PTE (Patent Term Extension) is used in effective_filing_year calculations:
  effective_filing_year = filing_year + (pte_months / 12)
"""

from datetime import datetime
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────

_TIMELINE_STAGES = [
    "Preclinical", "Phase 1", "Phase 2", "Phase 3", "Pre-registration", "Marketed"
]
_STAGE_RANK: Dict[str, int] = {s: i for i, s in enumerate(_TIMELINE_STAGES)}


# ─────────────────────────────────────────────
# PTE-adjusted effective filing year
# ─────────────────────────────────────────────

def effective_filing_year(patent: Dict) -> Optional[float]:
    """
    Returns the PTE-adjusted effective filing year for a patent.
      effective_filing_year = filing_year + (pte_months / 12)

    If no PTE, returns plain filing_year.
    If filing_date is missing, returns None.
    """
    filing_date = patent.get("filing_date")
    if not filing_date:
        return None
    try:
        filing_year = int(str(filing_date)[:4])
    except (ValueError, TypeError):
        return None

    pte_months = patent.get("pte")
    if pte_months:
        try:
            filing_year += float(pte_months) / 12
        except (ValueError, TypeError):
            pass

    return filing_year


# ─────────────────────────────────────────────
# 1. Estimated approval year
# ─────────────────────────────────────────────

def assign_estimated_approval_year(patents: List[Dict]) -> List[Dict]:
    """
    Sets estimated_approval_year on the single latest BLOCKING patent per jurisdiction.

    Formula:
      Phase 3 → current year + 3
      Phase 2 → current year + 5

    "Latest" = highest PTE-adjusted effective filing year.
    All other patents get None.
    """
    print("[APPROVAL YEAR] Calculating estimated approval year...")

    _PHASE_OFFSET = {"Phase 2": 5, "Phase 3": 3}
    current_year  = datetime.now().year

    for p in patents:
        p["estimated_approval_year"] = None

    # Dynamically find all jurisdictions in the data
    all_jurisdictions = sorted(set(
        (p.get("jurisdiction") or "").upper() for p in patents
        if p.get("jurisdiction")
    ))

    for jurisdiction in all_jurisdictions:
        candidates = [
            p for p in patents
            if p.get("tag") == "BLOCKING"
            and p.get("filing_date")
            and (p.get("jurisdiction") or "").upper() == jurisdiction
            and effective_filing_year(p) is not None
        ]

        if not candidates:
            print(f"[APPROVAL YEAR] No BLOCKING {jurisdiction} patents with filing dates.")
            continue

        latest = max(candidates, key=effective_filing_year)
        phase  = latest.get("phase_at_filing")
        offset = _PHASE_OFFSET.get(phase)
        eff_yr = effective_filing_year(latest)

        if offset is not None:
            latest["estimated_approval_year"] = current_year + offset
            pte_note = f" (PTE-adjusted: {eff_yr:.2f})" if latest.get("pte") else ""
            print(
                f"[APPROVAL YEAR] {latest.get('patent_number')} | {jurisdiction} | "
                f"Filed: {latest['filing_date']}{pte_note} | "
                f"Phase: {phase} | {current_year} + {offset} → {latest['estimated_approval_year']}"
            )
        else:
            print(
                f"[APPROVAL YEAR] {latest.get('patent_number')} | {jurisdiction} | "
                f"Phase: {phase} — no offset defined (only Phase 2/3 supported)"
            )

    return patents


# ─────────────────────────────────────────────
# 2. Exclusivity year
# ─────────────────────────────────────────────

def assign_exclusivity_year(patents: List[Dict]) -> List[Dict]:
    """
    Sets exclusivity_year on the single latest BLOCKING patent per jurisdiction.

    Base year priority:
      1. Real approval date (approval_date_us / approval_date_eu)
      2. Estimated approval year

    Offset: US → +5, EP → +10
    """
    print("[EXCLUSIVITY] Calculating exclusivity year...")

    _JURISDICTION_OFFSET = {"US": 5, "EP": 10}
    _DEFAULT_OFFSET = 8  # default for other jurisdictions

    for p in patents:
        p["exclusivity_year"] = None

    all_jurisdictions = sorted(set(
        (p.get("jurisdiction") or "").upper() for p in patents
        if p.get("jurisdiction")
    ))

    for jurisdiction in all_jurisdictions:
        offset = _JURISDICTION_OFFSET.get(jurisdiction, _DEFAULT_OFFSET)

        candidates = [
            p for p in patents
            if p.get("tag") == "BLOCKING"
            and p.get("filing_date")
            and (p.get("jurisdiction") or "").upper() == jurisdiction
            and effective_filing_year(p) is not None
        ]

        if not candidates:
            print(f"[EXCLUSIVITY] No BLOCKING {jurisdiction} patents with filing dates.")
            continue

        latest = max(candidates, key=effective_filing_year)

        real_date = (
            latest.get("approval_date_us") if jurisdiction == "US"
            else latest.get("approval_date_eu") if jurisdiction in ("EP", "EU")
            else latest.get(f"approval_date_{jurisdiction.lower()}")
        )

        base_year = None

        if real_date and str(real_date).lower() not in ("none", "null", "n/a", ""):
            try:
                base_year = int(datetime.strptime(str(real_date), "%d-%b-%Y").year)
                print(
                    f"[EXCLUSIVITY] {latest.get('patent_number')} | {jurisdiction} | "
                    f"Real approval date: {real_date} → year {base_year}"
                )
            except ValueError:
                try:
                    base_year = int(str(real_date)[:4])
                except (ValueError, TypeError):
                    pass

        if base_year is None:
            base_year = latest.get("estimated_approval_year")
            if base_year:
                print(
                    f"[EXCLUSIVITY] {latest.get('patent_number')} | {jurisdiction} | "
                    f"Using estimated approval year: {base_year}"
                )

        if base_year is not None:
            latest["exclusivity_year"] = int(base_year) + offset
            print(
                f"[EXCLUSIVITY] {latest.get('patent_number')} | {jurisdiction} | "
                f"{base_year} + {offset} → {latest['exclusivity_year']}"
            )
        else:
            print(
                f"[EXCLUSIVITY] {latest.get('patent_number')} | {jurisdiction} | "
                f"No base year available — skipping"
            )

    return patents


# ─────────────────────────────────────────────
# 3. Controlling patent expiry year
# ─────────────────────────────────────────────

def assign_controlling_patent_expiry_year(patents: List[Dict]) -> List[Dict]:
    """
    Sets controlling_patent_expiry_year on the single latest BLOCKING patent per jurisdiction.

    Formula:
      expiry = effective_filing_year + 20

    effective_filing_year includes PTE adjustment if present.
    """
    print("[CONTROLLING EXPIRY] Calculating controlling patent expiry year...")

    for p in patents:
        p["controlling_patent_expiry_year"] = None

    all_jurisdictions = sorted(set(
        (p.get("jurisdiction") or "").upper() for p in patents
        if p.get("jurisdiction")
    ))

    for jurisdiction in all_jurisdictions:
        candidates = [
            p for p in patents
            if p.get("tag") == "BLOCKING"
            and p.get("filing_date")
            and (p.get("jurisdiction") or "").upper() == jurisdiction
            and effective_filing_year(p) is not None
        ]

        if not candidates:
            print(f"[CONTROLLING EXPIRY] No BLOCKING {jurisdiction} patents with filing dates.")
            continue

        latest   = max(candidates, key=effective_filing_year)
        eff_yr   = effective_filing_year(latest)
        pte_note = ""

        if latest.get("pte"):
            pte_months  = float(latest["pte"])
            base_year   = int(str(latest["filing_date"])[:4])
            expiry_year = int(base_year + pte_months / 12) + 20
            pte_note    = f" + PTE {pte_months:.0f}mo ({pte_months/12:.2f}yr)"
        else:
            expiry_year = int(eff_yr) + 20

        latest["controlling_patent_expiry_year"] = expiry_year
        print(
            f"[CONTROLLING EXPIRY] {latest.get('patent_number')} | {jurisdiction} | "
            f"Filed: {latest['filing_date']}{pte_note} | "
            f"Effective year: {eff_yr:.2f} + 20 → {expiry_year}"
        )

    return patents


# ─────────────────────────────────────────────
# 4. Years to entry
# ─────────────────────────────────────────────

def assign_years_to_entry(patents: List[Dict]) -> List[Dict]:
    """
    Sets years_to_entry on each patent.

    Formula:
      years_to_entry = max(controlling_patent_expiry_year, exclusivity_year) - current_year

    Patents with neither value get None.
    """
    print("[YEARS TO ENTRY] Calculating years to entry...")

    current_year = datetime.now().year

    for p in patents:
        controlling = p.get("controlling_patent_expiry_year")
        exclusivity = p.get("exclusivity_year")
        candidates  = [v for v in [controlling, exclusivity] if v is not None]

        if candidates:
            p["years_to_entry"] = max(candidates) - current_year
            print(
                f"[YEARS TO ENTRY] {p.get('patent_number')} | "
                f"max({controlling}, {exclusivity}) - {current_year} = {p['years_to_entry']}"
            )
        else:
            p["years_to_entry"] = None

    return patents


# ─────────────────────────────────────────────
# 5. Pediatric exclusivity adjustment
# ─────────────────────────────────────────────

def apply_pediatric_exclusivity(patents: List[Dict]) -> List[Dict]:
    """
    US only: adds 0.5 years (6 months) to exclusivity_year if pediatric_exclusivity is True.
    Recalculates years_to_entry for affected patents.
    """
    print("[PEDIATRIC] Applying pediatric exclusivity adjustments...")

    current_year = datetime.now().year

    for p in patents:
        if (p.get("jurisdiction") or "").upper() != "US":
            continue
        if not p.get("pediatric_exclusivity"):
            continue
        if p.get("exclusivity_year") is None:
            continue

        original            = p["exclusivity_year"]
        p["exclusivity_year"] = original + 0.5
        print(
            f"[PEDIATRIC] {p.get('patent_number')} | US | "
            f"Exclusivity: {original} + 0.5 → {p['exclusivity_year']}"
        )

        controlling = p.get("controlling_patent_expiry_year")
        exclusivity = p["exclusivity_year"]
        candidates  = [v for v in [controlling, exclusivity] if v is not None]
        if candidates:
            p["years_to_entry"] = max(candidates) - current_year
            print(
                f"[PEDIATRIC] {p.get('patent_number')} | "
                f"Recalculated years_to_entry → {p['years_to_entry']}"
            )

    return patents


# ─────────────────────────────────────────────
# Scoring helper
# ─────────────────────────────────────────────

def _avg_to_score(avg: float) -> int:
    """
    Converts average years to entry into a 1-5 score.

    Score | Avg years to entry
      5   | <= 6 years
      4   | 7–8 years
      3   | 9–11 years
      2   | 12–13 years
      1   | > 13 years
    """
    if avg <= 6:
        return 5
    elif avg <= 8:
        return 4
    elif avg <= 11:
        return 3
    elif avg <= 13:
        return 2
    else:
        return 1


# ─────────────────────────────────────────────
# 6. Score (all jurisdictions)
# ─────────────────────────────────────────────

def assign_score(patents: List[Dict]) -> List[Dict]:
    """
    Calculates avg_years_to_entry and score across ALL jurisdictions.

    Steps:
      1. Collect years_to_entry from each jurisdiction's blocking patent.
      2. avg_years_to_entry = mean of available values.
      3. Score (1-5) based on avg_years_to_entry.
      4. Both values are assigned to ALL patents (drug-level metric).
    """
    print("[SCORE] Calculating avg years to entry and score...")

    for p in patents:
        p["avg_years_to_entry"] = None
        p["score"]              = None

    yte_values = []
    all_jurisdictions = sorted(set(
        (p.get("jurisdiction") or "").upper() for p in patents
        if p.get("jurisdiction")
    ))

    for jurisdiction in all_jurisdictions:
        match = next(
            (
                p for p in patents
                if (p.get("jurisdiction") or "").upper() == jurisdiction
                and p.get("years_to_entry") is not None
            ),
            None,
        )
        if match:
            yte_values.append(match["years_to_entry"])
            print(
                f"[SCORE] {jurisdiction} years_to_entry: "
                f"{match['years_to_entry']} (from {match.get('patent_number')})"
            )
        else:
            print(f"[SCORE] {jurisdiction} years_to_entry: not available")

    if not yte_values:
        print("[SCORE] No years_to_entry values available — score = N/A")
        return patents

    avg   = round(sum(yte_values) / len(yte_values), 2)
    score = _avg_to_score(avg)

    print(f"[SCORE] avg_years_to_entry: {avg} → Score: {score}")

    for p in patents:
        p["avg_years_to_entry"] = avg
        p["score"]              = score

    return patents


# ─────────────────────────────────────────────
# 7. US + EP specific score (IP Dimension 1)
# ─────────────────────────────────────────────

def assign_us_ep_score(patents: List[Dict]) -> List[Dict]:
    """
    Calculates avg_years_to_entry_us_ep and ip_dimension_1_score
    using ONLY US and EP jurisdictions.

    Steps:
      1. Collect years_to_entry from the US and EP blocking patents only.
      2. avg_years_to_entry_us_ep = mean of available US/EP values.
      3. ip_dimension_1_score (1-5) based on avg_years_to_entry_us_ep.
      4. Both values are assigned to ALL patents (drug-level metric).
    """
    print("[IP DIM 1] Calculating US+EP avg years to entry and IP Dimension 1 Score...")

    for p in patents:
        p["avg_years_to_entry_us_ep"] = None
        p["ip_dimension_1_score"]     = None

    yte_us_ep = []

    for jurisdiction in ("US", "EP"):
        match = next(
            (
                p for p in patents
                if (p.get("jurisdiction") or "").upper() == jurisdiction
                and p.get("years_to_entry") is not None
            ),
            None,
        )
        if match:
            yte_us_ep.append(match["years_to_entry"])
            print(
                f"[IP DIM 1] {jurisdiction} years_to_entry: "
                f"{match['years_to_entry']} (from {match.get('patent_number')})"
            )
        else:
            print(f"[IP DIM 1] {jurisdiction} years_to_entry: not available")

    if not yte_us_ep:
        print("[IP DIM 1] No US/EP years_to_entry available — IP Dimension 1 Score = N/A")
        return patents

    avg   = round(sum(yte_us_ep) / len(yte_us_ep), 2)
    score = _avg_to_score(avg)

    print(f"[IP DIM 1] avg_years_to_entry_us_ep: {avg} → IP Dimension 1 Score: {score}")

    for p in patents:
        p["avg_years_to_entry_us_ep"] = avg
        p["ip_dimension_1_score"]     = score

    return patents


# ─────────────────────────────────────────────
# Main public function — run all calculators
# ─────────────────────────────────────────────

def run_calculations(patents: List[Dict]) -> List[Dict]:
    """
    Runs all derived metric calculators in the correct order.

    Order matters:
      1. estimated_approval_year  (needs phase_at_filing)
      2. exclusivity_year         (needs estimated_approval_year or real approval date)
      3. controlling_expiry       (needs filing_date + pte)
      4. years_to_entry           (needs exclusivity_year + controlling_expiry)
      5. pediatric adjustment     (adjusts exclusivity_year + recalculates years_to_entry)
      6. score                    (needs years_to_entry — all jurisdictions)
      7. us_ep_score              (needs years_to_entry — US + EP only)

    Call this BEFORE fetching real approval dates, then call again after
    approval dates are attached (as done in the main pipeline).

    Args:
        patents: List of patent dicts (must already have phase_at_filing set)

    Returns:
        Updated patents list with all derived fields populated.
    """
    patents = assign_estimated_approval_year(patents)
    patents = assign_exclusivity_year(patents)
    patents = assign_controlling_patent_expiry_year(patents)
    patents = assign_years_to_entry(patents)
    patents = apply_pediatric_exclusivity(patents)
    patents = assign_score(patents)
    patents = assign_us_ep_score(patents)
    return patents
