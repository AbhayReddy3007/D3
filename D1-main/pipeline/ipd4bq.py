import pandas as pd
import google.generativeai as genai
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda **kw: None  # Not needed on Cloud Run
import os
import time
import json
import re
from datetime import date
from dateutil.relativedelta import relativedelta
from google.cloud import bigquery
from google.oauth2 import service_account

# ── BIGQUERY CONFIG ──────────────────────────────────────────────────────────
BQ_PROJECT_ID  = "cognito-prod-394707"
BQ_DATASET_ID  = "cognito_prod_datamart"
BQ_TABLE_ID    = "Master_LOE"
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
BQ_LOCATION    = "asia-south1"

# ── CONFIG ──────────────────────────────────────────────────────────────────
TARGET_JURISDICTIONS  = {"CN", "IN", "BR", "AU", "RU", "US", "CA", "JP", "MX", "TW", "KR", "TW"}
KEEP_CLAIM_CATEGORIES = {"Composition Of Matter", "Device", "Formulation"}
ALL_CRITICAL_CATEGORIES = sorted(KEEP_CLAIM_CATEGORIES)

KEEP_COLUMNS = [
    "Drug Name", "Patent Number", "Jurisdiction", "Tag",
    "Grant Date", "Filing Date", "Approval Date",
    "PTE (months)", "Step 1 Claim Category", "Years to Entry", "Type",
]
# ────────────────────────────────────────────────────────────────────────────

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def _get_credentials():
    """Get credentials: use service account file if available, else default (Cloud Run)."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None  # Use ADC (Application Default Credentials)


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED DATE NORMALIZER
#  Converts bare years like "2025" or "2025.0" → "2025-01-01"
#  Used for both Grant Date and Filing Date columns.
# ═══════════════════════════════════════════════════════════════════════════
def _normalize_year_only(val) -> str:
    """Return a parseable date string; bare years become YYYY-01-01."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    try:
        n = float(s)
        if n == int(n):          # bare year like 2025 or 2025.0
            return f"{int(n)}-01-01"
    except (ValueError, TypeError):
        pass
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER — safe Gemini call with retry
# ═══════════════════════════════════════════════════════════════════════════
def _call_gemini(prompt: str, retries: int = 3, backoff: float = 2.0) -> str | None:
    for attempt in range(retries):
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"    ⚠ Gemini error ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                print(f"    ✗ Gemini failed after {retries} attempts: {e}")
                return None


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  1. APPROVAL DATE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════
JURISDICTION_FULL = {
    "CN": "China (NMPA / formerly CFDA)",
    "IN": "India (CDSCO / DCGI)",
    "BR": "Brazil (ANVISA)",
    "AU": "Australia (TGA)",
    "RU": "Russia (Ministry of Health / Roszdravnadzor)",
    "US": "United States (FDA)",
    "JP": "Japan (PMDA / MHLW)",
    "KR": "South Korea (MFDS)",
    "TW": "Taiwan (TFDA)",
    "CA": "Canada (Health Canada)",
    "MX": "Mexico (COFEPRIS)",
}


def lookup_earliest_approval_date(drug_name: str, jurisdiction: str) -> str:
    jur_full = JURISDICTION_FULL.get(jurisdiction, jurisdiction)

    prompt_1 = (
        f"You are a pharmaceutical regulatory expert.\n"
        f"Drug: {drug_name}\n"
        f"Jurisdiction: {jur_full}\n\n"
        f"Task: List EVERY regulatory marketing-authorisation approval date "
        f"for this drug in this jurisdiction (all indications, formulations, "
        f"brand names). Use only factual, publicly verifiable data.\n\n"
        f"Rules:\n"
        f"- Respond ONLY with a JSON object, NO markdown, NO explanation.\n"
        f"- Format: {{\"dates\": [\"YYYY-MM-DD\", ...], \"source\": \"<brief source>\"}}\n"
        f"- If you are NOT confident or the drug is not approved there, "
        f"return: {{\"dates\": [], \"source\": \"unknown\"}}\n"
    )

    raw_1 = _call_gemini(prompt_1)
    if not raw_1:
        return "Not found"

    try:
        data_1 = json.loads(_clean_json(raw_1))
        dates_1 = data_1.get("dates", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"    ⚠ Pass-1 JSON parse failed for {drug_name}/{jurisdiction}")
        return "Not found"

    valid_dates = [pd.to_datetime(d, errors="coerce") for d in dates_1]
    valid_dates = [d for d in valid_dates if pd.notna(d)]
    if not valid_dates:
        return "Not found"

    earliest = min(valid_dates).strftime("%Y-%m-%d")

    prompt_2 = (
        f"You are a pharmaceutical regulatory expert.\n"
        f"Drug: {drug_name}\n"
        f"Jurisdiction: {jur_full}\n\n"
        f"A previous search returned the earliest approval date as {earliest}.\n"
        f"Is this date CORRECT for the first marketing-authorisation approval "
        f"of this drug in this jurisdiction?\n\n"
        f"Rules:\n"
        f"- Respond ONLY with a JSON object, NO markdown.\n"
        f"- Format: {{\"confirmed\": true/false, \"corrected_date\": \"YYYY-MM-DD or null\", "
        f"\"reason\": \"brief\"}}\n"
        f"- If you are not sure, set confirmed to false and corrected_date to null.\n"
    )

    raw_2 = _call_gemini(prompt_2)
    if raw_2:
        try:
            data_2 = json.loads(_clean_json(raw_2))
            if data_2.get("confirmed"):
                return earliest
            corrected = data_2.get("corrected_date")
            if corrected:
                cd = pd.to_datetime(corrected, errors="coerce")
                if pd.notna(cd):
                    return cd.strftime("%Y-%m-%d")
        except (json.JSONDecodeError, AttributeError):
            pass

    return earliest


# ═══════════════════════════════════════════════════════════════════════════
#  2. PTE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════
PTE_RULES = {
    "IN": {"allowed": False},
    "BR": {"allowed": False},
    "CN": {"allowed": True, "max_ext_years": 5, "max_effective_from_approval_years": 14},
    "AU": {"allowed": True, "max_ext_years": 5},
    "RU": {"allowed": True, "max_ext_years": 5},
    "US": {"allowed": True, "max_ext_years": 5},
    "JP": {"allowed": True, "max_ext_years": 5},
    "KR": {"allowed": True, "max_ext_years": 5, "max_effective_from_approval_years": 14},
    "TW": {"allowed": True, "max_ext_years": 5},
    "CA": {"allowed": True, "max_ext_years": 2, "post_expiry": True},
    "MX": {"allowed": False, "note": "USMCA mandates PTE but implementing regulations incomplete"},
}


def lookup_pte(drug_name: str, patent_number: str, jurisdiction: str) -> dict:
    rules = PTE_RULES.get(jurisdiction, {})
    if not rules.get("allowed"):
        return {"pte_months": 0, "pte_status": "Not applicable", "pte_source": ""}

    jur_full = JURISDICTION_FULL.get(jurisdiction, jurisdiction)
    registry_hint = {
        "CN": "CNIPA patent register",
        "AU": "AusPat (IP Australia)",
        "RU": "Rospatent / FIPS register",
        "US": "USPTO Orange Book / Patent Center",
        "JP": "JPO (Japan Patent Office) register",
        "KR": "KIPO (Korean Intellectual Property Office) register",
        "TW": "TIPO (Taiwan Intellectual Property Office) register",
        "CA": "Health Canada CSP (Certificate of Supplementary Protection) Registry",
        "MX": "IMPI records / court rulings",
    }.get(jurisdiction, "national patent office")

    if jurisdiction == "CA":
        task_desc = (
            f"Task: Has a Certificate of Supplementary Protection (CSP) been ISSUED "
            f"for this patent in Canada? If yes, how many months of additional "
            f"protection was granted? Note: CSP takes effect after patent expiry "
            f"and is capped at 2 years (24 months).\n\n"
        )
    else:
        task_desc = (
            f"Task: Has a patent term extension (PTE) been GRANTED for this patent "
            f"in this jurisdiction? If yes, how many months was the extension?\n\n"
        )

    prompt = (
        f"You are a patent-term-extension (PTE) expert.\n"
        f"Drug: {drug_name}\n"
        f"Patent number: {patent_number}\n"
        f"Jurisdiction: {jur_full}\n"
        f"Registry to check: {registry_hint}\n\n"
        f"{task_desc}"
        f"Rules:\n"
        f"- Respond ONLY with JSON, NO markdown.\n"
        f"- Format: {{"
        f"\"pte_status\": \"Granted\" | \"Pending\" | \"Not filed\" | \"Not found\","
        f"\"extension_months\": <int or null>,"
        f"\"source\": \"<brief>\""
        f"}}\n"
        f"- If unsure, use \"Not found\" and null.\n"
    )

    raw = _call_gemini(prompt)
    if not raw:
        return {"pte_months": 0, "pte_status": "Not found", "pte_source": ""}

    try:
        data = json.loads(_clean_json(raw))
        status = data.get("pte_status", "Not found")
        months = data.get("extension_months")
        source = data.get("source", "")

        if months is None or status != "Granted":
            return {"pte_months": 0, "pte_status": status, "pte_source": source}

        max_months = rules.get("max_ext_years", 5) * 12
        months = min(int(months), max_months)
        return {"pte_months": months, "pte_status": status, "pte_source": source}

    except (json.JSONDecodeError, AttributeError, ValueError):
        return {"pte_months": 0, "pte_status": "Not found", "pte_source": ""}


# ═══════════════════════════════════════════════════════════════════════════
#  3. EXPIRY CALCULATION
# ═══════════════════════════════════════════════════════════════════════════
def compute_expiry(filing_date, approval_date_str: str, pte_months: int, jurisdiction: str) -> dict:
    result = {"base_expiry": "Not found", "adjusted_expiry": "Not found", "pte_capped": False}

    if pd.isna(filing_date):
        return result

    base = filing_date.date() + relativedelta(years=20)
    result["base_expiry"] = base

    if pte_months == 0:
        result["adjusted_expiry"] = base
        return result

    rules = PTE_RULES.get(jurisdiction, {})
    adjusted = base + relativedelta(months=pte_months)

    max_eff = rules.get("max_effective_from_approval_years")
    if max_eff and approval_date_str and approval_date_str != "Not found":
        approval_dt = pd.to_datetime(approval_date_str, errors="coerce")
        if pd.notna(approval_dt):
            cap = approval_dt.date() + relativedelta(years=max_eff)
            if adjusted > cap:
                adjusted = cap
                result["pte_capped"] = True

    result["adjusted_expiry"] = adjusted
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  4. EXPIRY GAP
# ═══════════════════════════════════════════════════════════════════════════
def compute_expiry_gap(adjusted_expiry) -> int | str:
    """Fallback calculation for non-US rows."""
    exp = pd.to_datetime(adjusted_expiry, errors="coerce")
    if pd.isna(exp):
        return "N/A"
    return exp.year - date.today().year


# ═══════════════════════════════════════════════════════════════════════════
#  5. KEY PROTECTION GAP — data-driven (no Gemini)
# ═══════════════════════════════════════════════════════════════════════════
def determine_protection_gap(categories_present: set[str]) -> str:
    present = categories_present & KEEP_CLAIM_CATEGORIES
    absent  = KEEP_CLAIM_CATEGORIES - present

    if not present:
        return "No blocking patents; all categories absent"
    if not absent:
        return "Full protection; all categories covered"

    present_str = ", ".join(sorted(present))
    absent_str  = ", ".join(sorted(absent))
    return f"{present_str} present; {absent_str} absent"


# ═══════════════════════════════════════════════════════════════════════════
#  6. GEOGRAPHIC ARBITRAGE SCORE (per country, 1–5)
# ═══════════════════════════════════════════════════════════════════════════
def compute_arbitrage_score(loe_year: int, gap_vs_longest: int, current_year: int) -> int:
    if loe_year == current_year:
        return 5
    if gap_vs_longest >= 5:
        return 4
    if gap_vs_longest >= 3:
        return 3
    if gap_vs_longest == 2:
        return 2
    return 1


def get_dimension_iv_rating(avg_score: float) -> str:
    if avg_score <= 1.0:
        return "FAIL — No viable geographic arbitrage"
    elif avg_score <= 2.0:
        return "Limited Geographic Arbitrage Opportunity"
    elif avg_score <= 3.0:
        return "Moderate Geographic Arbitrage Opportunity"
    elif avg_score <= 4.0:
        return "Strong Geographic Arbitrage Opportunity"
    else:
        return "Very Strong Geographic Arbitrage Opportunity"


# ═══════════════════════════════════════════════════════════════════════════
#  7. ARBITRAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def compute_arbitrage(shortlisted: pd.DataFrame, full_filtered: pd.DataFrame) -> pd.DataFrame:
    current_year = date.today().year

    cats_lookup: dict[tuple, set] = {}
    for _, row in full_filtered.iterrows():
        key = (row["Drug Name"], row["Jurisdiction"])
        cat = str(row.get("Step 1 Claim Category", "")).strip().title()
        if cat in KEEP_CLAIM_CATEGORIES:
            cats_lookup.setdefault(key, set()).add(cat)

    drugs = shortlisted["Drug Name"].unique()
    arb_rows = []

    for drug in sorted(drugs):
        drug_df = shortlisted[shortlisted["Drug Name"] == drug].copy()

        drug_df["_adj_dt"] = pd.to_datetime(drug_df["Adjusted Expiry (with PTE)"], errors="coerce")
        loe_by_jur = drug_df.groupby("Jurisdiction")["_adj_dt"].max().dropna()

        if loe_by_jur.empty:
            continue

        loe_years = loe_by_jur.dt.year
        us_loe = loe_years.get("US")
        max_loe = int(loe_years.max())
        max_loe_jur = loe_years.idxmax()
        sorted_jurs = loe_years.sort_values(ascending=True)
        prev_loe = None
        country_scores = {}

        for launch_order, (jur, loe_year) in enumerate(sorted_jurs.items(), start=1):
            if jur == "US":
                gap_vs_us = "N/A"
            elif us_loe is not None:
                gap_vs_us = int(us_loe) - int(loe_year)
            else:
                gap_vs_us = "US data missing"

            if jur == max_loe_jur:
                gap_vs_longest = "N/A (longest)"
            else:
                gap_vs_longest = max_loe - int(loe_year)

            if jur == "US":
                ctc_gap = "N/A"
            elif prev_loe is None:
                ctc_gap = 0
            else:
                ctc_gap = int(loe_year) - int(prev_loe)

            cats_present = cats_lookup.get((drug, jur), set())
            gap_desc = determine_protection_gap(cats_present)

            if jur == "US":
                score = "N/A"
                signal = "Reference market"
            else:
                gap_for_scoring = 0 if gap_vs_longest == "N/A (longest)" else int(gap_vs_longest)
                score = compute_arbitrage_score(
                    loe_year=int(loe_year),
                    gap_vs_longest=gap_for_scoring,
                    current_year=current_year,
                )
                country_scores[jur] = score

                if score == 5:
                    signal = "Immediate opportunity"
                elif score == 4:
                    signal = "Strong arbitrage"
                elif score == 3:
                    signal = "Meaningful arbitrage"
                elif score == 2:
                    signal = "Limited arbitrage"
                else:
                    signal = "No arbitrage"

            arb_rows.append({
                "Drug Name": drug,
                "Launch Order": launch_order,
                "Jurisdiction": jur,
                "Product LOE (Year)": int(loe_year),
                "Gap vs US (Years)": gap_vs_us,
                "Gap vs Longest LOE (Years)": gap_vs_longest,
                "Country-to-Country Arbitrage (Years)": ctc_gap,
                "Key Protection Gap": gap_desc,
                "Arbitrage Score": score,
                "Arbitrage Signal": signal,
            })

            if jur != "US":
                prev_loe = loe_year

        if country_scores:
            dim4 = round(sum(country_scores.values()) / len(country_scores), 2)
            dim4_rating = get_dimension_iv_rating(dim4)
        else:
            dim4 = "N/A"
            dim4_rating = "N/A"

        for row in arb_rows:
            if row["Drug Name"] == drug:
                row["Dimension IV Score"] = dim4
                row["Dimension IV Rating"] = dim4_rating

    return pd.DataFrame(arb_rows)


# ═══════════════════════════════════════════════════════════════════════════
#  BIGQUERY HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def load_from_bigquery() -> pd.DataFrame:
    """Load Master_LOE table from BigQuery and return as DataFrame."""
    print(f"\nConnecting to BigQuery …")
    print(f"  Project : {BQ_PROJECT_ID}")
    print(f"  Dataset : {BQ_DATASET_ID}")
    print(f"  Table   : {BQ_TABLE_ID}")

    credentials = _get_credentials()
    client = bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )

    query = f"SELECT * FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`"
    print(f"  Query   : {query}")

    df = client.query(query).to_dataframe()
    df = df.astype(str).replace("None", pd.NA).replace("<NA>", pd.NA)
    print(f"  ✓ Loaded {len(df):,} rows from BigQuery")
    return df


def _to_snake_case(col: str) -> str:
    """Convert any column name to snake_case."""
    s = col.strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = s.strip("_").lower()
    return s


def write_to_bigquery(df: pd.DataFrame, table_id: str) -> None:
    """Write a DataFrame to a BigQuery table (WRITE_TRUNCATE)."""
    df = df.copy()
    df.columns = [_to_snake_case(c) for c in df.columns]
    df = df.astype(str).replace("nan", pd.NA).replace("<NA>", pd.NA)

    credentials = _get_credentials()
    client = bigquery.Client(
        project=BQ_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )

    full_table_id = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()
    print(f"  ✓ Written {len(df):,} rows → `{full_table_id}`")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  IPD PATENT ANALYSIS TOOL")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  Starting analysis…")
    print("=" * 60)

    # ── Load from BigQuery ────────────────────────────────────────────────
    df = load_from_bigquery()
    # Rename BQ underscore columns → space-separated names the rest of the code expects
    df.columns = df.columns.str.strip().str.replace("_", " ")
    # BQ has "PTE months"; code expects "PTE (months)"
    df.rename(columns={"PTE months": "PTE (months)"}, inplace=True)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    total = len(df)
    print(f"\nTotal rows loaded           : {total}")

    # Filter 1: Jurisdiction
    f1 = df["Jurisdiction"].isin(TARGET_JURISDICTIONS)
    print(f"After Jurisdiction filter   : {f1.sum()} (removed {total - f1.sum()})")

    # ── FIX: Normalize Grant Date before parsing ──────────────────────────
    grant_date_normalized = df["Grant Date"].apply(_normalize_year_only)
    df["Grant Date"] = grant_date_normalized

    # Filter 2: Grant Date is a valid date
    grant_date_parsed = pd.to_datetime(grant_date_normalized, errors="coerce")
    f2 = f1 & grant_date_parsed.notna()
    print(f"After Grant Date filter     : {f2.sum()} (removed {f1.sum() - f2.sum()})")

    # Filter 3: Tag = BLOCKING
    f3 = f2 & (df["Tag"].str.upper() == "BLOCKING")
    print(f"After Tag=BLOCKING filter   : {f3.sum()} (removed {f2.sum() - f3.sum()})")

    # Filter 4: Step 1 Claim Category
    f4 = f3 & (df["Step 1 Claim Category"].str.strip().str.title().isin(KEEP_CLAIM_CATEGORIES))
    print(f"After Claim Category filter : {f4.sum()} (removed {f3.sum() - f4.sum()})")
    print(f"{'='*60}")

    shortlisted = df[f4].reset_index(drop=True)
    cols_present = [c for c in KEEP_COLUMNS if c in shortlisted.columns]
    shortlisted = shortlisted[cols_present].copy()

    # Keep a copy of full filtered data (before dedup) for category lookup
    full_filtered = shortlisted.copy()

    # ── Build "Years to Entry" lookup from US rows in original df ─────────
    years_to_entry_map: dict[tuple, float] = {}
    if "Years to Entry" in df.columns:
        us_rows = df[df["Jurisdiction"] == "US"].dropna(subset=["Years to Entry"])
        for _, row in us_rows.iterrows():
            key = (row["Drug Name"], row["Patent Number"])
            try:
                years_to_entry_map[key] = float(row["Years to Entry"])
            except (ValueError, TypeError):
                pass
        print(f"\n'Years to Entry' entries loaded for US: {len(years_to_entry_map)}")
    else:
        print("\n  ⚠ Column 'Years to Entry' not found in input — "
              "falling back to calculated gap for all rows.")

    # ── 1. Fill missing Approval Dates ───────────────────────────────────
    approval_cache: dict[tuple, str] = {}
    groups_to_query: set[tuple] = set()

    for idx, row in shortlisted.iterrows():
        key = (row["Drug Name"], row["Jurisdiction"])
        val = row.get("Approval Date", "")
        is_missing = pd.isna(val) or val == "" or pd.isna(pd.to_datetime(val, errors="coerce"))
        if is_missing:
            groups_to_query.add(key)

    print(f"\nApproval-date lookups needed: {len(groups_to_query)}")

    for drug, jur in sorted(groups_to_query):
        result = lookup_earliest_approval_date(drug, jur)
        approval_cache[(drug, jur)] = result
        print(f"  ✓ {drug} / {jur} → {result}")
        time.sleep(1)

    for idx, row in shortlisted.iterrows():
        key = (row["Drug Name"], row["Jurisdiction"])
        if key in approval_cache:
            shortlisted.at[idx, "Approval Date"] = approval_cache[key]

    # ── 2. PTE lookup ────────────────────────────────────────────────────
    pte_cache: dict[tuple, dict] = {}
    pte_groups: set[tuple] = set()

    for idx, row in shortlisted.iterrows():
        jur = row["Jurisdiction"]
        pat_type = str(row.get("Type", "")).strip().lower()
        if pat_type == "forecasted":
            continue
        if PTE_RULES.get(jur, {}).get("allowed"):
            key = (row["Drug Name"], row["Patent Number"], jur)
            pte_groups.add(key)

    print(f"\nPTE lookups needed: {len(pte_groups)}")

    for drug, pat, jur in sorted(pte_groups):
        result = lookup_pte(drug, pat, jur)
        pte_cache[(drug, pat, jur)] = result
        print(f"  ✓ {pat} ({jur}) → {result['pte_status']}, {result['pte_months']} months")
        time.sleep(1)

    shortlisted["PTE Status"]           = ""
    shortlisted["PTE Months (Granted)"] = 0
    shortlisted["PTE Source"]           = ""

    for idx, row in shortlisted.iterrows():
        key = (row["Drug Name"], row["Patent Number"], row["Jurisdiction"])
        if key in pte_cache:
            info = pte_cache[key]
            shortlisted.at[idx, "PTE Status"]           = info["pte_status"]
            shortlisted.at[idx, "PTE Months (Granted)"] = info["pte_months"]
            shortlisted.at[idx, "PTE Source"]           = info["pte_source"]
        else:
            shortlisted.at[idx, "PTE Status"] = "Not applicable"

    # ── 3. Compute expiry dates ──────────────────────────────────────────
    shortlisted["Filing Date"] = shortlisted["Filing Date"].apply(_normalize_year_only)
    filing_dates      = pd.to_datetime(shortlisted["Filing Date"], errors="coerce")
    base_expiries     = []
    adjusted_expiries = []
    pte_capped_flags  = []

    for idx, row in shortlisted.iterrows():
        filing_dt = filing_dates.iloc[idx]
        approval  = row.get("Approval Date", "Not found")
        pte_m     = int(row.get("PTE Months (Granted)", 0) or 0)
        jur       = row["Jurisdiction"]

        exp = compute_expiry(filing_dt, approval, pte_m, jur)
        base_expiries.append(exp["base_expiry"])
        adjusted_expiries.append(exp["adjusted_expiry"])
        pte_capped_flags.append(exp["pte_capped"])

    shortlisted["Base Patent Term (Filing+20y)"] = base_expiries
    shortlisted["Adjusted Expiry (with PTE)"]    = adjusted_expiries
    shortlisted["PTE Capped (Post-Approval Cap)"]      = pte_capped_flags

    # ── 3b. Type-based expiry override ───────────────────────────────────
    if "Type" in df.columns:
        filtered_df = df[f4].copy()
        has_non_forecasted: dict[tuple, bool] = {}
        for _, row in filtered_df.iterrows():
            key = (row["Drug Name"], row["Jurisdiction"])
            pat_type = str(row.get("Type", "")).strip()
            if pat_type.lower() != "forecasted" and pat_type != "":
                has_non_forecasted[key] = True

        current_year_date = date(date.today().year, 1, 1)
        overrides_base = 0
        overrides_current_year = 0

        for idx, row in shortlisted.iterrows():
            key = (row["Drug Name"], row["Jurisdiction"])
            if has_non_forecasted.get(key, False):
                shortlisted.at[idx, "Adjusted Expiry (with PTE)"] = row["Base Patent Term (Filing+20y)"]
                shortlisted.at[idx, "PTE Months (Granted)"] = 0
                shortlisted.at[idx, "PTE Status"] = "Overridden (non-forecasted patent exists)"
                shortlisted.at[idx, "PTE Capped (Post-Approval Cap)"] = False
                overrides_base += 1
            else:
                shortlisted.at[idx, "Adjusted Expiry (with PTE)"] = current_year_date
                shortlisted.at[idx, "PTE Months (Granted)"] = 0
                shortlisted.at[idx, "PTE Status"] = "Overridden (all forecasted)"
                shortlisted.at[idx, "PTE Capped (Post-Approval Cap)"] = False
                overrides_current_year += 1

        print(f"\nType-based expiry overrides:")
        print(f"  → Set to Base Term (non-forecasted exists): {overrides_base}")
        print(f"  → Set to Current Year (all forecasted):     {overrides_current_year}")
    else:
        print("\n  ⚠ Column 'Type' not found in input — skipping Type-based expiry override.")

    # ── 4. Deduplicate ───────────────────────────────────────────────────
    before_dedup = len(shortlisted)
    shortlisted["Adjusted Expiry (with PTE)"] = pd.to_datetime(
        shortlisted["Adjusted Expiry (with PTE)"], errors="coerce"
    )
    shortlisted = (
        shortlisted
        .sort_values("Adjusted Expiry (with PTE)", ascending=False, na_position="last")
        .drop_duplicates(subset=["Drug Name", "Jurisdiction"], keep="first")
        .sort_values(["Drug Name", "Jurisdiction"])
        .reset_index(drop=True)
    )
    print(f"\nAfter Deduplication         : {len(shortlisted)} (removed {before_dedup - len(shortlisted)})")

    # ── 5. Expiry Gap ────────────────────────────────────────────────────
    def get_expiry_gap(row):
        if row["Jurisdiction"] == "US" and years_to_entry_map:
            key = (row["Drug Name"], row["Patent Number"])
            val = years_to_entry_map.get(key)
            if val is not None:
                return val
        return compute_expiry_gap(row["Adjusted Expiry (with PTE)"])

    shortlisted["Expiry Gap (Years)"] = shortlisted.apply(get_expiry_gap, axis=1)

    # ── 6. Arbitrage Analysis ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ARBITRAGE ANALYSIS")
    print(f"{'='*60}")
    arb_df = compute_arbitrage(shortlisted, full_filtered)

    if not arb_df.empty:
        non_ref = arb_df[arb_df["Arbitrage Score"] != "N/A"]
        for score_val in [5, 4, 3, 2, 1]:
            count = (non_ref["Arbitrage Score"] == score_val).sum()
            if count > 0:
                print(f"  Score {score_val}: {count} entries")

        print(f"\n  Dimension IV Scores:")
        for drug in arb_df["Drug Name"].unique():
            row = arb_df[arb_df["Drug Name"] == drug].iloc[0]
            print(f"    {drug}: {row['Dimension IV Score']} — {row['Dimension IV Rating']}")

    # ── 7. Merge Dimension IV Score back to shortlisted ──────────────────
    if not arb_df.empty:
        dim4_map = (
            arb_df[["Drug Name", "Dimension IV Score", "Dimension IV Rating"]]
            .drop_duplicates(subset=["Drug Name"])
        )
        shortlisted = shortlisted.merge(dim4_map, on="Drug Name", how="left")

    # ── 8. Final columns ─────────────────────────────────────────────────
    final_cols = [
        "Drug Name", "Patent Number", "Jurisdiction", "Tag",
        "Step 1 Claim Category",
        "Filing Date", "Grant Date", "Approval Date",
        "Base Patent Term (Filing+20y)",
        "PTE Status", "PTE Months (Granted)", "PTE Source",
        "Adjusted Expiry (with PTE)", "PTE Capped (Post-Approval Cap)",
        "Expiry Gap (Years)",
        "Dimension IV Score", "Dimension IV Rating",
    ]
    final_cols = [c for c in final_cols if c in shortlisted.columns]
    shortlisted = shortlisted[final_cols]

    # ── 9. Save to BigQuery ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"WRITING TO BIGQUERY")
    print(f"{'='*60}")

    write_to_bigquery(shortlisted, "shortlisted_secondary_patents_table")
    if not arb_df.empty:
        write_to_bigquery(arb_df, "arbitrage_summary_table")

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows loaded           : {total}")
    print(f"After Jurisdiction filter   : {f1.sum()}")
    print(f"After Grant Date filter     : {f2.sum()}")
    print(f"After Tag=BLOCKING filter   : {f3.sum()}")
    print(f"After Claim Category filter : {f4.sum()}")
    print(f"After Deduplication         : {len(shortlisted)}")
    print(f"{'='*60}")
    print(f"Jurisdictions    : {shortlisted['Jurisdiction'].value_counts().to_dict()}")
    print(f"Claim Categories : {shortlisted['Step 1 Claim Category'].value_counts().to_dict()}")
    print(f"{'='*60}")
    print(f"\n✓ Done — tables written to dataset `{BQ_DATASET_ID}`")


if __name__ == "__main__":
    main()
