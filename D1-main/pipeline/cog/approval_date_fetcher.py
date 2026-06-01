"""
approval_date_fetcher.py
─────────────────────────
Handles fetching real-world drug approval dates for marketed jurisdictions.

Three-step cascade per jurisdiction:
  Step A — Official API  (FDA Drugs@FDA / EMA EPAR)
  Step B — Gemini Search grounding
  Step C — Pharma news outlet scraping (Fierce Pharma, BioSpace, etc.)

Also handles:
  - Brand name resolution (FDA API / EMA API / BigQuery)
  - Date normalisation to DD-MMM-YYYY
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.oauth2 import service_account

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

BQ_PROJECT_ID      = os.getenv("BQ_PROJECT_ID")
BQ_DATASET_ID      = os.getenv("BQ_DATASET_ID")
BQ_SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
BQ_BRANDS_TABLE    = os.getenv("BQ_BRANDS_TABLE")

class _LazyGenaiClient:
    """Lazily instantiate the real genai.Client on first attribute access so
    importing this module does not require GOOGLE_API_KEY/GEMINI_API_KEY."""
    _client = None

    def _resolve(self):
        if _LazyGenaiClient._client is None:
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY must be set to use the "
                    "Gemini client."
                )
            _LazyGenaiClient._client = genai.Client(api_key=key)
        return _LazyGenaiClient._client

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


gemini_client = _LazyGenaiClient()

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
_HTTP_DELAY   = 0.7

_PHARMA_OUTLETS = [
    ("Fierce Pharma",  "https://www.fiercepharma.com/search?q={q}"),
    ("BioSpace",       "https://www.biospace.com/search/?q={q}"),
    ("PBR",            "https://www.pharmaceutical-business-review.com/search?q={q}"),
    ("Endpoints News", "https://endpts.com/?s={q}"),
    ("Reuters Health", "https://www.reuters.com/site-search/?query={q}&section=healthcare"),
]

_APPROVAL_GEO = {
    "US": {
        "regulator":  "US FDA",
        "approval":   "FDA approval / NDA or BLA",
        "exclude":    "NOT the EU authorisation",
        "reg_type":   "NDA or BLA approval announcements",
        "news_query": "{brand} FDA approved United States",
        "signals":    re.compile(
            r"FDA approved|FDA approval|NDA approved|BLA approved|"
            r"U\.S\. approval|approved (?:in|by) (?:the )?(?:US|United States)",
            re.IGNORECASE,
        ),
    },
    "EU": {
        "regulator":  "European Commission / EMA",
        "approval":   "EU marketing authorisation",
        "exclude":    "NOT the US FDA approval",
        "reg_type":   "MAA approval announcements",
        "news_query": "{brand} EU approval European Commission",
        "signals":    re.compile(
            r"European Commission|EC approval|EU approval|"
            r"marketing authoris|EMA approv|approved in (?:the )?Europe",
            re.IGNORECASE,
        ),
    },
}

_DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})\b",
    r"\b((?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{2}/\d{2}/\d{4})\b",
]

_DOMAIN_REACHABLE_CACHE: Dict[str, bool] = {}


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _find_date_in_text(text: str) -> Optional[str]:
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def format_approval_date(raw: Optional[str]) -> Optional[str]:
    """Normalises approval dates to DD-MMM-YYYY (e.g. 05-Dec-2017)."""
    if not raw or str(raw).lower() in ("none", "null", "n/a", ""):
        return None
    raw = str(raw).strip()
    formats = [
        "%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d %B %Y", "%B %d, %Y", "%B %d %Y", "%d-%b-%Y", "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%d-%b-%Y")
        except ValueError:
            continue
    return raw


def _http_get(url: str, **kwargs) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=15, **kwargs)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"[APPROVAL HTTP] {e}")
        return None


def _is_domain_reachable(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    if host in _DOMAIN_REACHABLE_CACHE:
        return _DOMAIN_REACHABLE_CACHE[host]
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        _DOMAIN_REACHABLE_CACHE[host] = True
        print(f"[NETWORK] {host} is reachable")
        return True
    except (socket.error, OSError):
        _DOMAIN_REACHABLE_CACHE[host] = False
        print(f"[NETWORK] {host} is NOT reachable — skipping Step A")
        return False


# ─────────────────────────────────────────────
# Brand name resolution
# ─────────────────────────────────────────────

def _resolve_brands_fda(generic: str) -> List[str]:
    r = _http_get("https://api.fda.gov/drug/drugsfda.json", params={
        "search": f'openfda.generic_name:"{generic}" OR openfda.substance_name:"{generic}"',
        "limit": 10,
    })
    if not r:
        return []
    seen, brands = set(), []
    for result in r.json().get("results", []):
        for b in result.get("openfda", {}).get("brand_name", []):
            if b.upper() not in seen:
                seen.add(b.upper())
                brands.append(b)
    time.sleep(_HTTP_DELAY)
    return brands


def _resolve_brands_ema(generic: str) -> List[str]:
    r = _http_get(
        "https://www.ema.europa.eu/en/medicines/search_api",
        params={"search_api_fulltext": generic, "f[0]": "field_ema_web_categories:25"},
    )
    brands = []
    if r:
        if "application/json" in r.headers.get("Content-Type", ""):
            for item in (r.json().get("rows") or r.json().get("results") or []):
                name = (item.get("title") or item.get("name") or "").strip()
                if name and name.upper() not in {b.upper() for b in brands}:
                    brands.append(name)
        else:
            for article in BeautifulSoup(r.text, "html.parser").select("article, .search-result"):
                h = article.find(["h2", "h3", "h4"])
                if h:
                    name = h.get_text(strip=True)
                    if name and name.upper() not in {b.upper() for b in brands}:
                        brands.append(name)
    if not brands:
        brands.append(generic)
    time.sleep(_HTTP_DELAY)
    return brands


def fetch_brands_from_bq(drug_name: str) -> Dict[str, List[str]]:
    """Fetches brand names for a drug from the BQ brands table, split by US/EU."""
    if not (BQ_PROJECT_ID and BQ_DATASET_ID and BQ_BRANDS_TABLE):
        print(f"[BRANDS BQ] BQ_BRANDS_TABLE not configured")
        return {"US": [], "EU": []}

    try:
        if BQ_SERVICE_ACCOUNT:
            credentials = service_account.Credentials.from_service_account_file(
                BQ_SERVICE_ACCOUNT,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        else:
            client = bigquery.Client(project=BQ_PROJECT_ID)

        fq_table = f"`{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_BRANDS_TABLE}`"
        query = f"""
        SELECT DISTINCT cleaned_generic_name, Brand_Name, Drug_Geography
        FROM {fq_table}
        WHERE Brand_Name IS NOT NULL
          AND TRIM(Brand_Name) <> ''
          AND LOWER(cleaned_generic_name) = LOWER(@drug_name)
          AND (
            UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
            OR UPPER(cleaned_Target) LIKE '%GLP-1%'
            OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
            OR (data_source = 'IPD' AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist')
          )
          AND LOWER(highest_development_stage) = 'marketed'
        ORDER BY cleaned_generic_name, Brand_Name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)
            ]
        )
        df = client.query(query, job_config=job_config).to_dataframe()
        print(f"[BRANDS BQ] '{drug_name}' → {len(df)} row(s)")

        result: Dict[str, List[str]] = {"US": [], "EU": []}
        for _, row in df.iterrows():
            brand   = str(row.get("Brand_Name") or "").strip()
            geo_raw = str(row.get("Drug_Geography") or "").strip()
            if not brand:
                continue
            for geo in [g.strip() for g in re.split(r"[,;]", geo_raw) if g.strip()]:
                geo_upper = geo.upper()
                if geo_upper in ("UNITED STATES", "US", "USA") and brand not in result["US"]:
                    result["US"].append(brand)
                elif geo_upper in ("EU", "EUROPE", "EUROPEAN UNION") and brand not in result["EU"]:
                    result["EU"].append(brand)

        print(f"[BRANDS BQ] US: {result['US']} | EU: {result['EU']}")
        return result

    except Exception as e:
        print(f"[BRANDS BQ] Query failed for '{drug_name}': {e}")
        return {"US": [], "EU": []}


# ─────────────────────────────────────────────
# Step A — Official APIs
# ─────────────────────────────────────────────

def _approval_step_a_fda(brand: str) -> Tuple[Optional[str], str]:
    r = _http_get(
        "https://api.fda.gov/drug/drugsfda.json",
        params={"search": f'openfda.brand_name:"{brand}"', "limit": 5},
    )
    if not r:
        time.sleep(_HTTP_DELAY)
        return None, "FDA: not found"

    results = r.json().get("results", [])
    if not results:
        time.sleep(_HTTP_DELAY)
        return None, "FDA: not found"

    original_approval_dates = []
    for app in results:
        app_type = app.get("application_number", "")[:3].upper()
        if app_type not in ("NDA", "BLA"):
            continue
        for sub in app.get("submissions", []):
            if sub.get("submission_status") == "AP" and sub.get("submission_type") in ("N", "NDA", "BLA"):
                if sub.get("submission_status_date"):
                    original_approval_dates.append(sub["submission_status_date"])

    if not original_approval_dates:
        for app in results:
            for sub in app.get("submissions", []):
                if sub.get("submission_status") == "AP" and sub.get("submission_status_date"):
                    original_approval_dates.append(sub["submission_status_date"])

    time.sleep(_HTTP_DELAY)
    if original_approval_dates:
        earliest = min(original_approval_dates)
        return earliest, "FDA Drugs@FDA API"
    return None, "FDA: not found"


def _parse_ema_date_from_page(html: str, url: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")

    _HIGH_PRIORITY = re.compile(
        r"^(?:marketing authorisation date|date of (?:first )?authorisation|"
        r"initial authorisation date|first authorised)$", re.I,
    )
    _IGNORE = re.compile(
        r"commission decision|ec decision|chmp opinion|opinion date|"
        r"renewal|revision|variation|re-examination|withdrawal|"
        r"extension of indication|new indication|line extension|new formulation", re.I,
    )

    def _from_sibling(tag) -> Optional[str]:
        sib = tag.find_next_sibling(["dd", "td"])
        return _find_date_in_text(sib.get_text(" ", strip=True)) if sib else None

    for tag_type in ("dt", "th"):
        for tag in soup.find_all(tag_type):
            label = tag.get_text(strip=True)
            if _IGNORE.search(label):
                continue
            if _HIGH_PRIORITY.search(label):
                d = _from_sibling(tag)
                if d:
                    return d

    for strong in soup.find_all(["strong", "b"]):
        label = strong.get_text(strip=True)
        if not _IGNORE.search(label) and _HIGH_PRIORITY.search(label):
            d = _find_date_in_text(strong.parent.get_text(" ", strip=True))
            if d:
                return d

    _BROAD = re.compile(r"authorisation date|authorised", re.I)
    for tag in soup.find_all(["dt", "th"]):
        label = tag.get_text(strip=True)
        if not _IGNORE.search(label) and _BROAD.search(label):
            d = _from_sibling(tag)
            if d:
                return d

    text = soup.get_text(" ", strip=True)
    for pattern in [
        r"marketing authorisation date[:\s]+([^\n]{5,40})",
        r"date of (?:first )?authorisation[:\s]+([^\n]{5,40})",
        r"first authorised[:\s]+([^\n]{5,30})",
    ]:
        for m in re.finditer(pattern, text, re.I):
            if not _IGNORE.search(text[max(0, m.start()-80): m.start()]):
                d = _find_date_in_text(m.group(1))
                if d:
                    return d

    return None


def _approval_step_a_ema(brand: str) -> Tuple[Optional[str], str]:
    epar_slug  = brand.lower().replace(" ", "-")
    direct_url = f"https://www.ema.europa.eu/en/medicines/human/EPAR/{epar_slug}"
    page = _http_get(direct_url)
    if page:
        d = _parse_ema_date_from_page(page.text, direct_url)
        if d:
            return d, f"EMA EPAR ({direct_url})"

    r = _http_get(
        "https://www.ema.europa.eu/en/medicines/search_api",
        params={"search_api_fulltext": brand, "f[0]": "field_ema_web_categories:25"},
    )
    if r:
        epar_url = None
        if "application/json" in r.headers.get("Content-Type", ""):
            items = r.json().get("rows") or r.json().get("results") or []
            path  = items[0].get("url", "") if items else ""
            epar_url = "https://www.ema.europa.eu" + path if path else None
        else:
            link = BeautifulSoup(r.text, "html.parser").select_one("a[href*='/medicines/human/EPAR/']")
            if link:
                epar_url = "https://www.ema.europa.eu" + link["href"]

        if epar_url:
            page = _http_get(epar_url)
            if page:
                d = _parse_ema_date_from_page(page.text, epar_url)
                if d:
                    return d, f"EMA EPAR ({epar_url})"

    time.sleep(_HTTP_DELAY)
    nat_url = (
        "https://www.ema.europa.eu/en/medicines/national-registers-authorised-medicines"
        f"?search_api_fulltext={quote_plus(brand)}"
    )
    page = _http_get(nat_url)
    if page:
        d = _parse_ema_date_from_page(page.text, nat_url)
        if d:
            return d, f"EMA national register ({nat_url})"

    return None, "EMA: not found"


# ─────────────────────────────────────────────
# Step B — Gemini search grounding
# ─────────────────────────────────────────────

def _approval_step_b_gemini(
    brand: str, companies: List[str], geo: str
) -> Tuple[Optional[str], str]:
    g             = _APPROVAL_GEO[geo]
    companies_str = ", ".join(companies) if companies else "the manufacturer"
    prompt = (
        f'Find the FIRST/ORIGINAL {g["approval"]} date for brand "{brand}" '
        f'by manufacturer(s) {companies_str}.\n'
        f'CRITICAL RULES:\n'
        f'- Return ONLY the date of the FIRST/ORIGINAL marketing authorisation — '
        f'NOT a later variation, line extension, new indication, new formulation, or renewal.\n'
        f'- The date must be {g["approval"]} — {g["exclude"]}.\n'
        f'- If you can only find a date for a variation or new indication (not the original approval), '
        f'return null.\n'
        f'Respond ONLY as JSON: {{"approval_date":"<YYYY-MM-DD or null>",'
        f'"source_url":"<url or null>","source_type":"<press_release|investor_relations'
        f'|regulatory_news|not_found>","confidence":"<high|medium|low>",'
        f'"notes":"<one sentence confirming this is the ORIGINAL approval date>"}}'
    )
    try:
        response = gemini_client.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = prompt,
            config   = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        raw   = response.text.strip()
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        p     = json.loads(clean)
        d     = p.get("approval_date")
        if d and str(d).lower() not in ("null", "none", "n/a", ""):
            return str(d), (
                f"Gemini [{p.get('source_type')}, {p.get('confidence')}] | "
                f"{p.get('source_url', '')} | {p.get('notes', '')}"
            )
        return None, f"Gemini: not found – {p.get('notes', '')}"
    except json.JSONDecodeError:
        d = _find_date_in_text(raw)
        return (d, "Gemini (regex fallback)") if d else (None, "Gemini: parse error")
    except Exception as e:
        return None, f"Gemini error: {e}"


# ─────────────────────────────────────────────
# Step C — Pharma news scraping
# ─────────────────────────────────────────────

def _scrape_article_for_date(url: str, brand: str, geo: str) -> Optional[str]:
    signals = _APPROVAL_GEO[geo]["signals"]
    r = _http_get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    for m in signals.finditer(text):
        d = _find_date_in_text(text[max(0, m.start() - 60): m.end() + 200])
        if d:
            return d
    idx = text.lower().find(brand.lower())
    while idx != -1:
        window = text[max(0, idx - 30): idx + 200]
        if signals.search(window):
            d = _find_date_in_text(window)
            if d:
                return d
        idx = text.lower().find(brand.lower(), idx + 1)
    return None


def _approval_step_c_news(brand: str, geo: str) -> Tuple[Optional[str], str]:
    query = _APPROVAL_GEO[geo]["news_query"].format(brand=brand)
    for outlet_name, url_template in _PHARMA_OUTLETS:
        search_url = url_template.replace("{q}", quote_plus(query))
        r = _http_get(search_url)
        if not r:
            continue
        soup  = BeautifulSoup(r.text, "html.parser")
        links, seen = [], set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            txt  = a.get_text(strip=True).lower()
            if brand.lower() in txt or "approv" in txt or "authoris" in txt:
                if href.startswith("/"):
                    href = "/".join(search_url.split("/")[:3]) + href
                if href.startswith("http") and href not in seen:
                    seen.add(href)
                    links.append(href)
            if len(links) >= 4:
                break
        if not links:
            for a in soup.select("h2 a, h3 a, article a"):
                href = a.get("href", "")
                if href.startswith("http") and href not in seen:
                    seen.add(href)
                    links.append(href)
                if len(links) >= 3:
                    break
        for article_url in links:
            d = _scrape_article_for_date(article_url, brand, geo)
            if d:
                return d, f"{outlet_name} | {article_url}"
            time.sleep(_HTTP_DELAY)
    return None, "Step C: not found"


# ─────────────────────────────────────────────
# Orchestrator per brand
# ─────────────────────────────────────────────

def _get_approval_date_for_brand(
    brand: str, companies: List[str], geo: str
) -> Tuple[Optional[str], str]:
    print(f"[APPROVAL] [{geo}] '{brand}'")

    if geo == "US":
        if _is_domain_reachable("api.fda.gov"):
            date, src = _approval_step_a_fda(brand)
            if date:
                return date, src
        else:
            print(f"[APPROVAL] [{geo}] Skipping Step A — api.fda.gov unreachable")
    else:
        if _is_domain_reachable("www.ema.europa.eu"):
            date, src = _approval_step_a_ema(brand)
            if date:
                return date, src
        else:
            print(f"[APPROVAL] [{geo}] Skipping Step A — ema.europa.eu unreachable")

    print(f"[APPROVAL] [{geo}] Step A failed → Step B (Gemini)")
    date, src = _approval_step_b_gemini(brand, companies, geo)
    if date:
        return date, src

    print(f"[APPROVAL] [{geo}] Step B failed → Step C (pharma news)")
    return _approval_step_c_news(brand, geo)


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

async def fetch_approval_dates(
    drug_name:    str,
    bq_companies: List[str],
    bq_brands:    List[str],
    fetch_us:     bool = True,
    fetch_eu:     bool = True,
) -> Dict[str, Dict]:
    """
    Fetches approval dates only for jurisdictions where phase is Marketed.

    Args:
        drug_name:    Generic drug name
        bq_companies: Company names from timeline (for Gemini search context)
        bq_brands:    Brand names from timeline (fallback if BQ brands table empty)
        fetch_us:     True if US phase is Marketed
        fetch_eu:     True if EU phase is Marketed

    Returns:
        {
          "US": {"date": str|None, "source": str, "brands": list},
          "EU": {"date": str|None, "source": str, "brands": list},
        }
    """
    result = {
        "US": {"date": None, "source": "Not Marketed — skipped", "brands": []},
        "EU": {"date": None, "source": "Not Marketed — skipped", "brands": []},
    }

    if not fetch_us and not fetch_eu:
        print(f"[APPROVAL] '{drug_name}' — not Marketed in any jurisdiction, skipping")
        return result

    loop        = asyncio.get_event_loop()
    bq_brand_map = await loop.run_in_executor(None, lambda: fetch_brands_from_bq(drug_name))

    for geo, should_fetch in (("US", fetch_us), ("EU", fetch_eu)):
        if not should_fetch:
            print(f"[APPROVAL] [{geo}] '{drug_name}' — not Marketed, skipping")
            continue

        brands = bq_brand_map.get(geo, [])
        if not brands:
            brands = [b.strip() for b in bq_brands if b.strip()]
            if brands:
                print(f"[BRANDS BQ] [{geo}] No BQ brands — using timeline brands: {brands}")
        if not brands:
            brands = [drug_name]
            print(f"[BRANDS BQ] [{geo}] No brands found — using generic name '{drug_name}'")

        result[geo]["brands"] = brands
        print(f"[APPROVAL] [{geo}] '{drug_name}' → trying brands: {brands}")

        found = []
        for brand in brands:
            date, src = await loop.run_in_executor(
                None,
                lambda b=brand, g=geo: _get_approval_date_for_brand(b, bq_companies, g),
            )
            if date:
                formatted = format_approval_date(date)
                if formatted and formatted != "N/A":
                    found.append((formatted, src, brand))
                    print(f"[APPROVAL] [{geo}] '{brand}' → {formatted}")

        if found:
            def _parse_for_sort(entry):
                try:
                    return datetime.strptime(entry[0], "%d-%b-%Y")
                except ValueError:
                    return datetime.max

            earliest = min(found, key=_parse_for_sort)
            result[geo]["date"]   = earliest[0]
            result[geo]["source"] = f"{earliest[2]} | {earliest[1]}"
            print(
                f"[APPROVAL] [{geo}] Earliest across {len(found)} brand(s): "
                f"{earliest[0]} ({earliest[2]})"
            )
        else:
            result[geo]["source"] = "Not found after all steps"

    return result
