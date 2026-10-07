from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.source_failures import mark_source_success, record_source_failure
from ai_stock_discovery.timeutils import utc_now_iso
from ai_stock_discovery.sources import valuation


FMP_PROFILE_SOURCE_NAME = "Financial Modeling Prep profile API"
FMP_PROFILE_URL = "https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"

AI_PROFILE_NAME_TERMS: dict[str, int] = {
    "semiconductor": 10,
    "micro devices": 9,
    "super micro": 9,
    "advanced micro": 8,
    "photonics": 8,
    "optoelectronics": 8,
    "network": 7,
    "networks": 7,
    "cyber": 7,
    "security": 7,
    "automation": 6,
    "cloud": 6,
    "data": 6,
    "electric power": 6,
    "advanced energy": 6,
    "electronics": 5,
    "communications": 5,
    "software": 5,
    "power": 4,
    "energy": 3,
    "systems": 2,
    "technology": 2,
}

PROFILE_NAME_EXCLUSION_TERMS = {
    "acquisition",
    "blank check",
    "capital acquisition",
    "etf",
    "fund",
    "rights",
    "trust",
    "unit",
    "warrant",
}


@dataclass(frozen=True)
class CompanyProfile:
    ticker: str
    company_name: str | None = None
    cik: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    description: str | None = None
    website: str | None = None
    ir_url: str | None = None
    source: str = "manual"


@dataclass(frozen=True)
class ProfileEnrichmentCandidate:
    ticker: str
    company_name: str
    matched_terms: tuple[str, ...]
    score: int
    source: str


@dataclass(frozen=True)
class FmpProfileEnrichmentSummary:
    candidates_considered: int
    profiles_written: int
    valuation_snapshots_written: int
    failures: tuple[str, ...]


def select_ai_profile_enrichment_candidates(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 25,
    include_existing: bool = False,
) -> list[ProfileEnrichmentCandidate]:
    if limit <= 0:
        return []
    explicit_tickers = _unique_tickers(tickers or [])
    if tickers:
        if not explicit_tickers:
            return []
        rows = conn.execute(
            """
            SELECT ticker, company_name, source, description
            FROM company_profile
            WHERE ticker IN (%s)
            """
            % _sql_placeholders(explicit_tickers),
            explicit_tickers,
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT ticker, company_name, source, description
            FROM company_profile
            WHERE COALESCE(company_name, '') <> ''
              AND (? = 1 OR COALESCE(description, '') = '')
              AND NOT EXISTS (
                  SELECT 1
                  FROM universe_exclusions ex
                  WHERE ex.ticker = company_profile.ticker
                    AND ex.is_active = 1
                    AND ex.severity = 'exclude'
              )
            """,
            (1 if include_existing else 0,),
        ).fetchall()

    profile_names = {
        str(row["ticker"] or "").strip().upper(): _normalize_name(str(row["company_name"] or ""))
        for row in rows
    }
    candidates: list[ProfileEnrichmentCandidate] = []
    found_tickers: set[str] = set()
    for row in rows:
        ticker = str(row["ticker"] or "").strip().upper()
        name = str(row["company_name"] or "").strip()
        if not ticker or not name:
            continue
        found_tickers.add(ticker)
        if not explicit_tickers and _is_likely_derivative_ticker(ticker, name, profile_names):
            continue
        if not explicit_tickers and not include_existing and str(row["description"] or "").strip():
            continue
        score, matched_terms = _profile_name_score(name)
        if not score and not explicit_tickers:
            continue
        candidates.append(
            ProfileEnrichmentCandidate(
                ticker=ticker,
                company_name=name,
                matched_terms=tuple(matched_terms),
                score=score,
                source=str(row["source"] or "company_profile"),
            )
        )
    for ticker in explicit_tickers:
        if ticker in found_tickers:
            continue
        candidates.append(
            ProfileEnrichmentCandidate(
                ticker=ticker,
                company_name=ticker,
                matched_terms=(),
                score=0,
                source="explicit_ticker",
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.ticker, item.company_name))
    return candidates[:limit]


def enrich_fmp_profile_candidates(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    api_key: str,
    tickers: list[str] | None = None,
    limit: int = 25,
    include_existing: bool = False,
    write_valuation_snapshot: bool = True,
) -> FmpProfileEnrichmentSummary:
    candidates = select_ai_profile_enrichment_candidates(
        conn,
        tickers=tickers,
        limit=limit,
        include_existing=include_existing,
    )
    profiles_written = 0
    valuation_snapshots_written = 0
    failures: list[str] = []
    for candidate in candidates:
        try:
            company_profile = fetch_fmp_profile(client, candidate.ticker, api_key)
            profiles_written += upsert_company_profiles(conn, [company_profile])
            if write_valuation_snapshot and company_profile.market_cap is not None:
                valuation_snapshots_written += valuation.upsert_valuation_snapshots(
                    conn,
                    [
                        valuation.ValuationSnapshot(
                            ticker=company_profile.ticker,
                            date=utc_now_iso()[:10],
                            market_cap=company_profile.market_cap,
                            source=company_profile.source,
                        )
                    ],
                )
            mark_source_success(
                conn,
                source_name=FMP_PROFILE_SOURCE_NAME,
                ticker=candidate.ticker,
                endpoint="profile",
            )
        except Exception as exc:
            failure = f"{candidate.ticker} profile: {_sanitize(str(exc), api_key)}"
            failures.append(failure)
            record_source_failure(
                conn,
                source_name=FMP_PROFILE_SOURCE_NAME,
                ticker=candidate.ticker,
                endpoint="profile",
                reason=failure,
                source_url=FMP_PROFILE_URL.format(ticker=candidate.ticker, api_key="***"),
            )
    return FmpProfileEnrichmentSummary(
        candidates_considered=len(candidates),
        profiles_written=profiles_written,
        valuation_snapshots_written=valuation_snapshots_written,
        failures=tuple(failures),
    )


def upsert_company_profiles(conn: sqlite3.Connection, profiles: list[CompanyProfile]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO company_profile (
            ticker, cik, company_name, sector, industry, market_cap,
            description, website, ir_url, source, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            cik=COALESCE(excluded.cik, company_profile.cik),
            company_name=COALESCE(excluded.company_name, company_profile.company_name),
            sector=COALESCE(excluded.sector, company_profile.sector),
            industry=COALESCE(excluded.industry, company_profile.industry),
            market_cap=COALESCE(excluded.market_cap, company_profile.market_cap),
            description=COALESCE(excluded.description, company_profile.description),
            website=COALESCE(excluded.website, company_profile.website),
            ir_url=COALESCE(excluded.ir_url, company_profile.ir_url),
            source=excluded.source,
            updated_at=excluded.updated_at
        """,
        [
            (
                profile.ticker.upper(),
                profile.cik,
                profile.company_name,
                profile.sector,
                profile.industry,
                profile.market_cap,
                profile.description,
                profile.website,
                profile.ir_url,
                profile.source,
                now,
            )
            for profile in profiles
        ],
    )
    return len(profiles)


def import_company_profiles_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    profiles: list[CompanyProfile] = []
    default_source = f"manual_profile_csv:{csv_path.name}"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = _field(row, "ticker", "Ticker")
            if not ticker:
                continue
            profiles.append(
                CompanyProfile(
                    ticker=ticker,
                    company_name=_none(_field(row, "company_name", "name", "companyName")),
                    cik=_none(_field(row, "cik", "CIK")),
                    sector=_none(_field(row, "sector")),
                    industry=_none(_field(row, "industry")),
                    market_cap=_number(_field(row, "market_cap", "marketCap")),
                    description=_none(_field(row, "description")),
                    website=_none(_field(row, "website")),
                    ir_url=_none(_field(row, "ir_url", "irUrl")),
                    source=_field(row, "source") or default_source,
                )
            )
    return upsert_company_profiles(conn, profiles)


def fetch_fmp_profile(client: HttpClient, ticker: str, api_key: str) -> CompanyProfile:
    ticker = ticker.upper()
    try:
        payload = client.get_json(FMP_PROFILE_URL.format(ticker=ticker, api_key=api_key))
    except FetchError as exc:
        raise FetchError(str(exc).replace(api_key, "***")) from exc
    result = _first_fmp_profile(payload)
    if not result:
        raise ValueError(f"No FMP profile result returned for {ticker}")
    return CompanyProfile(
        ticker=ticker,
        company_name=_none(result.get("companyName")),
        cik=_none(result.get("cik")),
        sector=_none(result.get("sector")),
        industry=_none(result.get("industry")),
        market_cap=_number(result.get("marketCap") or result.get("mktCap")),
        description=_none(result.get("description")),
        website=_none(result.get("website")),
        ir_url=None,
        source=FMP_PROFILE_URL.format(ticker=ticker, api_key="***"),
    )


def _first_fmp_profile(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    return first if isinstance(first, dict) else None


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_name_score(name: str) -> tuple[int, list[str]]:
    normalized = _normalize_name(name)
    if any(_contains_term(normalized, term) for term in PROFILE_NAME_EXCLUSION_TERMS):
        return 0, []
    matched: list[str] = []
    score = 0
    for term, weight in AI_PROFILE_NAME_TERMS.items():
        if _contains_term(normalized, term):
            matched.append(term)
            score += weight
    return score, matched


def _normalize_name(value: str) -> str:
    return " ".join(
        value.lower()
        .replace("&", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(".", " ")
        .split()
    )


def _contains_term(text: str, term: str) -> bool:
    padded = f" {text} "
    normalized_term = _normalize_name(term)
    return f" {normalized_term} " in padded


def _is_likely_derivative_ticker(
    ticker: str,
    company_name: str,
    profile_names: dict[str, str],
) -> bool:
    normalized_name = _normalize_name(company_name)
    if "-" in ticker and ticker.rsplit("-", 1)[-1] in {"UN", "U", "WT", "W", "WS", "RT", "R"}:
        return True
    if len(ticker) <= 2 or ticker[-1] not in {"U", "W", "Z", "R"}:
        return False
    base = ticker[:-1]
    return profile_names.get(base) == normalized_name


def _sql_placeholders(values: list[str]) -> str:
    return ",".join("?" for _ in values) or "NULL"


def _unique_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for ticker in tickers:
        value = ticker.strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _sanitize(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text
