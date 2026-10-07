from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3

from ai_stock_discovery.sources.nasdaq import NASDAQ_LISTED_URL, OTHER_LISTED_URL
from ai_stock_discovery.timeutils import utc_now_iso


DEFAULT_MIN_MARKET_CAP = 50_000_000.0

EXCLUDED_NAME_TERMS: tuple[tuple[str, str], ...] = (
    ("warrant", "warrant"),
    ("right", "right"),
    ("preferred", "preferred"),
    ("pfd", "preferred"),
    ("depositary", "depositary_share"),
    ("unit", "unit"),
    ("etn", "exchange_traded_note"),
    ("etf", "exchange_traded_fund"),
    ("fund", "fund"),
    ("trust", "trust"),
)

SPAC_NAME_TERMS: tuple[str, ...] = (
    "acquisition corp",
    "acquisition corporation",
    "acquisition company",
    "acquisition inc",
    "acquisition incorporated",
    "acquisition ltd",
    "acquisition limited",
    "blank check",
    "spac",
)


@dataclass(frozen=True)
class UniverseExclusion:
    ticker: str
    reason_type: str
    reason: str
    source_name: str
    source_url: str
    severity: str
    method: str
    content_hash: str


@dataclass(frozen=True)
class UniverseFilterResult:
    recorded: int
    active_rule_exclusions: int


def apply_universe_filters(
    conn: sqlite3.Connection,
    *,
    min_market_cap: float | None = DEFAULT_MIN_MARKET_CAP,
) -> UniverseFilterResult:
    exclusions = _collect_universe_exclusions(conn)
    if min_market_cap is not None:
        exclusions.extend(_collect_low_market_cap_exclusions(conn, min_market_cap=min_market_cap))

    conn.execute("UPDATE universe_exclusions SET is_active = 0 WHERE method = 'rule'")
    upsert_universe_exclusions(conn, exclusions)
    active_rule_exclusions = int(
        conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM universe_exclusions
            WHERE method = 'rule' AND is_active = 1
            """
        ).fetchone()["count"]
        or 0
    )
    return UniverseFilterResult(recorded=len(exclusions), active_rule_exclusions=active_rule_exclusions)


def upsert_universe_exclusions(
    conn: sqlite3.Connection,
    exclusions: list[UniverseExclusion],
) -> int:
    if not exclusions:
        return 0
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO universe_exclusions (
            ticker, reason_type, reason, source_name, source_url,
            severity, method, is_active, detected_at, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            reason=excluded.reason,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            severity=excluded.severity,
            method=excluded.method,
            is_active=1,
            detected_at=excluded.detected_at
        """,
        [
            (
                exclusion.ticker,
                exclusion.reason_type,
                exclusion.reason,
                exclusion.source_name,
                exclusion.source_url,
                exclusion.severity,
                exclusion.method,
                now,
                exclusion.content_hash,
            )
            for exclusion in exclusions
        ],
    )
    return len(exclusions)


def list_universe_exclusions(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    active_only: bool = True,
    limit: int = 50,
) -> list[sqlite3.Row]:
    where: list[str] = []
    params: list[object] = []
    if ticker:
        where.append("ticker = ?")
        params.append(ticker.upper())
    if active_only:
        where.append("is_active = 1")
    clause = "WHERE " + " AND ".join(where) if where else ""
    params.append(limit)
    return conn.execute(
        f"""
        SELECT ticker, reason_type, reason, source_name, source_url,
               severity, method, is_active, detected_at
        FROM universe_exclusions
        {clause}
        ORDER BY is_active DESC, detected_at DESC, ticker, reason_type
        LIMIT ?
        """,
        params,
    ).fetchall()


def _collect_universe_exclusions(conn: sqlite3.Connection) -> list[UniverseExclusion]:
    rows = conn.execute(
        """
        SELECT ticker, company_name, exchange, is_etf, is_preferred, is_unit, is_active
        FROM universe
        """
    ).fetchall()
    exclusions: list[UniverseExclusion] = []
    for row in rows:
        ticker = str(row["ticker"]).upper()
        company_name = row["company_name"] or ""
        source_url = _universe_source_url(row["exchange"])
        source_name = "Nasdaq Trader Symbol Directory"

        if int(row["is_active"] or 0) == 0:
            exclusions.append(
                _make_exclusion(
                    ticker=ticker,
                    reason_type="inactive_security",
                    reason="Nasdaq Trader marks this security as inactive or a test issue.",
                    source_name=source_name,
                    source_url=source_url,
                )
            )
        if int(row["is_etf"] or 0) == 1:
            exclusions.append(
                _make_exclusion(
                    ticker=ticker,
                    reason_type="etf_or_fund",
                    reason="Nasdaq Trader marks this security as an ETF/fund-like instrument.",
                    source_name=source_name,
                    source_url=source_url,
                )
            )
        if int(row["is_preferred"] or 0) == 1:
            exclusions.append(
                _make_exclusion(
                    ticker=ticker,
                    reason_type="preferred_security",
                    reason="Ticker/name indicates preferred stock rather than common equity.",
                    source_name=source_name,
                    source_url=source_url,
                )
            )
        if int(row["is_unit"] or 0) == 1:
            exclusions.append(
                _make_exclusion(
                    ticker=ticker,
                    reason_type="unit_security",
                    reason="Ticker/name indicates a unit rather than common equity.",
                    source_name=source_name,
                    source_url=source_url,
                )
            )
        if any(mark in ticker for mark in ("$", "^", "+", "*")):
            exclusions.append(
                _make_exclusion(
                    ticker=ticker,
                    reason_type="non_common_ticker_format",
                    reason="Ticker contains a suffix marker often used for preferreds, warrants, rights, or special issues.",
                    source_name=source_name,
                    source_url=source_url,
                )
            )

        normalized_name = _normalize_text(company_name)
        for term, reason_type in EXCLUDED_NAME_TERMS:
            if _contains_term(normalized_name, term):
                exclusions.append(
                    _make_exclusion(
                        ticker=ticker,
                        reason_type=reason_type,
                        reason=f"Company/security name contains '{term}', suggesting this is not plain common equity.",
                        source_name=source_name,
                        source_url=source_url,
                    )
                )
                break
        for term in SPAC_NAME_TERMS:
            if term in normalized_name:
                exclusions.append(
                    _make_exclusion(
                        ticker=ticker,
                        reason_type="blank_check_or_spac",
                        reason=f"Company name contains '{term}', suggesting a SPAC or blank-check security.",
                        source_name=source_name,
                        source_url=source_url,
                    )
                )
                break
    exclusions.extend(_collect_profile_name_exclusions(conn))
    return exclusions


def _collect_profile_name_exclusions(conn: sqlite3.Connection) -> list[UniverseExclusion]:
    rows = conn.execute(
        """
        SELECT ticker, company_name, source
        FROM company_profile
        WHERE COALESCE(company_name, '') <> ''
        """
    ).fetchall()
    exclusions: list[UniverseExclusion] = []
    for row in rows:
        ticker = str(row["ticker"]).upper()
        company_name = row["company_name"] or ""
        source_url = row["source"] or "company_profile"
        source_name = _profile_source_name(source_url)
        normalized_name = _normalize_text(company_name)

        for term, reason_type in EXCLUDED_NAME_TERMS:
            if _contains_term(normalized_name, term):
                exclusions.append(
                    _make_exclusion(
                        ticker=ticker,
                        reason_type=reason_type,
                        reason=f"Company profile name contains '{term}', suggesting this is not plain common equity.",
                        source_name=source_name,
                        source_url=source_url,
                    )
                )
                break
        for term in SPAC_NAME_TERMS:
            if term in normalized_name:
                exclusions.append(
                    _make_exclusion(
                        ticker=ticker,
                        reason_type="blank_check_or_spac",
                        reason=f"Company profile name contains '{term}', suggesting a SPAC or blank-check security.",
                        source_name=source_name,
                        source_url=source_url,
                    )
                )
                break
    return exclusions


def _collect_low_market_cap_exclusions(
    conn: sqlite3.Connection,
    *,
    min_market_cap: float,
) -> list[UniverseExclusion]:
    rows = conn.execute(
        """
        SELECT vs.ticker, vs.date, vs.market_cap, vs.source
        FROM valuation_snapshots AS vs
        WHERE vs.market_cap IS NOT NULL
          AND vs.id = (
              SELECT latest.id
              FROM valuation_snapshots AS latest
              WHERE latest.ticker = vs.ticker
                AND latest.market_cap IS NOT NULL
              ORDER BY latest.date DESC, latest.updated_at DESC, latest.id DESC
              LIMIT 1
          )
        """
    ).fetchall()
    exclusions: list[UniverseExclusion] = []
    for row in rows:
        market_cap = float(row["market_cap"])
        if market_cap >= min_market_cap:
            continue
        source_url = row["source"] or "valuation_snapshots"
        exclusions.append(
            _make_exclusion(
                ticker=str(row["ticker"]).upper(),
                reason_type="microcap",
                reason=(
                    f"Latest valuation snapshot market_cap={market_cap:g} on {row['date']} "
                    f"is below the configured threshold {min_market_cap:g}."
                ),
                source_name=_valuation_source_name(source_url),
                source_url=source_url,
            )
        )
    return exclusions


def _make_exclusion(
    *,
    ticker: str,
    reason_type: str,
    reason: str,
    source_name: str,
    source_url: str,
    severity: str = "exclude",
    method: str = "rule",
) -> UniverseExclusion:
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker.upper(),
                reason_type,
                reason,
                source_name,
                source_url,
                severity,
                method,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return UniverseExclusion(
        ticker=ticker.upper(),
        reason_type=reason_type,
        reason=reason,
        source_name=source_name,
        source_url=source_url,
        severity=severity,
        method=method,
        content_hash=content_hash,
    )


def _universe_source_url(exchange: str | None) -> str:
    if (exchange or "").upper() == "NASDAQ":
        return NASDAQ_LISTED_URL
    return OTHER_LISTED_URL


def _valuation_source_name(source: str) -> str:
    if "query1.finance.yahoo.com" in source or "finance.yahoo.com" in source:
        return "Yahoo Finance endpoint prototype"
    if "financialmodelingprep.com" in source:
        return "Financial Modeling Prep quote API"
    if source.startswith("manual_csv:"):
        return source
    return "Valuation snapshot source"


def _profile_source_name(source: str) -> str:
    if source == "SEC company_tickers":
        return "SEC company_tickers"
    if source.startswith("manual_csv:"):
        return source
    if "financialmodelingprep.com" in source:
        return "Financial Modeling Prep profile API"
    return "Company profile source"


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace(",", " ").split())


def _contains_term(text: str, term: str) -> bool:
    padded = f" {text} "
    return f" {term} " in padded or text.endswith(f" {term}s")
