from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

from ai_stock_discovery.analysis.industry_tags import infer_tags_for_ticker, replace_inferred_tags_for_ticker
from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.source_failures import mark_source_success, record_source_failure
from ai_stock_discovery.sources import analysts, financials, finra, profile, sec, valuation
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class SeedRefreshResult:
    tickers_considered: int
    financial_fact_periods: int = 0
    profile_records: int = 0
    valuation_snapshots: int = 0
    analyst_events: int = 0
    finra_records: int = 0
    ai_tags_written: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)


def select_ai_seed_tickers(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 25,
) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    explicit = _unique_tickers(tickers or [])
    if explicit:
        return explicit[:limit]
    rows = conn.execute(
        """
        SELECT ticker
        FROM (
            SELECT ticker, MAX(confidence) AS max_confidence, COUNT(*) AS source_count
            FROM ai_industry_tags
            GROUP BY ticker
            UNION
            SELECT ticker, MAX(confidence) AS max_confidence, COUNT(*) AS source_count
            FROM ai_relevance_signals
            WHERE ai_relevance_level >= 2
            GROUP BY ticker
        )
        WHERE ticker IS NOT NULL AND ticker != ''
        GROUP BY ticker
        ORDER BY MAX(max_confidence) DESC, SUM(source_count) DESC, ticker
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [str(row["ticker"]).upper() for row in rows]


def refresh_ai_seed_data(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    tickers: list[str],
    fmp_api_key: str | None,
    include_profile: bool = True,
    include_quote: bool = True,
    include_analyst: bool = True,
    include_finra: bool = True,
    include_sec_facts: bool = True,
    include_fmp_financials: bool = True,
    analyst_period: str = "annual",
    analyst_limit: int = 3,
    financial_period: str = "annual",
    financial_limit: int = 4,
    finra_lookback_days: int = 10,
    finra_file_code: str = finra.DEFAULT_DAILY_FILE_CODE,
) -> SeedRefreshResult:
    selected_tickers = _unique_tickers(tickers)
    failures: list[str] = []
    profile_failures: list[str] = []
    quote_failures: list[str] = []
    analyst_failures: list[str] = []
    sec_fact_failures: list[str] = []
    fmp_financial_failures: list[str] = []
    financial_fact_periods = 0
    sec_financial_fact_periods = 0
    fmp_financial_fact_periods = 0
    profile_records = 0
    valuation_snapshots = 0
    quote_valuation_snapshots = 0
    analyst_events = 0
    finra_records = 0
    ai_tags_written = 0

    if not selected_tickers:
        return SeedRefreshResult(tickers_considered=0)

    if include_sec_facts:
        for ticker in selected_tickers:
            try:
                cik = _ensure_cik(conn, client, ticker)
                payload = sec.fetch_companyfacts(client, cik)
                rows = sec.extract_financial_facts(ticker=ticker, cik=cik, companyfacts=payload)
                written = sec.upsert_financial_facts(conn, rows)
                if not written:
                    raise ValueError("SEC companyfacts returned 0 usable financial fact periods.")
                financial_fact_periods += written
                sec_financial_fact_periods += written
                mark_source_success(
                    conn,
                    source_name="SEC companyfacts",
                    ticker=ticker,
                    endpoint="companyfacts",
                )
                mark_source_success(
                    conn,
                    source_name=financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,
                    ticker=ticker,
                    endpoint="financial-statements",
                )
            except Exception as exc:
                failure = f"{ticker} SEC companyfacts: {exc}"
                sec_fact_failures.append(failure)
                failures.append(failure)
                record_source_failure(
                    conn,
                    source_name="SEC companyfacts",
                    ticker=ticker,
                    endpoint="companyfacts",
                    reason=failure,
                    source_url=_sec_companyfacts_url(conn, ticker),
                )

    if fmp_api_key:
        for ticker in selected_tickers:
            if include_profile:
                try:
                    company_profile = profile.fetch_fmp_profile(client, ticker, fmp_api_key)
                    profile_records += profile.upsert_company_profiles(conn, [company_profile])
                    if company_profile.market_cap is not None:
                        valuation_snapshots += valuation.upsert_valuation_snapshots(
                            conn,
                            [
                                valuation.ValuationSnapshot(
                                    ticker=ticker,
                                    date=utc_now_iso()[:10],
                                    market_cap=company_profile.market_cap,
                                    source=company_profile.source,
                                )
                            ],
                        )
                    mark_source_success(
                        conn,
                        source_name=profile.FMP_PROFILE_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="profile",
                    )
                except Exception as exc:
                    failure = f"{ticker} profile: {_sanitize(str(exc), fmp_api_key)}"
                    profile_failures.append(failure)
                    failures.append(failure)
                    record_source_failure(
                        conn,
                        source_name=profile.FMP_PROFILE_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="profile",
                        reason=failure,
                        source_url=profile.FMP_PROFILE_URL.format(ticker=ticker, api_key="***"),
                    )
            if include_quote:
                try:
                    snapshot = valuation.fetch_fmp_quote_snapshot(client, ticker, fmp_api_key)
                    written = valuation.upsert_valuation_snapshots(conn, [snapshot])
                    valuation_snapshots += written
                    quote_valuation_snapshots += written
                    mark_source_success(
                        conn,
                        source_name=valuation.FMP_QUOTE_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="quote",
                    )
                except Exception as exc:
                    failure = f"{ticker} quote: {_sanitize(str(exc), fmp_api_key)}"
                    quote_failures.append(failure)
                    failures.append(failure)
                    record_source_failure(
                        conn,
                        source_name=valuation.FMP_QUOTE_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="quote",
                        reason=failure,
                        source_url=valuation.FMP_QUOTE_URL.format(ticker=ticker, api_key="***"),
                    )
            if include_analyst:
                try:
                    events = analysts.fetch_fmp_analyst_estimate_events(
                        client,
                        ticker,
                        fmp_api_key,
                        period=analyst_period,
                        limit=analyst_limit,
                    )
                    analyst_events += analysts.replace_fmp_analyst_estimate_events(conn, ticker, events)
                    mark_source_success(
                        conn,
                        source_name=analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="analyst-estimates",
                    )
                except Exception as exc:
                    failure = f"{ticker} analyst: {_sanitize(str(exc), fmp_api_key)}"
                    analyst_failures.append(failure)
                    failures.append(failure)
                    record_source_failure(
                        conn,
                        source_name=analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="analyst-estimates",
                        reason=failure,
                        source_url=_fmp_analyst_url(ticker, period=analyst_period, limit=analyst_limit),
                    )
            if include_fmp_financials:
                try:
                    rows = financials.fetch_fmp_financial_facts(
                        client,
                        ticker,
                        fmp_api_key,
                        period=financial_period,
                        limit=financial_limit,
                    )
                    written = sec.upsert_financial_facts(conn, rows)
                    financial_fact_periods += written
                    fmp_financial_fact_periods += written
                    mark_source_success(
                        conn,
                        source_name=financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="financial-statements",
                    )
                except Exception as exc:
                    failure = f"{ticker} FMP financial statements: {_sanitize(str(exc), fmp_api_key)}"
                    fmp_financial_failures.append(failure)
                    failures.append(failure)
                    record_source_failure(
                        conn,
                        source_name=financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,
                        ticker=ticker,
                        endpoint="financial-statements",
                        reason=failure,
                        source_url=_fmp_financials_url(ticker, period=financial_period, limit=financial_limit),
                    )
    else:
        _mark_fmp_sources_missing(
            conn,
            include_profile=include_profile,
            include_quote=include_quote,
            include_analyst=include_analyst,
            include_financials=include_fmp_financials,
        )
        if include_profile or include_quote or include_analyst or include_fmp_financials:
            failures.append("FMP_API_KEY is not configured; FMP-backed seed refresh steps were skipped.")
            _record_missing_fmp_key_failures(
                conn,
                tickers=selected_tickers,
                include_profile=include_profile,
                include_quote=include_quote,
                include_analyst=include_analyst,
                include_financials=include_fmp_financials,
                analyst_period=analyst_period,
                analyst_limit=analyst_limit,
                financial_period=financial_period,
                financial_limit=financial_limit,
            )

    _mark_count_source(
        conn,
        enabled=include_sec_facts,
        source_name="SEC companyfacts",
        count=sec_financial_fact_periods,
        item_label="financial fact period",
        failure_text="; ".join(sec_fact_failures[:5]),
    )
    if include_finra:
        try:
            finra_result = finra.fetch_latest_finra_daily_short_sale_volume(
                client,
                lookback_days=finra_lookback_days,
                tickers=selected_tickers,
                file_code=finra_file_code,
            )
            finra_records = finra.upsert_short_sale_volume(conn, finra_result.records)
            sec.mark_source_status(
                conn,
                source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
                status="ok" if finra_records else "degraded",
                reason=(
                    f"Seed refresh fetched FINRA daily short sale volume for {finra_result.trade_date}; "
                    f"ticker_filter={','.join(selected_tickers)}; imported {finra_records} row(s). "
                    "This is not short interest."
                ),
            )
        except Exception as exc:
            failures.append(f"FINRA: {exc}")
            sec.mark_source_status(
                conn,
                source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
                status="degraded",
                reason=str(exc)[:500],
            )

    for ticker in selected_tickers:
        tags = infer_tags_for_ticker(conn, ticker)
        ai_tags_written += replace_inferred_tags_for_ticker(conn, ticker, tags)

    _mark_fmp_source_statuses(
        conn,
        include_profile=include_profile,
        include_quote=include_quote,
        include_analyst=include_analyst,
        include_financials=include_fmp_financials,
        profile_records=profile_records,
        quote_valuation_snapshots=quote_valuation_snapshots,
        analyst_events=analyst_events,
        fmp_financial_fact_periods=fmp_financial_fact_periods,
        profile_failures=profile_failures,
        quote_failures=quote_failures,
        analyst_failures=analyst_failures,
        fmp_financial_failures=fmp_financial_failures,
        fmp_configured=bool(fmp_api_key),
    )
    sec.mark_source_status(
        conn,
        source_name="Automated AI industry tag inference",
        status="ok" if ai_tags_written else "degraded",
        reason=(
            f"Seed refresh considered {len(selected_tickers)} ticker(s); "
            f"wrote {ai_tags_written} inferred AI industry-chain tag(s)."
        ),
    )

    return SeedRefreshResult(
        tickers_considered=len(selected_tickers),
        financial_fact_periods=financial_fact_periods,
        profile_records=profile_records,
        valuation_snapshots=valuation_snapshots,
        analyst_events=analyst_events,
        finra_records=finra_records,
        ai_tags_written=ai_tags_written,
        failures=tuple(failures),
    )


def _mark_fmp_sources_missing(
    conn: sqlite3.Connection,
    *,
    include_profile: bool,
    include_quote: bool,
    include_analyst: bool,
    include_financials: bool,
) -> None:
    for include, source_name in (
        (include_profile, profile.FMP_PROFILE_SOURCE_NAME),
        (include_quote, valuation.FMP_QUOTE_SOURCE_NAME),
        (include_analyst, analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME),
        (include_financials, financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME),
    ):
        if include:
            sec.mark_source_status(conn, source_name=source_name, status="unavailable", reason="FMP_API_KEY is not configured.")


def _mark_fmp_source_statuses(
    conn: sqlite3.Connection,
    *,
    include_profile: bool,
    include_quote: bool,
    include_analyst: bool,
    include_financials: bool,
    profile_records: int,
    quote_valuation_snapshots: int,
    analyst_events: int,
    fmp_financial_fact_periods: int,
    profile_failures: list[str],
    quote_failures: list[str],
    analyst_failures: list[str],
    fmp_financial_failures: list[str],
    fmp_configured: bool,
) -> None:
    if not fmp_configured:
        return
    _mark_count_source(
        conn,
        enabled=include_profile,
        source_name=profile.FMP_PROFILE_SOURCE_NAME,
        count=profile_records,
        item_label="profile record",
        failure_text="; ".join(profile_failures[:5]),
    )
    _mark_count_source(
        conn,
        enabled=include_quote,
        source_name=valuation.FMP_QUOTE_SOURCE_NAME,
        count=quote_valuation_snapshots,
        item_label="valuation snapshot",
        failure_text="; ".join(quote_failures[:5]),
    )
    _mark_count_source(
        conn,
        enabled=include_analyst,
        source_name=analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
        count=analyst_events,
        item_label="analyst estimate event",
        failure_text="; ".join(analyst_failures[:5]),
    )
    _mark_count_source(
        conn,
        enabled=include_financials,
        source_name=financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,
        count=fmp_financial_fact_periods,
        item_label="financial fact period",
        failure_text="; ".join(fmp_financial_failures[:5]),
    )


def _mark_count_source(
    conn: sqlite3.Connection,
    *,
    enabled: bool,
    source_name: str,
    count: int,
    item_label: str,
    failure_text: str,
) -> None:
    if not enabled:
        return
    if count:
        status = "degraded" if failure_text else "ok"
        reason = f"Seed refresh stored {count} {item_label}(s)."
        if failure_text:
            reason += " Partial failures: " + failure_text[:350]
    else:
        status = "degraded"
        reason = "Seed refresh stored 0 rows."
        if failure_text:
            reason += " Failures: " + failure_text[:400]
    sec.mark_source_status(conn, source_name=source_name, status=status, reason=reason)


def _unique_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for ticker in tickers:
        value = ticker.strip().upper()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _sanitize(value: str, api_key: str) -> str:
    return value.replace(api_key, "***")


def _ensure_cik(conn: sqlite3.Connection, client: HttpClient, ticker: str) -> str:
    cik = sec.get_cik(conn, ticker)
    if cik:
        return cik
    sec.sync_company_tickers(conn, client)
    cik = sec.get_cik(conn, ticker)
    if not cik:
        raise ValueError("missing SEC CIK; run sync-sec-tickers or import company_profile first.")
    return cik


def _record_missing_fmp_key_failures(
    conn: sqlite3.Connection,
    *,
    tickers: list[str],
    include_profile: bool,
    include_quote: bool,
    include_analyst: bool,
    include_financials: bool,
    analyst_period: str,
    analyst_limit: int,
    financial_period: str,
    financial_limit: int,
) -> None:
    source_specs = (
        (
            include_profile,
            profile.FMP_PROFILE_SOURCE_NAME,
            "profile",
            lambda ticker: profile.FMP_PROFILE_URL.format(ticker=ticker, api_key="***"),
        ),
        (
            include_quote,
            valuation.FMP_QUOTE_SOURCE_NAME,
            "quote",
            lambda ticker: valuation.FMP_QUOTE_URL.format(ticker=ticker, api_key="***"),
        ),
        (
            include_analyst,
            analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
            "analyst-estimates",
            lambda ticker: _fmp_analyst_url(ticker, period=analyst_period, limit=analyst_limit),
        ),
        (
            include_financials,
            financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,
            "financial-statements",
            lambda ticker: _fmp_financials_url(ticker, period=financial_period, limit=financial_limit),
        ),
    )
    for enabled, source_name, endpoint, source_url_for_ticker in source_specs:
        if not enabled:
            continue
        for ticker in tickers:
            record_source_failure(
                conn,
                source_name=source_name,
                ticker=ticker,
                endpoint=endpoint,
                reason="FMP_API_KEY is not configured.",
                source_url=source_url_for_ticker(ticker),
                failure_type="missing_configuration",
            )


def _sec_companyfacts_url(conn: sqlite3.Connection, ticker: str) -> str | None:
    cik = sec.get_cik(conn, ticker)
    if not cik:
        return None
    return sec.COMPANYFACTS_URL.format(cik=sec.cik10(cik))


def _fmp_analyst_url(ticker: str, *, period: str, limit: int) -> str:
    request_limit = max(limit, analysts.FMP_ANALYST_ESTIMATES_MAX_LIMIT)
    return (
        f"{analysts.FMP_ANALYST_ESTIMATES_URL}"
        f"?symbol={ticker}&period={period}&limit={request_limit}&apikey=***"
    )


def _fmp_financials_url(ticker: str, *, period: str, limit: int) -> str:
    return ";".join(
        f"{financials.FMP_FINANCIAL_STATEMENTS_BASE_URL}/{statement}"
        f"?symbol={ticker}&period={period}&limit={limit}&apikey=***"
        for statement in ("income-statement", "balance-sheet-statement", "cash-flow-statement")
    )
