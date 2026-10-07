from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

from ai_stock_discovery.analysis.catalysts import extract_catalysts_from_evidence, store_catalysts
from ai_stock_discovery.analysis.risk_extraction import (
    extract_risk_flags_from_evidence,
    store_extracted_risk_flags,
)
from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.refresh_plan import build_refresh_plan
from ai_stock_discovery.sources import finra, gdelt, news_rss, sec


FREE_SOURCE_ACTIONS = {"refresh_short_sale_context", "search_news_for_catalysts"}


@dataclass(frozen=True)
class FreeSourceRefreshResult:
    tickers_considered: int
    finra_records: int = 0
    news_events_seen: int = 0
    news_events_mapped: int = 0
    gdelt_events_seen: int = 0
    gdelt_events_mapped: int = 0
    catalysts_inserted: int = 0
    risk_flags_inserted: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)


def select_free_source_refresh_tickers(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 20,
    plan_limit: int = 300,
) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    explicit = _unique_tickers(tickers or [])
    if explicit:
        return explicit[:limit]
    rows = build_refresh_plan(conn, limit=max(plan_limit, limit))
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.action_type not in FREE_SOURCE_ACTIONS:
            continue
        ticker = row.ticker.upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        selected.append(ticker)
        if len(selected) >= limit:
            break
    return selected


def refresh_free_source_context(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    tickers: list[str],
    include_finra: bool = True,
    include_news: bool = True,
    include_gdelt: bool = False,
    extract_catalysts: bool = True,
    extract_risks: bool = True,
    finra_lookback_days: int = 10,
    finra_file_code: str = finra.DEFAULT_DAILY_FILE_CODE,
    news_limit: int = 50,
    gdelt_limit: int = 10,
    gdelt_timespan: str = "30d",
    extraction_limit: int = 200,
) -> FreeSourceRefreshResult:
    selected_tickers = _unique_tickers(tickers)
    failures: list[str] = []
    finra_records = 0
    news_events_seen = 0
    news_events_mapped = 0
    gdelt_events_seen = 0
    gdelt_events_mapped = 0
    catalysts_inserted = 0
    risk_flags_inserted = 0

    if not selected_tickers:
        return FreeSourceRefreshResult(tickers_considered=0)

    if include_finra:
        try:
            result = finra.fetch_latest_finra_daily_short_sale_volume(
                client,
                lookback_days=finra_lookback_days,
                tickers=selected_tickers,
                file_code=finra_file_code,
            )
            finra_records = finra.upsert_short_sale_volume(conn, result.records)
            sec.mark_source_status(
                conn,
                source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
                status="ok" if finra_records else "degraded",
                reason=(
                    f"Free source refresh fetched FINRA daily short sale volume for {result.trade_date}; "
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

    if include_news:
        for source in news_rss.DEFAULT_NEWS_RSS_SOURCES:
            try:
                events = news_rss.fetch_news_rss(
                    client,
                    url=source.url,
                    source_name=source.source_name,
                    limit=news_limit,
                )
                stored = news_rss.store_news_events(conn, events)
                news_events_seen += stored.events_seen
                news_events_mapped += stored.mapped
                sec.mark_source_status(
                    conn,
                    source_name=source.source_name,
                    status="ok",
                    reason="Free source refresh stored RSS title/summary/url evidence only.",
                )
            except Exception as exc:
                failures.append(f"{source.source_name}: {exc}")
                sec.mark_source_status(
                    conn,
                    source_name=source.source_name,
                    status="degraded",
                    reason=str(exc)[:500],
                )

    if include_gdelt:
        for ticker in selected_tickers:
            query = _gdelt_query(conn, ticker)
            try:
                result = gdelt.fetch_gdelt_doc_news(
                    client,
                    query=query,
                    limit=gdelt_limit,
                    timespan=gdelt_timespan,
                )
                stored = news_rss.store_news_events(
                    conn,
                    result.events,
                    source_type="News Search",
                    content_hash_prefix="gdelt-doc-news",
                    mapped_confidence=0.6,
                    unmapped_confidence=0.35,
                )
                gdelt_events_seen += stored.events_seen
                gdelt_events_mapped += stored.mapped
            except Exception as exc:
                failures.append(f"{ticker} GDELT: {exc}")
        sec.mark_source_status(
            conn,
            source_name=gdelt.GDELT_DOC_API_SOURCE_NAME,
            status="ok" if gdelt_events_seen and not any("GDELT" in failure for failure in failures) else "degraded",
            reason=(
                "Free source refresh stored GDELT article metadata only; "
                f"tickers={len(selected_tickers)}; events={gdelt_events_seen}; mapped={gdelt_events_mapped}; "
                f"failures={sum(1 for failure in failures if 'GDELT' in failure)}."
            ),
        )

    if extract_catalysts:
        for ticker in selected_tickers:
            candidates = extract_catalysts_from_evidence(conn, ticker=ticker, limit=extraction_limit)
            catalysts_inserted += store_catalysts(conn, candidates)
        sec.mark_source_status(
            conn,
            source_name="Automated catalyst extraction",
            status="ok",
            reason=(
                f"Free source refresh extracted catalyst candidates for {len(selected_tickers)} ticker(s); "
                f"inserted {catalysts_inserted} new row(s)."
            ),
        )

    if extract_risks:
        for ticker in selected_tickers:
            flags = extract_risk_flags_from_evidence(conn, ticker=ticker, limit=extraction_limit)
            risk_flags_inserted += store_extracted_risk_flags(conn, flags)
        sec.mark_source_status(
            conn,
            source_name="Automated risk text extraction",
            status="ok",
            reason=(
                f"Free source refresh extracted risk candidates for {len(selected_tickers)} ticker(s); "
                f"inserted {risk_flags_inserted} new row(s)."
            ),
        )

    return FreeSourceRefreshResult(
        tickers_considered=len(selected_tickers),
        finra_records=finra_records,
        news_events_seen=news_events_seen,
        news_events_mapped=news_events_mapped,
        gdelt_events_seen=gdelt_events_seen,
        gdelt_events_mapped=gdelt_events_mapped,
        catalysts_inserted=catalysts_inserted,
        risk_flags_inserted=risk_flags_inserted,
        failures=tuple(failures),
    )


def _gdelt_query(conn: sqlite3.Connection, ticker: str) -> str:
    company_name = _company_name(conn, ticker)
    subject = company_name or ticker.upper()
    return (
        f'"{subject}" '
        '(AI OR "data center" OR earnings OR guidance OR backlog OR order OR customer OR partnership)'
    )


def _company_name(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        """
        SELECT company_name
        FROM company_profile
        WHERE ticker = ? AND COALESCE(company_name, '') != ''
        UNION
        SELECT company_name
        FROM universe
        WHERE ticker = ? AND COALESCE(company_name, '') != ''
        LIMIT 1
        """,
        (ticker.upper(), ticker.upper()),
    ).fetchone()
    return str(row["company_name"]).strip() if row and row["company_name"] else None


def _unique_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        selected.append(ticker)
    return selected
