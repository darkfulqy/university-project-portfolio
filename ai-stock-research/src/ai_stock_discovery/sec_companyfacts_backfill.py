from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.source_failures import mark_source_success, record_source_failure
from ai_stock_discovery.sources import financials, sec


@dataclass(frozen=True)
class SecCompanyfactsBackfillResult:
    tickers_considered: int
    financial_fact_periods: int = 0
    successes: tuple[str, ...] = field(default_factory=tuple)
    failures: tuple[str, ...] = field(default_factory=tuple)


def select_sec_companyfacts_backfill_tickers(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    from_source_failures: bool = True,
    limit: int = 25,
) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    explicit = _unique_tickers(tickers or [])
    if explicit:
        return explicit[:limit]
    if not from_source_failures:
        return []
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM source_failures
        WHERE status = 'open'
          AND source_name = ?
          AND endpoint = 'financial-statements'
          AND ticker IS NOT NULL
          AND ticker != ''
        ORDER BY last_seen_at DESC, ticker
        LIMIT ?
        """,
        (financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME, limit),
    ).fetchall()
    return [str(row["ticker"]).upper() for row in rows]


def backfill_sec_companyfacts(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    tickers: list[str],
) -> SecCompanyfactsBackfillResult:
    selected_tickers = _unique_tickers(tickers)
    financial_fact_periods = 0
    successes: list[str] = []
    failures: list[str] = []
    for ticker in selected_tickers:
        try:
            cik = _ensure_cik(conn, client, ticker)
            payload = sec.fetch_companyfacts(client, cik)
            rows = sec.extract_financial_facts(ticker=ticker, cik=cik, companyfacts=payload)
            written = sec.upsert_financial_facts(conn, rows)
            if not written:
                raise ValueError("SEC companyfacts returned 0 usable financial fact periods.")
            financial_fact_periods += written
            successes.append(ticker)
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
            reason = f"{ticker} SEC companyfacts: {exc}"
            failures.append(reason)
            record_source_failure(
                conn,
                source_name="SEC companyfacts",
                ticker=ticker,
                endpoint="companyfacts",
                reason=reason,
                source_url=_sec_companyfacts_url(conn, ticker),
            )
    _mark_sec_companyfacts_status(
        conn,
        successes=successes,
        failures=failures,
        financial_fact_periods=financial_fact_periods,
    )
    return SecCompanyfactsBackfillResult(
        tickers_considered=len(selected_tickers),
        financial_fact_periods=financial_fact_periods,
        successes=tuple(successes),
        failures=tuple(failures),
    )


def record_sec_companyfacts_backfill_unavailable(
    conn: sqlite3.Connection,
    *,
    tickers: list[str],
    reason: str = "SEC_USER_AGENT is not configured; SEC companyfacts backfill skipped.",
) -> int:
    selected_tickers = _unique_tickers(tickers)
    sec.mark_source_status(
        conn,
        source_name="SEC companyfacts",
        status="unavailable",
        reason=reason,
    )
    for ticker in selected_tickers:
        record_source_failure(
            conn,
            source_name="SEC companyfacts",
            ticker=ticker,
            endpoint="companyfacts",
            reason=f"{ticker} SEC companyfacts: {reason}",
            source_url=_sec_companyfacts_url(conn, ticker),
            failure_type="missing_configuration",
        )
    return len(selected_tickers)


def _ensure_cik(conn: sqlite3.Connection, client: HttpClient, ticker: str) -> str:
    cik = sec.get_cik(conn, ticker)
    if cik:
        return cik
    sec.sync_company_tickers(conn, client)
    cik = sec.get_cik(conn, ticker)
    if not cik:
        raise ValueError("missing SEC CIK; run sync-sec-tickers or import company_profile first.")
    return cik


def _mark_sec_companyfacts_status(
    conn: sqlite3.Connection,
    *,
    successes: list[str],
    failures: list[str],
    financial_fact_periods: int,
) -> None:
    if successes and not failures:
        status = "ok"
    elif successes or failures:
        status = "degraded"
    else:
        status = "degraded"
    reason = (
        f"SEC companyfacts backfill stored {financial_fact_periods} financial fact period(s) "
        f"for {len(successes)} ticker(s)."
    )
    if failures:
        reason += " Failures: " + "; ".join(failures[:5])[:500]
    sec.mark_source_status(conn, source_name="SEC companyfacts", status=status, reason=reason)


def _sec_companyfacts_url(conn: sqlite3.Connection, ticker: str) -> str | None:
    cik = sec.get_cik(conn, ticker)
    if not cik:
        return None
    return sec.COMPANYFACTS_URL.format(cik=sec.cik10(cik))


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
