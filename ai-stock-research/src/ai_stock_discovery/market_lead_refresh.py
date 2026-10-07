from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import sqlite3

from ai_stock_discovery.data_mining_leads import build_data_mining_leads
from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.source_failures import record_source_failure
from ai_stock_discovery.sources import market, sec
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class MarketLeadRefreshResult:
    tickers_considered: int
    bars_written: int = 0
    anomaly_tickers_checked: int = 0
    anomaly_signals_written: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)


def select_market_bar_fallback_tickers(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 20,
    lead_limit: int = 300,
    min_score: float | None = None,
) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if lead_limit <= 0:
        raise ValueError("lead_limit must be positive.")
    explicit = _unique_tickers(tickers or [])
    if explicit:
        return explicit[:limit]
    leads = build_data_mining_leads(conn, limit=max(lead_limit, limit), min_score=min_score)
    selected: list[str] = []
    for lead in leads:
        if lead.mining_priority != "P0":
            continue
        if "Financial Modeling Prep historical price API:historical-price-eod" not in lead.source_blockers:
            continue
        selected.append(lead.ticker)
        if len(selected) >= limit:
            break
    return selected


def refresh_yahoo_market_bars_from_leads(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    tickers: list[str],
    benchmark_ticker: str = "QQQ",
    lookback_days: int = 90,
    detect_anomalies: bool = True,
    anomaly_lookback: int = 20,
) -> MarketLeadRefreshResult:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if anomaly_lookback <= 0:
        raise ValueError("anomaly_lookback must be positive.")
    selected_tickers = _unique_tickers(tickers)
    benchmark_ticker = benchmark_ticker.strip().upper()
    if not benchmark_ticker:
        raise ValueError("benchmark_ticker is required.")
    if not selected_tickers:
        return MarketLeadRefreshResult(tickers_considered=0)

    failures: list[str] = []
    bars_written = 0
    for ticker in selected_tickers:
        try:
            bars = market.fetch_yahoo_market_price_bars(
                client,
                ticker,
                benchmark_ticker=benchmark_ticker,
                lookback_days=lookback_days,
            )
        except Exception as exc:
            failure = f"{ticker}: {exc}"
            failures.append(failure)
            record_source_failure(
                conn,
                source_name=market.YAHOO_CHART_SOURCE_NAME,
                ticker=ticker,
                endpoint="chart-1d",
                reason=failure,
                source_url=_yahoo_source_url(ticker, benchmark_ticker=benchmark_ticker, lookback_days=lookback_days),
            )
            continue
        bars_written += market.upsert_market_price_bars(conn, bars)

    _mark_yahoo_status(
        conn,
        ticker_count=len(selected_tickers),
        bars_written=bars_written,
        benchmark_ticker=benchmark_ticker,
        lookback_days=lookback_days,
        failures=failures,
    )

    anomaly_tickers_checked = 0
    anomaly_signals_written = 0
    if detect_anomalies and bars_written:
        anomaly_result = market.detect_market_anomalies(
            conn,
            tickers=selected_tickers,
            limit=len(selected_tickers),
            lookback=anomaly_lookback,
        )
        anomaly_tickers_checked = anomaly_result.tickers_checked
        anomaly_signals_written = anomaly_result.signals_written
        sec.mark_source_status(
            conn,
            source_name="Local market anomaly detector",
            status="ok" if anomaly_tickers_checked else "unavailable",
            reason=(
                f"Checked {anomaly_tickers_checked} ticker(s) after market-lead refresh; "
                f"wrote {anomaly_signals_written} signal(s)."
            )
            if anomaly_tickers_checked
            else "No refreshed market bars were usable for anomaly detection.",
        )

    return MarketLeadRefreshResult(
        tickers_considered=len(selected_tickers),
        bars_written=bars_written,
        anomaly_tickers_checked=anomaly_tickers_checked,
        anomaly_signals_written=anomaly_signals_written,
        failures=tuple(failures),
    )


def _mark_yahoo_status(
    conn: sqlite3.Connection,
    *,
    ticker_count: int,
    bars_written: int,
    benchmark_ticker: str,
    lookback_days: int,
    failures: list[str],
) -> None:
    if failures:
        status = "degraded" if bars_written else "unavailable"
        reason = (
            "Prototype-only source used as FMP EOD fallback from data-mining leads; "
            f"stored {bars_written} EOD price/volume bar(s) for {ticker_count} ticker(s); "
            f"failures={len(failures)}. Yahoo chart is not a production-critical data source."
        )
    else:
        status = "ok"
        reason = (
            "Prototype-only source used as FMP EOD fallback from data-mining leads; "
            f"stored {bars_written} EOD price/volume bar(s) for {ticker_count} ticker(s); "
            f"benchmark={benchmark_ticker}; lookback_days={lookback_days}. "
            "Yahoo chart is not a production-critical data source."
        )
    sec.mark_source_status(
        conn,
        source_name=market.YAHOO_CHART_SOURCE_NAME,
        status=status,
        reason=reason,
    )


def _yahoo_source_url(ticker: str, *, benchmark_ticker: str, lookback_days: int) -> str:
    to_date = date.fromisoformat(utc_now_iso()[:10])
    from_date = to_date - timedelta(days=lookback_days)
    return (
        f"{market.yahoo_chart_url(ticker, from_date, to_date)}; "
        f"benchmark={benchmark_ticker}:{market.yahoo_chart_url(benchmark_ticker, from_date, to_date)}"
    )


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
