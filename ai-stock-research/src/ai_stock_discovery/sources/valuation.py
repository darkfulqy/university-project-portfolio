from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d"
FMP_QUOTE_SOURCE_NAME = "Financial Modeling Prep quote API"
FMP_QUOTE_URL = "https://financialmodelingprep.com/stable/quote?symbol={ticker}&apikey={api_key}"
EASTMONEY_US_QUOTE_SOURCE_NAME = "Eastmoney US quote API"
EASTMONEY_US_QUOTE_FIELDS = ",".join(
    [
        "f43",  # latest regular-session price
        "f44",  # high
        "f45",  # low
        "f46",  # open
        "f47",  # volume
        "f57",  # ticker
        "f58",  # company/security name
        "f59",  # price decimals
        "f60",  # previous close
        "f86",  # source timestamp
        "f170",  # change percent
        "f152",  # percent decimals
        "f600",  # trading status label
    ]
)
EASTMONEY_US_QUOTE_URL = (
    "https://push2.eastmoney.com/api/qt/stock/get"
    "?secid=105.{ticker}&fields={fields}&invt=2&fltt=1"
)


@dataclass(frozen=True)
class ValuationSnapshot:
    ticker: str
    date: str
    price: float | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None
    ev_sales: float | None = None
    ev_ebitda: float | None = None
    pe: float | None = None
    fcf_yield: float | None = None
    sector_percentile: float | None = None
    source: str = "manual"


def upsert_valuation_snapshots(
    conn: sqlite3.Connection,
    snapshots: list[ValuationSnapshot],
) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO valuation_snapshots (
            ticker, date, price, market_cap, enterprise_value, ev_sales,
            ev_ebitda, pe, fcf_yield, sector_percentile, source, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date, source) DO UPDATE SET
            price=excluded.price,
            market_cap=excluded.market_cap,
            enterprise_value=excluded.enterprise_value,
            ev_sales=excluded.ev_sales,
            ev_ebitda=excluded.ev_ebitda,
            pe=excluded.pe,
            fcf_yield=excluded.fcf_yield,
            sector_percentile=excluded.sector_percentile,
            updated_at=excluded.updated_at
        """,
        [
            (
                snapshot.ticker.upper(),
                snapshot.date,
                snapshot.price,
                snapshot.market_cap,
                snapshot.enterprise_value,
                snapshot.ev_sales,
                snapshot.ev_ebitda,
                snapshot.pe,
                snapshot.fcf_yield,
                snapshot.sector_percentile,
                snapshot.source,
                now,
            )
            for snapshot in snapshots
        ],
    )
    for snapshot in snapshots:
        _store_snapshot_evidence(conn, snapshot=snapshot, fetched_at=now)
    return len(snapshots)


def import_valuation_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    snapshots: list[ValuationSnapshot] = []
    default_source = f"manual_csv:{csv_path.name}"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = _field(row, "ticker", "Ticker")
            date = _field(row, "date", "Date")
            if not ticker or not date:
                continue
            snapshots.append(
                ValuationSnapshot(
                    ticker=ticker,
                    date=date,
                    price=_number(_field(row, "price")),
                    market_cap=_number(_field(row, "market_cap", "marketCap")),
                    enterprise_value=_number(_field(row, "enterprise_value", "enterpriseValue")),
                    ev_sales=_number(_field(row, "ev_sales", "evSales")),
                    ev_ebitda=_number(_field(row, "ev_ebitda", "evEbitda")),
                    pe=_number(_field(row, "pe", "trailingPE")),
                    fcf_yield=_number(_field(row, "fcf_yield", "fcfYield")),
                    sector_percentile=_number(_field(row, "sector_percentile", "sectorPercentile")),
                    source=_field(row, "source") or default_source,
                )
            )
    return upsert_valuation_snapshots(conn, snapshots)


def fetch_yahoo_snapshot(client: HttpClient, ticker: str) -> ValuationSnapshot:
    ticker = ticker.upper()
    url = YAHOO_QUOTE_URL.format(ticker=ticker)
    try:
        payload = client.get_json(url)
        result = _first_yahoo_quote(payload)
    except FetchError:
        result = None
    if result:
        return ValuationSnapshot(
            ticker=ticker,
            date=utc_now_iso()[:10],
            price=_number(result.get("regularMarketPrice")),
            market_cap=_number(result.get("marketCap")),
            pe=_number(result.get("trailingPE")),
            source=url,
        )
    fallback_url = YAHOO_CHART_URL.format(ticker=ticker)
    fallback_payload = client.get_json(fallback_url)
    meta = _first_yahoo_chart_meta(fallback_payload)
    if not meta:
        raise ValueError(f"No Yahoo quote or chart result returned for {ticker}")
    return ValuationSnapshot(
        ticker=ticker,
        date=utc_now_iso()[:10],
        price=_number(meta.get("regularMarketPrice") or meta.get("chartPreviousClose")),
        source=fallback_url,
    )


def fetch_fmp_quote_snapshot(client: HttpClient, ticker: str, api_key: str) -> ValuationSnapshot:
    ticker = ticker.upper()
    url = FMP_QUOTE_URL.format(ticker=ticker, api_key=api_key)
    try:
        payload = client.get_json(url)
    except FetchError as exc:
        raise FetchError(str(exc).replace(api_key, "***")) from exc
    result = _first_fmp_quote(payload)
    if not result:
        raise ValueError(f"No FMP quote result returned for {ticker}")
    source_url = FMP_QUOTE_URL.format(ticker=ticker, api_key="***")
    return ValuationSnapshot(
        ticker=ticker,
        date=utc_now_iso()[:10],
        price=_number(result.get("price")),
        market_cap=_number(result.get("marketCap")),
        enterprise_value=_number(result.get("enterpriseValue")),
        pe=_number(result.get("pe")),
        source=source_url,
    )


def fetch_eastmoney_quote_snapshot(client: HttpClient, ticker: str) -> ValuationSnapshot:
    ticker = ticker.upper()
    url = eastmoney_quote_url(ticker)
    payload = client.get_json(url)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Eastmoney quote payload for {ticker}.")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"No Eastmoney quote result returned for {ticker}.")
    price_decimals = data.get("f59")
    price = _scaled_number(data.get("f43"), price_decimals)
    if price is None:
        raise ValueError(f"No usable Eastmoney latest price returned for {ticker}.")
    source_timestamp = _eastmoney_timestamp(data.get("f86"))
    return ValuationSnapshot(
        ticker=ticker,
        date=(source_timestamp or utc_now_iso())[:10],
        price=price,
        source=url,
    )


def _store_snapshot_evidence(
    conn: sqlite3.Connection,
    *,
    snapshot: ValuationSnapshot,
    fetched_at: str,
) -> None:
    summary = (
        "Valuation snapshot. Values are stored only when supplied by the source; "
        "missing fields must not be inferred."
    )
    snippet = ", ".join(
        [
            f"ticker={snapshot.ticker.upper()}",
            f"date={snapshot.date}",
            f"price={_fmt(snapshot.price)}",
            f"market_cap={_fmt(snapshot.market_cap)}",
            f"enterprise_value={_fmt(snapshot.enterprise_value)}",
            f"ev_sales={_fmt(snapshot.ev_sales)}",
            f"ev_ebitda={_fmt(snapshot.ev_ebitda)}",
            f"pe={_fmt(snapshot.pe)}",
            f"fcf_yield={_fmt(snapshot.fcf_yield)}",
        ]
    )
    content_hash = hashlib.sha256(
        json.dumps(
            {
                "ticker": snapshot.ticker.upper(),
                "date": snapshot.date,
                "source": snapshot.source,
                "price": snapshot.price,
                "market_cap": snapshot.market_cap,
                "enterprise_value": snapshot.enterprise_value,
                "ev_sales": snapshot.ev_sales,
                "ev_ebitda": snapshot.ev_ebitda,
                "pe": snapshot.pe,
                "fcf_yield": snapshot.fcf_yield,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    conn.execute(
        """
        INSERT OR IGNORE INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at, raw_title, summary,
            evidence_snippet, confidence, related_ticker, related_module, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Valuation Snapshot",
            _source_name(snapshot.source),
            snapshot.source,
            snapshot.date,
            fetched_at,
            f"{snapshot.ticker.upper()} valuation snapshot",
            summary,
            snippet,
            _confidence(snapshot.source),
            snapshot.ticker.upper(),
            "valuation",
            f"valuation:{content_hash}",
        ),
    )


def _first_yahoo_quote(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    quote_response = payload.get("quoteResponse")
    if not isinstance(quote_response, dict):
        return None
    results = quote_response.get("result")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    return first if isinstance(first, dict) else None


def _first_yahoo_chart_meta(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return None
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    meta = first.get("meta")
    return meta if isinstance(meta, dict) else None


def _first_fmp_quote(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    return first if isinstance(first, dict) else None


def eastmoney_quote_url(ticker: str) -> str:
    return EASTMONEY_US_QUOTE_URL.format(ticker=ticker.upper(), fields=EASTMONEY_US_QUOTE_FIELDS)


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


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


def _scaled_number(value: object, decimals: object) -> float | None:
    number = _number(value)
    if number is None:
        return None
    try:
        places = int(decimals)
    except (TypeError, ValueError):
        places = 2
    return number / (10**places)


def _eastmoney_timestamp(value: object) -> str | None:
    if value is None:
        return None
    try:
        from datetime import datetime, timezone

        raw = int(value)
    except (TypeError, ValueError):
        return None
    if raw <= 0:
        return None
    return datetime.fromtimestamp(raw, timezone.utc).replace(microsecond=0).isoformat()


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:g}"


def _source_name(source: str) -> str:
    if "push2.eastmoney.com/api/qt/stock/get" in source:
        return EASTMONEY_US_QUOTE_SOURCE_NAME
    if "finance.yahoo.com" in source or "query1.finance.yahoo.com" in source:
        return "Yahoo Finance endpoint prototype"
    if "financialmodelingprep.com" in source:
        return "Financial Modeling Prep quote API"
    if source.startswith("manual_csv:"):
        return source
    return "Valuation source"


def _confidence(source: str) -> float:
    if source.startswith("manual_csv:"):
        return 0.6
    if "query1.finance.yahoo.com" in source:
        return 0.55
    if "push2.eastmoney.com/api/qt/stock/get" in source:
        return 0.58
    if "financialmodelingprep.com" in source:
        return 0.75
    return 0.7
