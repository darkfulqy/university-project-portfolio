from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3
from typing import Any

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


FMP_HISTORICAL_PRICE_SOURCE_NAME = "Financial Modeling Prep historical price API"
FMP_HISTORICAL_PRICE_SOURCE_TYPE = "fmp_historical_eod_price_api"
FMP_HISTORICAL_PRICE_EOD_URL = (
    "https://financialmodelingprep.com/stable/historical-price-eod/full"
    "?symbol={ticker}&from={from_date}&to={to_date}&apikey={api_key}"
)
YAHOO_CHART_SOURCE_NAME = "Yahoo Finance chart endpoint prototype"
YAHOO_CHART_SOURCE_TYPE = "yahoo_chart_endpoint_prototype"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval=1d"

KNOWN_LOCAL_MARKET_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("Local options_flow.sqlite", "options_flow.sqlite", "options_flow_sqlite"),
    ("Local opening_confirmation_system.py", "opening_confirmation_system.py", "premarket_script"),
    ("Local realtime_options_monitor.py", "realtime_options_monitor.py", "options_flow_script"),
)


@dataclass(frozen=True)
class LocalMarketSource:
    source_name: str
    path: Path
    source_type: str
    exists: bool


@dataclass(frozen=True)
class MarketConfirmationSignal:
    ticker: str
    signal_date: str | None
    source_type: str
    source_name: str
    source_path: str | None
    signal_type: str
    direction: str
    magnitude: float | None
    description: str
    confidence: float
    content_hash: str


@dataclass(frozen=True)
class MarketPriceBar:
    ticker: str
    bar_date: str
    open: float | None
    high: float | None
    low: float | None
    close: float
    volume: float
    benchmark_close: float | None
    premarket_price: float | None
    source_type: str
    source_name: str
    source_path: str
    content_hash: str


@dataclass(frozen=True)
class MarketAnomalyResult:
    tickers_checked: int
    signals_written: int


def detect_local_market_sources(base_path: Path) -> list[LocalMarketSource]:
    return [
        LocalMarketSource(
            source_name=source_name,
            path=base_path / relative_path,
            source_type=source_type,
            exists=(base_path / relative_path).exists(),
        )
        for source_name, relative_path, source_type in KNOWN_LOCAL_MARKET_SOURCES
    ]


def fetch_fmp_market_price_bars(
    client: HttpClient,
    ticker: str,
    api_key: str,
    *,
    benchmark_ticker: str = "QQQ",
    lookback_days: int = 90,
    end_date: date | None = None,
) -> list[MarketPriceBar]:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    ticker = ticker.strip().upper()
    benchmark_ticker = benchmark_ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required.")
    if not benchmark_ticker:
        raise ValueError("benchmark_ticker is required.")

    to_date = end_date or date.fromisoformat(utc_now_iso()[:10])
    from_date = to_date - timedelta(days=lookback_days)
    source_rows = _fetch_fmp_historical_eod_rows(client, ticker, api_key, from_date, to_date)
    benchmark_rows = _fetch_fmp_historical_eod_rows(client, benchmark_ticker, api_key, from_date, to_date)
    benchmark_close_by_date = {
        bar_date: close
        for bar_date, close in (
            _fmp_historical_close(benchmark_ticker, row) for row in benchmark_rows
        )
    }
    source_path = (
        f"{_fmp_historical_price_url(ticker, from_date, to_date, '***')}; "
        f"benchmark={benchmark_ticker}:{_fmp_historical_price_url(benchmark_ticker, from_date, to_date, '***')}"
    )
    bars = [
        _fmp_market_price_bar_from_row(
            ticker=ticker,
            row=row,
            benchmark_close_by_date=benchmark_close_by_date,
            source_path=source_path,
        )
        for row in source_rows
    ]
    if not bars:
        raise ValueError(f"No usable FMP historical EOD price rows returned for {ticker}.")
    return sorted(bars, key=lambda bar: bar.bar_date)


def fetch_yahoo_market_price_bars(
    client: HttpClient,
    ticker: str,
    *,
    benchmark_ticker: str = "QQQ",
    lookback_days: int = 90,
    end_date: date | None = None,
) -> list[MarketPriceBar]:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    ticker = ticker.strip().upper()
    benchmark_ticker = benchmark_ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required.")
    if not benchmark_ticker:
        raise ValueError("benchmark_ticker is required.")

    to_date = end_date or date.fromisoformat(utc_now_iso()[:10])
    from_date = to_date - timedelta(days=lookback_days)
    source_rows = _fetch_yahoo_chart_rows(client, ticker, from_date, to_date)
    benchmark_rows = _fetch_yahoo_chart_rows(client, benchmark_ticker, from_date, to_date)
    benchmark_close_by_date = {
        row_date: close
        for row_date, close in (
            _yahoo_chart_close(benchmark_ticker, row) for row in benchmark_rows
        )
    }
    source_path = (
        f"{yahoo_chart_url(ticker, from_date, to_date)}; "
        f"benchmark={benchmark_ticker}:{yahoo_chart_url(benchmark_ticker, from_date, to_date)}"
    )
    bars = [
        _yahoo_market_price_bar_from_row(
            ticker=ticker,
            row=row,
            benchmark_close_by_date=benchmark_close_by_date,
            source_path=source_path,
        )
        for row in source_rows
    ]
    if not bars:
        raise ValueError(f"No usable Yahoo chart price rows returned for {ticker}.")
    return sorted(bars, key=lambda bar: bar.bar_date)


def import_market_price_bars_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "date", "close", "volume"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Market price bars CSV missing required columns: {', '.join(sorted(missing))}")
        bars = [_price_bar_from_row(row, csv_path) for row in reader]
    return upsert_market_price_bars(conn, bars)


def upsert_market_price_bars(conn: sqlite3.Connection, bars: list[MarketPriceBar]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO market_price_bars (
            ticker, bar_date, open, high, low, close, volume,
            benchmark_close, premarket_price, source_type, source_name,
            source_path, content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            benchmark_close=excluded.benchmark_close,
            premarket_price=excluded.premarket_price,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_path=excluded.source_path,
            updated_at=excluded.updated_at
        """,
        [
            (
                bar.ticker,
                bar.bar_date,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.benchmark_close,
                bar.premarket_price,
                bar.source_type,
                bar.source_name,
                bar.source_path,
                bar.content_hash,
                now,
            )
            for bar in bars
        ],
    )
    return len(bars)


def detect_market_anomalies(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    lookback: int = 20,
    min_volume_multiple: float = 2.0,
    min_gap_pct: float = 0.03,
    min_relative_strength_pct: float = 0.03,
) -> MarketAnomalyResult:
    selected_tickers = [ticker.upper() for ticker in tickers] if tickers else _market_bar_tickers(conn, limit)
    checked = 0
    signals: list[MarketConfirmationSignal] = []
    for ticker in selected_tickers[:limit]:
        rows = _price_rows_for_ticker(conn, ticker, lookback=max(lookback, 2))
        if len(rows) < 2:
            continue
        checked += 1
        signals.extend(
            _signals_from_price_rows(
                ticker,
                rows,
                lookback=max(lookback, 2),
                min_volume_multiple=min_volume_multiple,
                min_gap_pct=min_gap_pct,
                min_relative_strength_pct=min_relative_strength_pct,
            )
        )
    written = upsert_market_confirmation_signals(conn, signals) if signals else 0
    return MarketAnomalyResult(tickers_checked=checked, signals_written=written)


def import_market_confirmation_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "signal_type", "direction", "description", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Market confirmation CSV missing required columns: {', '.join(sorted(missing))}")
        signals = [_signal_from_row(row, csv_path) for row in reader]
    return upsert_market_confirmation_signals(conn, signals)


def upsert_market_confirmation_signals(
    conn: sqlite3.Connection,
    signals: list[MarketConfirmationSignal],
) -> int:
    now = utc_now_iso()
    rows = [
        (
            signal.ticker,
            signal.signal_date,
            signal.source_type,
            signal.source_name,
            signal.source_path,
            signal.signal_type,
            signal.direction,
            signal.magnitude,
            signal.description,
            signal.confidence,
            signal.content_hash,
            now,
        )
        for signal in signals
    ]
    conn.executemany(
        """
        INSERT INTO market_confirmation_signals (
            ticker, signal_date, source_type, source_name, source_path,
            signal_type, direction, magnitude, description, confidence,
            content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            signal_date=excluded.signal_date,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_path=excluded.source_path,
            signal_type=excluded.signal_type,
            direction=excluded.direction,
            magnitude=excluded.magnitude,
            description=excluded.description,
            confidence=excluded.confidence,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.executemany(
        """
        INSERT INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at, raw_title, summary,
            evidence_snippet, confidence, related_ticker, related_module, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            fetched_at=excluded.fetched_at,
            raw_title=excluded.raw_title,
            summary=excluded.summary,
            evidence_snippet=excluded.evidence_snippet,
            confidence=excluded.confidence,
            related_ticker=excluded.related_ticker
        """,
        [
            (
                signal.source_type,
                signal.source_name,
                signal.source_path or signal.source_name,
                signal.signal_date,
                now,
                signal.signal_type,
                signal.description,
                signal.description,
                signal.confidence,
                signal.ticker,
                "market_confirmation",
                f"market-confirmation:{signal.content_hash}",
            )
            for signal in signals
        ],
    )
    return len(signals)


def _price_bar_from_row(row: dict[str, str], csv_path: Path) -> MarketPriceBar:
    ticker = row.get("ticker", "").strip().upper()
    bar_date = row.get("date", "").strip() or row.get("bar_date", "").strip()
    if not ticker:
        raise ValueError("Market price bars CSV contains a row without ticker.")
    if not bar_date:
        raise ValueError(f"Market price bars CSV row for {ticker} is missing date.")
    close = _required_float(row.get("close"), f"Market price bars CSV row for {ticker} is missing close.")
    volume = _required_float(row.get("volume"), f"Market price bars CSV row for {ticker} is missing volume.")
    if close <= 0:
        raise ValueError(f"Market price bars CSV row for {ticker} has non-positive close.")
    if volume < 0:
        raise ValueError(f"Market price bars CSV row for {ticker} has negative volume.")
    source_type = row.get("source_type", "").strip() or "local_market_price_bars_csv"
    source_name = row.get("source_name", "").strip() or csv_path.name
    source_path = row.get("source_path", "").strip() or str(csv_path)
    content_hash = hashlib.sha256(
        "|".join([ticker, bar_date, source_type, source_name, source_path]).encode("utf-8")
    ).hexdigest()
    return MarketPriceBar(
        ticker=ticker,
        bar_date=bar_date,
        open=_optional_float(row.get("open")),
        high=_optional_float(row.get("high")),
        low=_optional_float(row.get("low")),
        close=close,
        volume=volume,
        benchmark_close=_optional_float(row.get("benchmark_close")),
        premarket_price=_optional_float(row.get("premarket_price")),
        source_type=source_type,
        source_name=source_name,
        source_path=source_path,
        content_hash=content_hash,
    )


def _fetch_fmp_historical_eod_rows(
    client: HttpClient,
    ticker: str,
    api_key: str,
    from_date: date,
    to_date: date,
) -> list[dict[str, Any]]:
    url = _fmp_historical_price_url(ticker, from_date, to_date, api_key)
    try:
        payload = client.get_json(url)
    except FetchError as exc:
        raise FetchError(str(exc).replace(api_key, "***")) from exc
    rows = _fmp_historical_rows(payload, ticker)
    if not rows:
        raise ValueError(f"No FMP historical EOD price rows returned for {ticker}.")
    return rows


def _fmp_historical_price_url(ticker: str, from_date: date, to_date: date, api_key: str) -> str:
    return FMP_HISTORICAL_PRICE_EOD_URL.format(
        ticker=ticker.upper(),
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        api_key=api_key,
    )


def yahoo_chart_url(ticker: str, from_date: date, to_date: date) -> str:
    period1, period2 = _yahoo_chart_periods(from_date, to_date)
    return YAHOO_CHART_URL.format(ticker=ticker.upper(), period1=period1, period2=period2)


def _fmp_historical_rows(payload: object, ticker: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        message = _fmp_error_message(payload)
        if message:
            raise ValueError(f"FMP historical EOD price API returned an error for {ticker}: {message}")
        for key in ("historical", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unexpected FMP historical EOD price payload for {ticker}.")


def _fmp_error_message(payload: dict[str, Any]) -> str | None:
    for key in ("Error Message", "Information", "Note", "message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _fmp_market_price_bar_from_row(
    *,
    ticker: str,
    row: dict[str, Any],
    benchmark_close_by_date: dict[str, float],
    source_path: str,
) -> MarketPriceBar:
    bar_date = _required_text(row.get("date"), f"FMP historical EOD row for {ticker} is missing date.")
    close = _required_number(row.get("close"), f"FMP historical EOD row for {ticker} on {bar_date} is missing close.")
    volume = _required_number(row.get("volume"), f"FMP historical EOD row for {ticker} on {bar_date} is missing volume.")
    if close <= 0:
        raise ValueError(f"FMP historical EOD row for {ticker} on {bar_date} has non-positive close.")
    if volume < 0:
        raise ValueError(f"FMP historical EOD row for {ticker} on {bar_date} has negative volume.")
    content_hash = hashlib.sha256(
        "|".join([ticker, bar_date, FMP_HISTORICAL_PRICE_SOURCE_TYPE, FMP_HISTORICAL_PRICE_SOURCE_NAME]).encode(
            "utf-8"
        )
    ).hexdigest()
    return MarketPriceBar(
        ticker=ticker,
        bar_date=bar_date,
        open=_optional_number(row.get("open")),
        high=_optional_number(row.get("high")),
        low=_optional_number(row.get("low")),
        close=close,
        volume=volume,
        benchmark_close=benchmark_close_by_date.get(bar_date),
        premarket_price=None,
        source_type=FMP_HISTORICAL_PRICE_SOURCE_TYPE,
        source_name=FMP_HISTORICAL_PRICE_SOURCE_NAME,
        source_path=source_path,
        content_hash=content_hash,
    )


def _fmp_historical_close(ticker: str, row: dict[str, Any]) -> tuple[str, float]:
    bar_date = _required_text(row.get("date"), f"FMP benchmark row for {ticker} is missing date.")
    close = _required_number(row.get("close"), f"FMP benchmark row for {ticker} on {bar_date} is missing close.")
    if close <= 0:
        raise ValueError(f"FMP benchmark row for {ticker} on {bar_date} has non-positive close.")
    return bar_date, close


def _fetch_yahoo_chart_rows(
    client: HttpClient,
    ticker: str,
    from_date: date,
    to_date: date,
) -> list[dict[str, Any]]:
    payload = client.get_json(yahoo_chart_url(ticker, from_date, to_date))
    rows = _yahoo_chart_rows(payload, ticker)
    if not rows:
        raise ValueError(f"No Yahoo chart price rows returned for {ticker}.")
    return rows


def _yahoo_chart_periods(from_date: date, to_date: date) -> tuple[int, int]:
    period1 = int(datetime.combine(from_date, datetime.min.time(), timezone.utc).timestamp())
    period2 = int(datetime.combine(to_date + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp())
    return period1, period2


def _yahoo_chart_rows(payload: object, ticker: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Yahoo chart payload for {ticker}.")
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError(f"Unexpected Yahoo chart payload for {ticker}.")
    errors = chart.get("error")
    if errors:
        raise ValueError(f"Yahoo chart endpoint returned an error for {ticker}: {errors}")
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        raise ValueError(f"No Yahoo chart result returned for {ticker}.")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected Yahoo chart result for {ticker}.")
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ValueError(f"Yahoo chart result for {ticker} is missing timestamps or indicators.")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
        raise ValueError(f"Yahoo chart result for {ticker} is missing quote indicators.")
    quote = quotes[0]
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, (int, float)):
            continue
        row = {
            "date": datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat(),
            "open": _list_value(quote.get("open"), index),
            "high": _list_value(quote.get("high"), index),
            "low": _list_value(quote.get("low"), index),
            "close": _list_value(quote.get("close"), index),
            "volume": _list_value(quote.get("volume"), index),
        }
        if row["close"] is None or row["volume"] is None:
            continue
        rows.append(row)
    return rows


def _list_value(values: object, index: int) -> Any:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def _yahoo_market_price_bar_from_row(
    *,
    ticker: str,
    row: dict[str, Any],
    benchmark_close_by_date: dict[str, float],
    source_path: str,
) -> MarketPriceBar:
    bar_date = _required_text(row.get("date"), f"Yahoo chart row for {ticker} is missing date.")
    close = _required_number(row.get("close"), f"Yahoo chart row for {ticker} on {bar_date} is missing close.")
    volume = _required_number(row.get("volume"), f"Yahoo chart row for {ticker} on {bar_date} is missing volume.")
    if close <= 0:
        raise ValueError(f"Yahoo chart row for {ticker} on {bar_date} has non-positive close.")
    if volume < 0:
        raise ValueError(f"Yahoo chart row for {ticker} on {bar_date} has negative volume.")
    content_hash = hashlib.sha256(
        "|".join([ticker, bar_date, YAHOO_CHART_SOURCE_TYPE, YAHOO_CHART_SOURCE_NAME]).encode("utf-8")
    ).hexdigest()
    return MarketPriceBar(
        ticker=ticker,
        bar_date=bar_date,
        open=_optional_number(row.get("open")),
        high=_optional_number(row.get("high")),
        low=_optional_number(row.get("low")),
        close=close,
        volume=volume,
        benchmark_close=benchmark_close_by_date.get(bar_date),
        premarket_price=None,
        source_type=YAHOO_CHART_SOURCE_TYPE,
        source_name=YAHOO_CHART_SOURCE_NAME,
        source_path=source_path,
        content_hash=content_hash,
    )


def _yahoo_chart_close(ticker: str, row: dict[str, Any]) -> tuple[str, float]:
    bar_date = _required_text(row.get("date"), f"Yahoo benchmark row for {ticker} is missing date.")
    close = _required_number(row.get("close"), f"Yahoo benchmark row for {ticker} on {bar_date} is missing close.")
    if close <= 0:
        raise ValueError(f"Yahoo benchmark row for {ticker} on {bar_date} has non-positive close.")
    return bar_date, close


def _market_bar_tickers(conn: sqlite3.Connection, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT ticker
        FROM market_price_bars
        WHERE ticker IS NOT NULL AND ticker != ''
        GROUP BY ticker
        ORDER BY ticker
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row["ticker"] for row in rows]


def _price_rows_for_ticker(conn: sqlite3.Connection, ticker: str, *, lookback: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT ticker, bar_date, open, close, volume, benchmark_close,
               premarket_price, source_type, source_name, source_path
        FROM market_price_bars
        WHERE ticker = ?
        ORDER BY bar_date DESC, updated_at DESC
        LIMIT ?
        """,
        (ticker, lookback + 2),
    ).fetchall()


def _signals_from_price_rows(
    ticker: str,
    rows: list[sqlite3.Row],
    *,
    lookback: int,
    min_volume_multiple: float,
    min_gap_pct: float,
    min_relative_strength_pct: float,
) -> list[MarketConfirmationSignal]:
    latest = rows[0]
    previous = rows[1]
    history = rows[1:]
    signals: list[MarketConfirmationSignal] = []
    previous_close = _row_float(previous, "close")
    latest_close = _row_float(latest, "close")
    if previous_close is None or previous_close <= 0 or latest_close is None:
        return signals

    historical_volumes = [
        value for value in (_row_float(row, "volume") for row in history[:lookback]) if value is not None and value > 0
    ]
    if historical_volumes:
        average_volume = sum(historical_volumes) / len(historical_volumes)
        latest_volume = _row_float(latest, "volume") or 0.0
        if average_volume > 0:
            volume_multiple = latest_volume / average_volume
            if volume_multiple >= min_volume_multiple:
                direction = "bullish" if latest_close >= previous_close else "neutral"
                signals.append(
                    _detected_signal(
                        latest,
                        signal_type="volume_spike",
                        direction=direction,
                        magnitude=volume_multiple,
                        confidence=min(0.9, 0.45 + min(volume_multiple / 6.0, 0.45)),
                        description=(
                            f"Local price-bar evidence: latest volume {latest_volume:.0f} is "
                            f"{volume_multiple:.2f}x the {len(historical_volumes)}-bar average "
                            f"for {ticker}; market-confirmation input only."
                        ),
                    )
                )

    gap_reference = previous_close
    gap_price = _row_float(latest, "premarket_price")
    gap_label = "premarket_price"
    if gap_price is None:
        gap_price = _row_float(latest, "open")
        gap_label = "open"
    if gap_price is not None and gap_reference > 0:
        gap_pct = (gap_price - gap_reference) / gap_reference
        if abs(gap_pct) >= min_gap_pct:
            signals.append(
                _detected_signal(
                    latest,
                    signal_type="premarket_or_open_gap",
                    direction="bullish" if gap_pct > 0 else "bearish",
                    magnitude=gap_pct,
                    confidence=min(0.9, 0.45 + min(abs(gap_pct) / 0.12, 0.45)),
                    description=(
                        f"Local price-bar evidence: {gap_label} gap for {ticker} was "
                        f"{gap_pct * 100:.2f}% versus the prior close; market-confirmation input only."
                    ),
                )
            )

    latest_benchmark = _row_float(latest, "benchmark_close")
    previous_benchmark = _row_float(previous, "benchmark_close")
    if latest_benchmark is not None and previous_benchmark is not None and previous_benchmark > 0:
        stock_return = (latest_close - previous_close) / previous_close
        benchmark_return = (latest_benchmark - previous_benchmark) / previous_benchmark
        relative_return = stock_return - benchmark_return
        if abs(relative_return) >= min_relative_strength_pct:
            signals.append(
                _detected_signal(
                    latest,
                    signal_type="relative_strength",
                    direction="bullish" if relative_return > 0 else "bearish",
                    magnitude=relative_return,
                    confidence=min(0.9, 0.45 + min(abs(relative_return) / 0.12, 0.45)),
                    description=(
                        f"Local price-bar evidence: {ticker} one-bar return exceeded benchmark "
                        f"by {relative_return * 100:.2f} percentage points; confirmation input only."
                    ),
                )
            )

    closes = [value for value in (_row_float(row, "close") for row in history[:lookback]) if value is not None]
    if len(closes) >= lookback:
        moving_average = sum(closes) / len(closes)
        if previous_close <= moving_average < latest_close:
            signals.append(
                _detected_signal(
                    latest,
                    signal_type="moving_average_breakout",
                    direction="bullish",
                    magnitude=(latest_close - moving_average) / moving_average if moving_average else None,
                    confidence=0.65,
                    description=(
                        f"Local price-bar evidence: {ticker} close crossed above the "
                        f"{lookback}-bar average; confirmation input only."
                    ),
                )
            )
        elif previous_close >= moving_average > latest_close:
            signals.append(
                _detected_signal(
                    latest,
                    signal_type="moving_average_breakdown",
                    direction="bearish",
                    magnitude=(latest_close - moving_average) / moving_average if moving_average else None,
                    confidence=0.65,
                    description=(
                        f"Local price-bar evidence: {ticker} close crossed below the "
                        f"{lookback}-bar average; risk-review input only."
                    ),
                )
            )
    return signals


def _detected_signal(
    row: sqlite3.Row,
    *,
    signal_type: str,
    direction: str,
    magnitude: float | None,
    confidence: float,
    description: str,
) -> MarketConfirmationSignal:
    ticker = str(row["ticker"]).upper()
    signal_date = str(row["bar_date"])
    source_type = str(row["source_type"])
    source_name = str(row["source_name"])
    source_path = str(row["source_path"])
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                signal_date,
                source_type,
                source_name,
                source_path,
                signal_type,
                direction,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return MarketConfirmationSignal(
        ticker=ticker,
        signal_date=signal_date,
        source_type=source_type,
        source_name=source_name,
        source_path=source_path,
        signal_type=signal_type,
        direction=direction,
        magnitude=magnitude,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _signal_from_row(row: dict[str, str], csv_path: Path) -> MarketConfirmationSignal:
    ticker = row.get("ticker", "").strip().upper()
    signal_type = row.get("signal_type", "").strip().lower()
    direction = _normalize_direction(row.get("direction", "unknown"))
    description = row.get("description", "").strip()
    confidence = _bounded_float(row.get("confidence"), default=0.5, minimum=0.0, maximum=1.0)
    magnitude = _optional_float(row.get("magnitude"))
    signal_date = row.get("signal_date", "").strip() or None
    source_type = row.get("source_type", "").strip() or "local_market_confirmation_csv"
    source_name = row.get("source_name", "").strip() or csv_path.name
    source_path = row.get("source_path", "").strip() or str(csv_path)
    if not ticker:
        raise ValueError("Market confirmation CSV contains a row without ticker.")
    if not signal_type:
        raise ValueError(f"Market confirmation CSV row for {ticker} is missing signal_type.")
    if not description:
        raise ValueError(f"Market confirmation CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                signal_date or "",
                source_type,
                source_name,
                source_path,
                signal_type,
                direction,
                str(magnitude) if magnitude is not None else "",
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return MarketConfirmationSignal(
        ticker=ticker,
        signal_date=signal_date,
        source_type=source_type,
        source_name=source_name,
        source_path=source_path,
        signal_type=signal_type,
        direction=direction,
        magnitude=magnitude,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _normalize_direction(raw: str | None) -> str:
    value = (raw or "unknown").strip().lower()
    if value in {"bullish", "bearish", "neutral"}:
        return value
    return "unknown"


def _optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _required_float(raw: str | None, error: str) -> float:
    if raw is None or raw.strip() == "":
        raise ValueError(error)
    return float(raw)


def _required_text(raw: object, error: str) -> str:
    if raw is None:
        raise ValueError(error)
    value = str(raw).strip()
    if not value:
        raise ValueError(error)
    return value


def _required_number(raw: object, error: str) -> float:
    value = _optional_number(raw)
    if value is None:
        raise ValueError(error)
    return value


def _optional_number(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip().replace(",", "")
        if not raw:
            return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _row_float(row: sqlite3.Row, key: str) -> float | None:
    value = row[key]
    if value is None:
        return None
    return float(value)


def _bounded_float(raw: str | None, *, default: float, minimum: float, maximum: float) -> float:
    if raw is None or raw.strip() == "":
        value = default
    else:
        value = float(raw)
    return max(minimum, min(maximum, value))
