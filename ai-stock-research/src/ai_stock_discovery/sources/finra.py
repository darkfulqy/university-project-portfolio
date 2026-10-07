from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sqlite3

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


DEFAULT_SHORT_SALE_SOURCE = "FINRA Daily Short Sale Volume"
DEFAULT_DAILY_FILE_CODE = "CNMS"
FINRA_DAILY_SHORT_SALE_URL = "https://cdn.finra.org/equity/regsho/daily/{file_code}shvol{yyyymmdd}.txt"


@dataclass(frozen=True)
class ShortSaleVolumeRecord:
    ticker: str
    trade_date: str
    market: str | None
    short_volume: float | None
    short_exempt_volume: float | None
    total_volume: float | None
    short_volume_ratio: float | None
    source_name: str
    source_url: str


@dataclass(frozen=True)
class FinraDailyFetchResult:
    trade_date: str
    source_url: str
    records: list[ShortSaleVolumeRecord]


def import_short_sale_volume_file(
    conn: sqlite3.Connection,
    file_path: Path,
    *,
    source_name: str = DEFAULT_SHORT_SALE_SOURCE,
    source_url: str | None = None,
) -> int:
    records = parse_short_sale_volume_file(
        file_path,
        source_name=source_name,
        source_url=source_url or str(file_path),
    )
    return upsert_short_sale_volume(conn, records)


def fetch_finra_daily_short_sale_volume(
    client: HttpClient,
    *,
    trade_date: date,
    tickers: list[str] | None = None,
    file_code: str = DEFAULT_DAILY_FILE_CODE,
    source_name: str = DEFAULT_SHORT_SALE_SOURCE,
) -> FinraDailyFetchResult:
    source_url = finra_daily_short_sale_url(trade_date=trade_date, file_code=file_code)
    try:
        text = client.get_text(source_url, accept="text/plain,*/*")
    except FetchError as exc:
        raise FetchError(str(exc)) from exc
    records = parse_short_sale_volume_text(
        text,
        source_name=source_name,
        source_url=source_url,
        tickers=tickers,
    )
    if not text.strip():
        raise ValueError(f"FINRA daily short sale file is empty for {trade_date.isoformat()}.")
    return FinraDailyFetchResult(
        trade_date=trade_date.isoformat(),
        source_url=source_url,
        records=records,
    )


def fetch_latest_finra_daily_short_sale_volume(
    client: HttpClient,
    *,
    lookback_days: int = 10,
    end_date: date | None = None,
    tickers: list[str] | None = None,
    file_code: str = DEFAULT_DAILY_FILE_CODE,
    source_name: str = DEFAULT_SHORT_SALE_SOURCE,
) -> FinraDailyFetchResult:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    anchor_date = end_date or date.fromisoformat(utc_now_iso()[:10])
    failures: list[str] = []
    for offset in range(lookback_days):
        trade_date = anchor_date - timedelta(days=offset)
        try:
            return fetch_finra_daily_short_sale_volume(
                client,
                trade_date=trade_date,
                tickers=tickers,
                file_code=file_code,
                source_name=source_name,
            )
        except Exception as exc:
            failures.append(f"{trade_date.isoformat()}: {exc}")
    raise ValueError(
        "No FINRA daily short sale volume file was available in the lookback window. "
        + " | ".join(failures[:3])
    )


def finra_daily_short_sale_url(*, trade_date: date, file_code: str = DEFAULT_DAILY_FILE_CODE) -> str:
    normalized_file_code = file_code.strip().upper()
    if not normalized_file_code:
        raise ValueError("file_code must not be empty.")
    return FINRA_DAILY_SHORT_SALE_URL.format(
        file_code=normalized_file_code,
        yyyymmdd=trade_date.strftime("%Y%m%d"),
    )


def parse_short_sale_volume_file(
    file_path: Path,
    *,
    source_name: str = DEFAULT_SHORT_SALE_SOURCE,
    source_url: str,
) -> list[ShortSaleVolumeRecord]:
    text = file_path.read_text(encoding="utf-8-sig")
    return parse_short_sale_volume_text(text, source_name=source_name, source_url=source_url)


def parse_short_sale_volume_text(
    text: str,
    *,
    source_name: str = DEFAULT_SHORT_SALE_SOURCE,
    source_url: str,
    tickers: list[str] | None = None,
) -> list[ShortSaleVolumeRecord]:
    if not text.strip():
        return []
    ticker_filter = {ticker.strip().upper() for ticker in tickers or [] if ticker.strip()}
    first_line = text.splitlines()[0]
    delimiter = "|" if "|" in first_line else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    records: list[ShortSaleVolumeRecord] = []
    for row in reader:
        normalized = {_normalize_header(key): (value or "").strip() for key, value in row.items() if key}
        ticker = _first(normalized, "symbol", "ticker").upper()
        trade_date = _normalize_date(_first(normalized, "date", "trade_date"))
        if not ticker or not trade_date:
            continue
        if ticker_filter and ticker not in ticker_filter:
            continue
        short_volume = _optional_float(_first(normalized, "shortvolume", "short_volume"))
        short_exempt_volume = _optional_float(_first(normalized, "shortexemptvolume", "short_exempt_volume"))
        total_volume = _optional_float(_first(normalized, "totalvolume", "total_volume"))
        ratio = None
        if short_volume is not None and total_volume and total_volume > 0:
            ratio = short_volume / total_volume
        records.append(
            ShortSaleVolumeRecord(
                ticker=ticker,
                trade_date=trade_date,
                market=_first(normalized, "market") or None,
                short_volume=short_volume,
                short_exempt_volume=short_exempt_volume,
                total_volume=total_volume,
                short_volume_ratio=ratio,
                source_name=source_name,
                source_url=source_url,
            )
        )
    return records


def upsert_short_sale_volume(conn: sqlite3.Connection, records: list[ShortSaleVolumeRecord]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO short_sale_volume (
            ticker, trade_date, market, short_volume, short_exempt_volume,
            total_volume, short_volume_ratio, source_name, source_url, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, trade_date, market, source_name) DO UPDATE SET
            short_volume=excluded.short_volume,
            short_exempt_volume=excluded.short_exempt_volume,
            total_volume=excluded.total_volume,
            short_volume_ratio=excluded.short_volume_ratio,
            source_url=excluded.source_url,
            updated_at=excluded.updated_at
        """,
        [
            (
                record.ticker,
                record.trade_date,
                record.market,
                record.short_volume,
                record.short_exempt_volume,
                record.total_volume,
                record.short_volume_ratio,
                record.source_name,
                record.source_url,
                now,
            )
            for record in records
        ],
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
            summary=excluded.summary,
            evidence_snippet=excluded.evidence_snippet,
            related_ticker=excluded.related_ticker
        """,
        [
            (
                "FINRA short sale volume",
                record.source_name,
                record.source_url,
                record.trade_date,
                now,
                "Daily short sale volume",
                _record_summary(record),
                _record_summary(record),
                0.75,
                record.ticker,
                "short_sale_volume",
                f"finra-short-sale:{record.source_name}:{record.market or ''}:{record.ticker}:{record.trade_date}",
            )
            for record in records
        ],
    )
    return len(records)


def _record_summary(record: ShortSaleVolumeRecord) -> str:
    ratio = "NA" if record.short_volume_ratio is None else f"{record.short_volume_ratio:.4f}"
    return (
        f"{record.ticker} {record.trade_date} FINRA short sale volume: "
        f"short_volume={_fmt(record.short_volume)}, "
        f"short_exempt_volume={_fmt(record.short_exempt_volume)}, "
        f"total_volume={_fmt(record.total_volume)}, ratio={ratio}. "
        "This is daily short sale volume, not short interest."
    )


def _normalize_header(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum() or char == "_")


def _first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        normalized = _normalize_header(key)
        if row.get(normalized):
            return row[normalized]
    return ""


def _normalize_date(raw: str) -> str:
    value = raw.strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _optional_float(raw: str) -> float | None:
    if raw.strip() == "":
        return None
    return float(raw.replace(",", ""))


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:g}"
