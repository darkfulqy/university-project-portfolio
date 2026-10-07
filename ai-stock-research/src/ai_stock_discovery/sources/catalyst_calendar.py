from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from pathlib import Path
import sqlite3
from typing import Any

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


FMP_EARNINGS_CALENDAR_SOURCE_NAME = "Financial Modeling Prep earnings calendar API"
FMP_EARNINGS_CALENDAR_URL = "https://financialmodelingprep.com/stable/earnings-calendar"


@dataclass(frozen=True)
class CatalystCalendarEntry:
    ticker: str
    catalyst_type: str
    catalyst_date: str
    description: str
    source_url: str
    confidence: float
    status: str
    content_hash: str
    source_name: str = "Manual catalyst calendar CSV"


def fetch_fmp_earnings_calendar_entries(
    client: HttpClient,
    tickers: list[str],
    api_key: str,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
    limit_per_ticker: int = 2,
) -> list[CatalystCalendarEntry]:
    if limit_per_ticker <= 0:
        raise ValueError("limit_per_ticker must be positive.")
    normalized_tickers = _unique_tickers(tickers)
    if not normalized_tickers:
        raise ValueError("at least one ticker is required.")
    start = date.fromisoformat(from_date) if from_date else date.fromisoformat(utc_now_iso()[:10])
    end = date.fromisoformat(to_date) if to_date else start + timedelta(days=365)
    if end < start:
        raise ValueError("to_date must be on or after from_date.")

    url = _fmp_earnings_calendar_url(from_date=start.isoformat(), to_date=end.isoformat(), api_key=api_key)
    try:
        payload = client.get_json(url)
    except FetchError as exc:
        raise FetchError(str(exc).replace(api_key, "***")) from exc
    if not isinstance(payload, list):
        raise ValueError("Unexpected FMP earnings calendar payload.")

    source_url = _fmp_earnings_calendar_url(from_date=start.isoformat(), to_date=end.isoformat(), api_key="***")
    by_ticker: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in normalized_tickers}
    for row in payload:
        if not isinstance(row, dict):
            continue
        ticker = _text(row.get("symbol")).upper()
        if ticker in by_ticker and _text(row.get("date")):
            by_ticker[ticker].append(row)

    entries: list[CatalystCalendarEntry] = []
    for ticker, rows in by_ticker.items():
        rows = sorted(rows, key=lambda row: _text(row.get("date")))[:limit_per_ticker]
        for row in rows:
            catalyst_date = _text(row.get("date"))
            description = (
                f"FMP earnings calendar lists {ticker} earnings date {catalyst_date}. "
                "Consensus estimate fields, when present, are source-provided metadata and are not inferred by this system."
            )
            entries.append(
                CatalystCalendarEntry(
                    ticker=ticker,
                    catalyst_type="earnings_or_filing",
                    catalyst_date=catalyst_date,
                    description=description,
                    source_url=source_url,
                    confidence=0.7,
                    status="scheduled",
                    content_hash=_content_hash(ticker, "earnings_or_filing", catalyst_date, description, source_url),
                    source_name=FMP_EARNINGS_CALENDAR_SOURCE_NAME,
                )
            )
    return entries


def import_catalyst_calendar_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "catalyst_type", "catalyst_date", "description", "source_url", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Catalyst calendar CSV missing required columns: {', '.join(sorted(missing))}")
        entries = [_entry_from_row(row) for row in reader]
    return upsert_catalyst_calendar_entries(conn, entries)


def upsert_catalyst_calendar_entries(
    conn: sqlite3.Connection,
    entries: list[CatalystCalendarEntry],
) -> int:
    now = utc_now_iso()
    inserted = 0
    for entry in entries:
        existing = conn.execute(
            """
            SELECT id FROM catalysts
            WHERE ticker = ?
              AND catalyst_type = ?
              AND catalyst_date = ?
              AND description = ?
              AND source_url = ?
            """,
            (
                entry.ticker,
                entry.catalyst_type,
                entry.catalyst_date,
                entry.description,
                entry.source_url,
            ),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE catalysts
                SET confidence = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (entry.confidence, entry.status, now, existing["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO catalysts (
                    ticker, catalyst_type, catalyst_date, description,
                    source_url, confidence, status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.ticker,
                    entry.catalyst_type,
                    entry.catalyst_date,
                    entry.description,
                    entry.source_url,
                    entry.confidence,
                    entry.status,
                    now,
                ),
            )
            inserted += 1
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
                "Catalyst calendar",
                entry.source_name,
                entry.source_url,
                entry.catalyst_date,
                now,
                entry.catalyst_type,
                entry.description,
                entry.description,
                entry.confidence,
                entry.ticker,
                "catalyst_calendar",
                f"catalyst-calendar:{entry.content_hash}",
            )
            for entry in entries
        ],
    )
    return inserted


def list_upcoming_catalysts(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    from_date: str | None = None,
    days: int = 365,
    limit: int = 50,
) -> list[sqlite3.Row]:
    start = date.fromisoformat(from_date) if from_date else date.today()
    end = start + timedelta(days=max(0, days))
    params: list[object] = [start.isoformat(), end.isoformat()]
    where = "catalyst_date IS NOT NULL AND catalyst_date BETWEEN ? AND ?"
    if ticker:
        where += " AND ticker = ?"
        params.append(ticker.upper())
    params.append(limit)
    return conn.execute(
        f"""
        SELECT ticker, catalyst_type, catalyst_date, confidence, status, source_url, description
        FROM catalysts
        WHERE {where}
          AND status IN ('scheduled', 'candidate', 'confirmed')
        ORDER BY catalyst_date, confidence DESC, ticker
        LIMIT ?
        """,
        params,
    ).fetchall()


def _entry_from_row(row: dict[str, str]) -> CatalystCalendarEntry:
    ticker = row.get("ticker", "").strip().upper()
    catalyst_type = row.get("catalyst_type", "").strip().lower()
    catalyst_date = row.get("catalyst_date", "").strip()
    description = row.get("description", "").strip()
    source_url = row.get("source_url", "").strip()
    confidence = _bounded_float(row.get("confidence"), default=0.5, minimum=0.0, maximum=1.0)
    status = _normalize_status(row.get("status"))
    if not ticker:
        raise ValueError("Catalyst calendar CSV contains a row without ticker.")
    if not catalyst_type:
        raise ValueError(f"Catalyst calendar CSV row for {ticker} is missing catalyst_type.")
    if not catalyst_date:
        raise ValueError(f"Catalyst calendar CSV row for {ticker} is missing catalyst_date.")
    date.fromisoformat(catalyst_date)
    if not description:
        raise ValueError(f"Catalyst calendar CSV row for {ticker} is missing description.")
    if not source_url:
        raise ValueError(f"Catalyst calendar CSV row for {ticker} is missing source_url.")
    content_hash = _content_hash(ticker, catalyst_type, catalyst_date, description, source_url)
    return CatalystCalendarEntry(
        ticker=ticker,
        catalyst_type=catalyst_type,
        catalyst_date=catalyst_date,
        description=description,
        source_url=source_url,
        confidence=confidence,
        status=status,
        content_hash=content_hash,
    )


def _normalize_status(raw: str | None) -> str:
    value = (raw or "scheduled").strip().lower()
    if value in {"scheduled", "candidate", "confirmed", "cancelled", "done"}:
        return value
    return "scheduled"


def _bounded_float(raw: str | None, *, default: float, minimum: float, maximum: float) -> float:
    if raw is None or raw.strip() == "":
        value = default
    else:
        value = float(raw)
    return max(minimum, min(maximum, value))


def _fmp_earnings_calendar_url(*, from_date: str, to_date: str, api_key: str) -> str:
    return f"{FMP_EARNINGS_CALENDAR_URL}?from={from_date}&to={to_date}&apikey={api_key}"


def _content_hash(ticker: str, catalyst_type: str, catalyst_date: str, description: str, source_url: str) -> str:
    return hashlib.sha256(
        "|".join([ticker, catalyst_type, catalyst_date, description, source_url]).encode("utf-8")
    ).hexdigest()


def _unique_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for ticker in tickers:
        value = ticker.strip().upper()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
