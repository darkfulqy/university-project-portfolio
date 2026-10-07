from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class InstitutionalHoldingEvent:
    ticker: str
    event_date: str | None
    report_period: str | None
    filing_type: str | None
    institution_name: str
    manager_cik: str | None
    event_type: str
    direction: str
    shares_held: float | None
    shares_change: float | None
    market_value: float | None
    percent_shares_outstanding: float | None
    source_type: str
    source_name: str
    source_url: str
    description: str
    confidence: float
    content_hash: str


def import_institutional_holding_events_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {
            "ticker",
            "source_url",
            "institution_name",
            "event_type",
            "description",
            "confidence",
        }
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Institutional holding events CSV missing required columns: {', '.join(sorted(missing))}")
        events = [_event_from_row(row, csv_path) for row in reader]
    return upsert_institutional_holding_events(conn, events)


def upsert_institutional_holding_events(
    conn: sqlite3.Connection,
    events: list[InstitutionalHoldingEvent],
) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO institutional_holding_events (
            ticker, event_date, report_period, filing_type, institution_name,
            manager_cik, event_type, direction, shares_held, shares_change,
            market_value, percent_shares_outstanding, source_type, source_name,
            source_url, description, confidence, content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            event_date=excluded.event_date,
            report_period=excluded.report_period,
            filing_type=excluded.filing_type,
            institution_name=excluded.institution_name,
            manager_cik=excluded.manager_cik,
            event_type=excluded.event_type,
            direction=excluded.direction,
            shares_held=excluded.shares_held,
            shares_change=excluded.shares_change,
            market_value=excluded.market_value,
            percent_shares_outstanding=excluded.percent_shares_outstanding,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            description=excluded.description,
            confidence=excluded.confidence,
            updated_at=excluded.updated_at
        """,
        [
            (
                event.ticker,
                event.event_date,
                event.report_period,
                event.filing_type,
                event.institution_name,
                event.manager_cik,
                event.event_type,
                event.direction,
                event.shares_held,
                event.shares_change,
                event.market_value,
                event.percent_shares_outstanding,
                event.source_type,
                event.source_name,
                event.source_url,
                event.description,
                event.confidence,
                event.content_hash,
                now,
            )
            for event in events
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
            raw_title=excluded.raw_title,
            summary=excluded.summary,
            evidence_snippet=excluded.evidence_snippet,
            confidence=excluded.confidence,
            related_ticker=excluded.related_ticker
        """,
        [
            (
                event.source_type,
                event.source_name,
                event.source_url,
                event.event_date,
                now,
                f"{event.ticker} institutional holding {event.event_type}",
                event.description,
                _evidence_snippet(event),
                event.confidence,
                event.ticker,
                "institutional_holding_event",
                f"institutional-holding:{event.content_hash}",
            )
            for event in events
        ],
    )
    return len(events)


def _event_from_row(row: dict[str, str], csv_path: Path) -> InstitutionalHoldingEvent:
    ticker = _field(row, "ticker").upper()
    source_url = _field(row, "source_url")
    institution_name = _field(row, "institution_name", "manager_name", "holder_name")
    event_type = _normalize_event_type(_field(row, "event_type"))
    explicit_direction = _field(row, "direction")
    direction = _normalize_direction(explicit_direction) if explicit_direction else _direction_from_event_type(event_type)
    description = _field(row, "description")
    confidence = _bounded_float(_field(row, "confidence"), default=0.5, minimum=0.0, maximum=1.0)
    event_date = _field(row, "event_date", "filing_date", "date") or None
    report_period = _field(row, "report_period", "period") or None
    filing_type = _field(row, "filing_type", "form") or None
    manager_cik = _field(row, "manager_cik", "cik") or None
    shares_held = _optional_float(_field(row, "shares_held", "current_shares"))
    shares_change = _optional_float(_field(row, "shares_change", "change_shares", "delta_shares"))
    market_value = _optional_float(_field(row, "market_value", "value"))
    percent_shares_outstanding = _optional_float(_field(row, "percent_shares_outstanding", "ownership_pct"))
    source_type = _field(row, "source_type") or "institutional_holding_csv"
    source_name = _field(row, "source_name") or csv_path.name
    if not ticker:
        raise ValueError("Institutional holding events CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Institutional holding events CSV row for {ticker} is missing source_url.")
    if not institution_name:
        raise ValueError(f"Institutional holding events CSV row for {ticker} is missing institution_name.")
    if not event_type:
        raise ValueError(f"Institutional holding events CSV row for {ticker} is missing event_type.")
    if not description:
        raise ValueError(f"Institutional holding events CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                event_date or "",
                report_period or "",
                filing_type or "",
                institution_name,
                manager_cik or "",
                event_type,
                direction,
                f"{shares_held:g}" if shares_held is not None else "",
                f"{shares_change:g}" if shares_change is not None else "",
                f"{market_value:g}" if market_value is not None else "",
                f"{percent_shares_outstanding:g}" if percent_shares_outstanding is not None else "",
                source_type,
                source_name,
                source_url,
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return InstitutionalHoldingEvent(
        ticker=ticker,
        event_date=event_date,
        report_period=report_period,
        filing_type=filing_type,
        institution_name=institution_name,
        manager_cik=manager_cik,
        event_type=event_type,
        direction=direction,
        shares_held=shares_held,
        shares_change=shares_change,
        market_value=market_value,
        percent_shares_outstanding=percent_shares_outstanding,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _normalize_event_type(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "new": "new_position",
        "new_position": "new_position",
        "initiated": "new_position",
        "increase": "increased_position",
        "increased": "increased_position",
        "increased_position": "increased_position",
        "decrease": "decreased_position",
        "decreased": "decreased_position",
        "decreased_position": "decreased_position",
        "exit": "exited_position",
        "exited": "exited_position",
        "sold_out": "exited_position",
        "activist": "activist_stake",
        "activist_stake": "activist_stake",
        "13d": "activist_stake",
        "13g": "passive_large_holder",
        "passive_large_holder": "passive_large_holder",
        "unchanged": "unchanged_position",
    }
    return aliases.get(value, value)


def _normalize_direction(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "bullish": "bullish",
        "supportive": "bullish",
        "accumulation": "bullish",
        "bearish": "bearish",
        "distribution": "bearish",
        "negative": "bearish",
        "neutral": "neutral",
        "mixed": "neutral",
    }
    return aliases.get(value, "neutral")


def _direction_from_event_type(event_type: str) -> str:
    if event_type in {"new_position", "increased_position", "activist_stake", "passive_large_holder"}:
        return "bullish"
    if event_type in {"decreased_position", "exited_position"}:
        return "bearish"
    return "neutral"


def _evidence_snippet(event: InstitutionalHoldingEvent) -> str:
    values = [
        f"institution={event.institution_name}",
        f"event={event.event_type}",
        f"direction={event.direction}",
    ]
    if event.report_period:
        values.append(f"period={event.report_period}")
    if event.filing_type:
        values.append(f"filing={event.filing_type}")
    if event.shares_held is not None:
        values.append(f"shares_held={event.shares_held:g}")
    if event.shares_change is not None:
        values.append(f"shares_change={event.shares_change:g}")
    if event.market_value is not None:
        values.append(f"market_value={event.market_value:g}")
    if event.percent_shares_outstanding is not None:
        values.append(f"ownership_pct={event.percent_shares_outstanding:g}")
    values.append(event.description)
    return "; ".join(values)


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _optional_float(raw: str) -> float | None:
    if not raw:
        return None
    return float(raw.replace(",", ""))


def _bounded_float(raw: str, *, default: float, minimum: float, maximum: float) -> float:
    value = default if not raw else float(raw)
    return max(minimum, min(maximum, value))
