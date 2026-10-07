from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class GuidanceEvent:
    ticker: str
    guidance_date: str | None
    fiscal_period: str | None
    metric: str
    previous_value: float | None
    current_value: float | None
    unit: str | None
    direction: str
    source_type: str
    source_name: str
    source_url: str
    description: str
    confidence: float
    content_hash: str


def import_guidance_events_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "source_url", "metric", "direction", "description", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Guidance events CSV missing required columns: {', '.join(sorted(missing))}")
        events = [_event_from_row(row, csv_path) for row in reader]
    return upsert_guidance_events(conn, events)


def upsert_guidance_events(conn: sqlite3.Connection, events: list[GuidanceEvent]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO guidance_events (
            ticker, guidance_date, fiscal_period, metric, previous_value,
            current_value, unit, direction, source_type, source_name, source_url,
            description, confidence, content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            guidance_date=excluded.guidance_date,
            fiscal_period=excluded.fiscal_period,
            metric=excluded.metric,
            previous_value=excluded.previous_value,
            current_value=excluded.current_value,
            unit=excluded.unit,
            direction=excluded.direction,
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
                event.guidance_date,
                event.fiscal_period,
                event.metric,
                event.previous_value,
                event.current_value,
                event.unit,
                event.direction,
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
                event.guidance_date,
                now,
                f"{event.ticker} guidance event",
                event.description,
                _evidence_snippet(event),
                event.confidence,
                event.ticker,
                "guidance_event",
                f"guidance-event:{event.content_hash}",
            )
            for event in events
        ],
    )
    return len(events)


def _event_from_row(row: dict[str, str], csv_path: Path) -> GuidanceEvent:
    ticker = _field(row, "ticker").upper()
    source_url = _field(row, "source_url")
    metric = _normalize_metric(_field(row, "metric"))
    direction = _normalize_direction(_field(row, "direction"))
    description = _field(row, "description")
    confidence = _bounded_float(_field(row, "confidence"), default=0.5, minimum=0.0, maximum=1.0)
    guidance_date = _field(row, "guidance_date", "date") or None
    fiscal_period = _field(row, "fiscal_period", "period") or None
    previous_value = _optional_float(_field(row, "previous_value", "prior_value"))
    current_value = _optional_float(_field(row, "current_value", "new_value"))
    unit = _field(row, "unit") or None
    source_type = _field(row, "source_type") or "guidance_event_csv"
    source_name = _field(row, "source_name") or csv_path.name
    if not ticker:
        raise ValueError("Guidance events CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Guidance events CSV row for {ticker} is missing source_url.")
    if not metric:
        raise ValueError(f"Guidance events CSV row for {ticker} is missing metric.")
    if not description:
        raise ValueError(f"Guidance events CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                guidance_date or "",
                fiscal_period or "",
                metric,
                f"{previous_value:g}" if previous_value is not None else "",
                f"{current_value:g}" if current_value is not None else "",
                unit or "",
                direction,
                source_type,
                source_name,
                source_url,
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return GuidanceEvent(
        ticker=ticker,
        guidance_date=guidance_date,
        fiscal_period=fiscal_period,
        metric=metric,
        previous_value=previous_value,
        current_value=current_value,
        unit=unit,
        direction=direction,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _normalize_direction(raw: str) -> str:
    value = raw.strip().lower()
    aliases = {
        "raise": "raised",
        "raised": "raised",
        "increase": "raised",
        "increased": "raised",
        "improve": "raised",
        "improved": "raised",
        "upgrade": "raised",
        "upgraded": "raised",
        "lower": "lowered",
        "lowered": "lowered",
        "cut": "lowered",
        "reduced": "lowered",
        "decrease": "lowered",
        "decreased": "lowered",
        "reiterate": "reiterated",
        "reiterated": "reiterated",
        "initiated": "initiated",
        "withdrawn": "withdrawn",
        "neutral": "neutral",
    }
    return aliases.get(value, "neutral")


def _normalize_metric(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "revenue": "revenue",
        "sales": "revenue",
        "eps": "eps",
        "ebitda": "ebitda",
        "operating_income": "operating_income",
        "gross_margin": "gross_margin",
        "margin": "margin",
        "free_cash_flow": "free_cash_flow",
        "capex": "capex",
        "rpo": "rpo",
        "backlog": "backlog",
        "billings": "billings",
        "ai_revenue": "ai_revenue",
    }
    return aliases.get(value, value)


def _evidence_snippet(event: GuidanceEvent) -> str:
    values = []
    if event.previous_value is not None:
        values.append(f"previous={event.previous_value:g}")
    if event.current_value is not None:
        values.append(f"current={event.current_value:g}")
    if event.unit:
        values.append(f"unit={event.unit}")
    value_text = ", ".join(values) if values else "values=NA"
    return (
        f"{event.metric} guidance {event.direction} for {event.fiscal_period or 'period NA'}; "
        f"{value_text}. {event.description}"
    )


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
