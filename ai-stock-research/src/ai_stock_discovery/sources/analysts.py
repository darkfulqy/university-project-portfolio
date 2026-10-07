from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import urlencode

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


FMP_ANALYST_ESTIMATES_SOURCE_NAME = "Financial Modeling Prep analyst estimates API"
FMP_ANALYST_ESTIMATES_SOURCE_TYPE = "fmp_analyst_estimates_api"
FMP_ANALYST_ESTIMATES_URL = "https://financialmodelingprep.com/stable/analyst-estimates"
FMP_ANALYST_ESTIMATES_MAX_LIMIT = 10


@dataclass(frozen=True)
class AnalystEstimateEvent:
    ticker: str
    event_date: str | None
    fiscal_period: str | None
    metric: str | None
    event_type: str
    direction: str
    previous_value: float | None
    current_value: float | None
    unit: str | None
    analyst_firm: str | None
    source_type: str
    source_name: str
    source_url: str
    description: str
    confidence: float
    content_hash: str


def import_analyst_estimate_events_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "source_url", "event_type", "direction", "description", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Analyst estimate events CSV missing required columns: {', '.join(sorted(missing))}")
        events = [_event_from_row(row, csv_path) for row in reader]
    return upsert_analyst_estimate_events(conn, events)


def fetch_fmp_analyst_estimate_events(
    client: HttpClient,
    ticker: str,
    api_key: str,
    *,
    period: str = "annual",
    limit: int = 5,
    low_coverage_threshold: int = 3,
) -> list[AnalystEstimateEvent]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if limit > FMP_ANALYST_ESTIMATES_MAX_LIMIT:
        raise ValueError(f"limit must be <= {FMP_ANALYST_ESTIMATES_MAX_LIMIT} for the FMP analyst estimates endpoint.")
    if low_coverage_threshold < 0:
        raise ValueError("low_coverage_threshold must be non-negative.")
    ticker = ticker.strip().upper()
    period = period.strip().lower()
    if not ticker:
        raise ValueError("ticker is required.")
    if period not in {"annual", "quarter"}:
        raise ValueError("period must be annual or quarter.")

    request_limit = max(limit, FMP_ANALYST_ESTIMATES_MAX_LIMIT)
    rows = _fetch_fmp_analyst_estimate_rows(
        client,
        ticker=ticker,
        api_key=api_key,
        period=period,
        limit=request_limit,
    )
    selected_rows = _select_fmp_estimate_rows(rows, limit=limit)
    source_url = _fmp_analyst_estimates_url(
        ticker=ticker,
        period=period,
        limit=request_limit,
        api_key="***",
    )
    events: list[AnalystEstimateEvent] = []
    for row in selected_rows:
        events.extend(
            _events_from_fmp_estimate_row(
                ticker=ticker,
                row=row,
                source_url=source_url,
                low_coverage_threshold=low_coverage_threshold,
            )
        )
    if not events:
        raise ValueError(f"No usable FMP analyst estimate rows returned for {ticker}.")
    return events


def _select_fmp_estimate_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    today = utc_now_iso()[:10]
    dated_rows = [(_text(row.get("date")), row) for row in rows]
    dated_rows = [(row_date, row) for row_date, row in dated_rows if row_date]
    future_rows = [(row_date, row) for row_date, row in dated_rows if row_date >= today]
    if future_rows:
        ordered = sorted(future_rows, key=lambda item: item[0])
    else:
        ordered = sorted(dated_rows, key=lambda item: item[0], reverse=True)
    return [row for _row_date, row in ordered[:limit]]


def upsert_analyst_estimate_events(
    conn: sqlite3.Connection,
    events: list[AnalystEstimateEvent],
) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO analyst_estimate_events (
            ticker, event_date, fiscal_period, metric, event_type, direction,
            previous_value, current_value, unit, analyst_firm, source_type,
            source_name, source_url, description, confidence, content_hash,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            event_date=excluded.event_date,
            fiscal_period=excluded.fiscal_period,
            metric=excluded.metric,
            event_type=excluded.event_type,
            direction=excluded.direction,
            previous_value=excluded.previous_value,
            current_value=excluded.current_value,
            unit=excluded.unit,
            analyst_firm=excluded.analyst_firm,
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
                event.fiscal_period,
                event.metric,
                event.event_type,
                event.direction,
                event.previous_value,
                event.current_value,
                event.unit,
                event.analyst_firm,
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
                f"{event.ticker} {event.event_type}",
                event.description,
                _evidence_snippet(event),
                event.confidence,
                event.ticker,
                "analyst_estimate_event",
                f"analyst-estimate:{event.content_hash}",
            )
            for event in events
        ],
    )
    return len(events)


def replace_fmp_analyst_estimate_events(
    conn: sqlite3.Connection,
    ticker: str,
    events: list[AnalystEstimateEvent],
) -> int:
    ticker = ticker.strip().upper()
    conn.execute(
        """
        DELETE FROM analyst_estimate_events
        WHERE ticker = ? AND source_type = ?
        """,
        (ticker, FMP_ANALYST_ESTIMATES_SOURCE_TYPE),
    )
    conn.execute(
        """
        DELETE FROM evidence_items
        WHERE related_ticker = ?
          AND related_module = 'analyst_estimate_event'
          AND source_type = ?
        """,
        (ticker, FMP_ANALYST_ESTIMATES_SOURCE_TYPE),
    )
    return upsert_analyst_estimate_events(conn, events)


def _event_from_row(row: dict[str, str], csv_path: Path) -> AnalystEstimateEvent:
    ticker = _field(row, "ticker").upper()
    source_url = _field(row, "source_url")
    event_type = _normalize_event_type(_field(row, "event_type", "signal_type"))
    direction = _normalize_direction(_field(row, "direction", "gap_direction"))
    description = _field(row, "description")
    confidence = _bounded_float(_field(row, "confidence"), default=0.5, minimum=0.0, maximum=1.0)
    event_date = _field(row, "event_date", "date", "signal_date") or None
    fiscal_period = _field(row, "fiscal_period", "period") or None
    metric = _normalize_metric(_field(row, "metric")) or None
    previous_value = _optional_float(_field(row, "previous_value", "prior_value", "previous_estimate"))
    current_value = _optional_float(_field(row, "current_value", "new_value", "current_estimate"))
    unit = _field(row, "unit") or None
    analyst_firm = _field(row, "analyst_firm", "firm", "source_firm") or None
    source_type = _field(row, "source_type") or "analyst_estimate_csv"
    source_name = _field(row, "source_name") or csv_path.name
    if not ticker:
        raise ValueError("Analyst estimate events CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Analyst estimate events CSV row for {ticker} is missing source_url.")
    if not event_type:
        raise ValueError(f"Analyst estimate events CSV row for {ticker} is missing event_type.")
    if not description:
        raise ValueError(f"Analyst estimate events CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                event_date or "",
                fiscal_period or "",
                metric or "",
                event_type,
                direction,
                f"{previous_value:g}" if previous_value is not None else "",
                f"{current_value:g}" if current_value is not None else "",
                unit or "",
                analyst_firm or "",
                source_type,
                source_name,
                source_url,
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return AnalystEstimateEvent(
        ticker=ticker,
        event_date=event_date,
        fiscal_period=fiscal_period,
        metric=metric,
        event_type=event_type,
        direction=direction,
        previous_value=previous_value,
        current_value=current_value,
        unit=unit,
        analyst_firm=analyst_firm,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _fetch_fmp_analyst_estimate_rows(
    client: HttpClient,
    *,
    ticker: str,
    api_key: str,
    period: str,
    limit: int,
) -> list[dict[str, Any]]:
    url = _fmp_analyst_estimates_url(
        ticker=ticker,
        period=period,
        limit=limit,
        api_key=api_key,
    )
    try:
        payload = client.get_json(url)
    except FetchError as exc:
        raise FetchError(str(exc).replace(api_key, "***")) from exc
    rows = _fmp_response_rows(payload, ticker)
    if not rows:
        raise ValueError(f"No FMP analyst estimate rows returned for {ticker}.")
    return rows


def _fmp_analyst_estimates_url(
    *,
    ticker: str,
    period: str,
    limit: int,
    api_key: str,
) -> str:
    params: dict[str, str | int] = {
        "symbol": ticker.upper(),
        "period": period,
        "page": 0,
        "limit": limit,
        "apikey": api_key,
    }
    return f"{FMP_ANALYST_ESTIMATES_URL}?{urlencode(params)}"


def _fmp_response_rows(payload: object, ticker: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        message = _fmp_error_message(payload)
        if message:
            raise ValueError(f"FMP analyst estimates API returned an error for {ticker}: {message}")
        for key in ("data", "analystEstimates", "estimates"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unexpected FMP analyst estimates payload for {ticker}.")


def _fmp_error_message(payload: dict[str, Any]) -> str | None:
    for key in ("Error Message", "Information", "Note", "message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _events_from_fmp_estimate_row(
    *,
    ticker: str,
    row: dict[str, Any],
    source_url: str,
    low_coverage_threshold: int,
) -> list[AnalystEstimateEvent]:
    row_ticker = str(row.get("symbol") or ticker).strip().upper()
    if row_ticker and row_ticker != ticker:
        return []
    fiscal_period = _text(row.get("date"))
    if not fiscal_period:
        return []

    revenue_avg = _optional_number(row.get("revenueAvg"))
    revenue_count = _optional_int(row.get("numAnalystsRevenue"))
    eps_avg = _optional_number(row.get("epsAvg"))
    eps_count = _optional_int(row.get("numAnalystsEps"))
    events: list[AnalystEstimateEvent] = []

    if revenue_avg is not None or revenue_count is not None:
        events.append(
            _make_fmp_event(
                ticker=ticker,
                fiscal_period=fiscal_period,
                metric="revenue",
                event_type="estimate_snapshot",
                direction="neutral",
                current_value=revenue_avg,
                unit=None,
                source_url=source_url,
                description=(
                    f"FMP analyst estimates snapshot for {ticker} period ending {fiscal_period}: "
                    f"revenueAvg={_fmt(revenue_avg)}, numAnalystsRevenue={_fmt_int(revenue_count)}. "
                    "Stored as source-backed analyst context only; no expectation-gap conclusion is inferred."
                ),
                confidence=0.75,
            )
        )
    if eps_avg is not None or eps_count is not None:
        events.append(
            _make_fmp_event(
                ticker=ticker,
                fiscal_period=fiscal_period,
                metric="eps",
                event_type="estimate_snapshot",
                direction="neutral",
                current_value=eps_avg,
                unit=None,
                source_url=source_url,
                description=(
                    f"FMP analyst estimates snapshot for {ticker} period ending {fiscal_period}: "
                    f"epsAvg={_fmt(eps_avg)}, numAnalystsEps={_fmt_int(eps_count)}. "
                    "Stored as source-backed analyst context only; no expectation-gap conclusion is inferred."
                ),
                confidence=0.75,
            )
        )

    observed_counts = [count for count in (revenue_count, eps_count) if count is not None]
    if observed_counts and min(observed_counts) <= low_coverage_threshold:
        events.append(
            _make_fmp_event(
                ticker=ticker,
                fiscal_period=fiscal_period,
                metric="coverage",
                event_type="low_analyst_coverage",
                direction="supports_gap",
                current_value=float(min(observed_counts)),
                unit="analyst_count",
                source_url=source_url,
                description=(
                    f"FMP analyst estimates snapshot for {ticker} period ending {fiscal_period} "
                    f"shows low coverage by configured threshold: min analyst count "
                    f"{min(observed_counts)} <= {low_coverage_threshold}. "
                    "This is a review cue only and does not prove a market expectation gap."
                ),
                confidence=0.65,
            )
        )

    return events


def _make_fmp_event(
    *,
    ticker: str,
    fiscal_period: str,
    metric: str,
    event_type: str,
    direction: str,
    current_value: float | None,
    unit: str | None,
    source_url: str,
    description: str,
    confidence: float,
) -> AnalystEstimateEvent:
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                fiscal_period,
                metric,
                event_type,
                direction,
                "" if current_value is None else f"{current_value:g}",
                unit or "",
                FMP_ANALYST_ESTIMATES_SOURCE_TYPE,
                FMP_ANALYST_ESTIMATES_SOURCE_NAME,
                source_url,
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return AnalystEstimateEvent(
        ticker=ticker,
        event_date=None,
        fiscal_period=fiscal_period,
        metric=metric,
        event_type=event_type,
        direction=direction,
        previous_value=None,
        current_value=current_value,
        unit=unit,
        analyst_firm=None,
        source_type=FMP_ANALYST_ESTIMATES_SOURCE_TYPE,
        source_name=FMP_ANALYST_ESTIMATES_SOURCE_NAME,
        source_url=source_url,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _normalize_event_type(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "estimate_lag": "estimate_lag",
        "consensus_lag": "estimate_lag",
        "not_revised": "estimate_lag",
        "estimate_revision": "estimate_revision",
        "consensus_revision": "estimate_revision",
        "revenue_estimate_revision": "estimate_revision",
        "eps_estimate_revision": "estimate_revision",
        "price_target_revision": "price_target_revision",
        "target_revision": "price_target_revision",
        "rating_change": "rating_change",
        "rating_revision": "rating_change",
        "coverage_change": "coverage_change",
        "analyst_coverage": "coverage_change",
        "low_analyst_coverage": "low_analyst_coverage",
    }
    return aliases.get(value, value)


def _normalize_direction(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "supports_gap": "supports_gap",
        "support": "supports_gap",
        "supportive": "supports_gap",
        "lagging": "supports_gap",
        "stale": "supports_gap",
        "not_revised": "supports_gap",
        "no_revision": "supports_gap",
        "contradicts_gap": "contradicts_gap",
        "contradict": "contradicts_gap",
        "contradicting": "contradicts_gap",
        "downward_revision": "contradicts_gap",
        "lowered": "contradicts_gap",
        "negative_revision": "contradicts_gap",
        "neutral": "neutral",
        "mixed": "neutral",
        "upward_revision": "neutral",
        "raised": "neutral",
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
        "free_cash_flow": "free_cash_flow",
        "price_target": "price_target",
        "target_price": "price_target",
        "rating": "rating",
        "coverage": "coverage",
    }
    return aliases.get(value, value)


def _evidence_snippet(event: AnalystEstimateEvent) -> str:
    values = []
    if event.metric:
        values.append(f"metric={event.metric}")
    if event.fiscal_period:
        values.append(f"period={event.fiscal_period}")
    if event.previous_value is not None:
        values.append(f"previous={event.previous_value:g}")
    if event.current_value is not None:
        values.append(f"current={event.current_value:g}")
    if event.unit:
        values.append(f"unit={event.unit}")
    if event.analyst_firm:
        values.append(f"firm={event.analyst_firm}")
    value_text = ", ".join(values) if values else "values=NA"
    return f"{event.event_type} {event.direction}; {value_text}. {event.description}"


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _text(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


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


def _optional_int(raw: object) -> int | None:
    value = _optional_number(raw)
    if value is None:
        return None
    return int(value)


def _optional_float(raw: str) -> float | None:
    if not raw:
        return None
    return float(raw.replace(",", ""))


def _bounded_float(raw: str, *, default: float, minimum: float, maximum: float) -> float:
    value = default if not raw else float(raw)
    return max(minimum, min(maximum, value))


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:g}"


def _fmt_int(value: int | None) -> str:
    return "NA" if value is None else str(value)
