from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import urlencode

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_PUBLIC_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
EIA_ELECTRICITY_RETAIL_SALES_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"


@dataclass(frozen=True)
class MacroIndicator:
    series_id: str
    source_name: str
    metric_name: str
    category: str
    geography: str | None
    frequency: str | None
    period: str
    value: float
    unit: str | None
    source_url: str
    content_hash: str


def import_macro_indicators_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    indicators: list[MacroIndicator] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"series_id", "source_name", "metric_name", "category", "period", "value", "source_url"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Macro indicators CSV missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            indicators.append(
                _make_indicator(
                    series_id=_field(row, "series_id"),
                    source_name=_field(row, "source_name"),
                    metric_name=_field(row, "metric_name"),
                    category=_field(row, "category"),
                    geography=_field(row, "geography") or None,
                    frequency=_field(row, "frequency") or None,
                    period=_field(row, "period", "date"),
                    value=_number(_field(row, "value")),
                    unit=_field(row, "unit") or None,
                    source_url=_field(row, "source_url"),
                )
            )
    return upsert_macro_indicators(conn, indicators)


def fetch_fred_observations(
    client: HttpClient,
    *,
    api_key: str,
    series_id: str,
    metric_name: str | None = None,
    category: str = "macro",
    limit: int = 12,
    observation_start: str | None = None,
) -> list[MacroIndicator]:
    params: dict[str, str | int] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    if observation_start:
        params["observation_start"] = observation_start
    url = f"{FRED_OBSERVATIONS_URL}?{urlencode(params)}"
    payload = client.get_json(url)
    observations = _fred_observations(payload)
    source_url = _sanitize_fred_url(series_id=series_id, limit=limit, observation_start=observation_start)
    indicators: list[MacroIndicator] = []
    for row in observations:
        value = _optional_number(row.get("value"))
        period = str(row.get("date") or "").strip()
        if value is None or not period:
            continue
        indicators.append(
            _make_indicator(
                series_id=series_id.upper(),
                source_name="FRED",
                metric_name=metric_name or series_id.upper(),
                category=category,
                geography="US",
                frequency=None,
                period=period,
                value=value,
                unit=None,
                source_url=source_url,
            )
        )
    return indicators


def fetch_fred_public_observations(
    client: HttpClient,
    *,
    series_id: str,
    metric_name: str | None = None,
    category: str = "macro",
    limit: int = 12,
    observation_start: str | None = None,
) -> list[MacroIndicator]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    series_id = series_id.strip().upper()
    if not series_id:
        raise ValueError("series_id is required.")
    url = _fred_public_csv_url(series_id=series_id, observation_start=observation_start)
    csv_text = client.get_text(url, accept="text/csv,text/plain,*/*")
    rows = _fred_public_csv_rows(csv_text, series_id=series_id, observation_start=observation_start)
    source_url = _fred_public_csv_url(series_id=series_id, observation_start=observation_start)
    indicators: list[MacroIndicator] = []
    for period, value in rows[:limit]:
        indicators.append(
            _make_indicator(
                series_id=series_id,
                source_name="FRED",
                metric_name=metric_name or series_id,
                category=category,
                geography="US",
                frequency=None,
                period=period,
                value=value,
                unit=None,
                source_url=source_url,
            )
        )
    return indicators


def fetch_eia_electricity_retail_sales(
    client: HttpClient,
    *,
    api_key: str,
    limit: int = 12,
    frequency: str = "monthly",
    stateid: str | None = None,
    sectorid: str | None = None,
) -> list[MacroIndicator]:
    params: list[tuple[str, str | int]] = [
        ("api_key", api_key),
        ("frequency", frequency),
        ("data[0]", "sales"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("offset", 0),
        ("length", limit),
    ]
    if stateid:
        params.append(("facets[stateid][]", stateid))
    if sectorid:
        params.append(("facets[sectorid][]", sectorid))
    url = f"{EIA_ELECTRICITY_RETAIL_SALES_URL}?{urlencode(params)}"
    payload = client.get_json(url)
    rows = _eia_response_rows(payload)
    source_url = _sanitize_eia_url(limit=limit, frequency=frequency, stateid=stateid, sectorid=sectorid)
    indicators: list[MacroIndicator] = []
    for row in rows:
        value = _optional_number(row.get("sales"))
        period = str(row.get("period") or "").strip()
        if value is None or not period:
            continue
        geography = str(row.get("stateid") or stateid or "").strip() or None
        sector = str(row.get("sectorid") or sectorid or "").strip()
        series_id = "EIA_ELECTRICITY_RETAIL_SALES"
        if geography:
            series_id += f"_{geography.upper()}"
        if sector:
            series_id += f"_{sector.upper()}"
        indicators.append(
            _make_indicator(
                series_id=series_id,
                source_name="EIA Open Data",
                metric_name="Electricity retail sales",
                category="electricity",
                geography=geography,
                frequency=frequency,
                period=period,
                value=value,
                unit=str(row.get("sales-units") or row.get("unit") or "").strip() or None,
                source_url=source_url,
            )
        )
    return indicators


def upsert_macro_indicators(conn: sqlite3.Connection, indicators: list[MacroIndicator]) -> int:
    if not indicators:
        return 0
    fetched_at = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO macro_indicators (
            series_id, source_name, metric_name, category, geography,
            frequency, period, value, unit, source_url, fetched_at, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            source_name=excluded.source_name,
            metric_name=excluded.metric_name,
            category=excluded.category,
            geography=excluded.geography,
            frequency=excluded.frequency,
            value=excluded.value,
            unit=excluded.unit,
            source_url=excluded.source_url,
            fetched_at=excluded.fetched_at
        """,
        [
            (
                item.series_id,
                item.source_name,
                item.metric_name,
                item.category,
                item.geography,
                item.frequency,
                item.period,
                item.value,
                item.unit,
                item.source_url,
                fetched_at,
                item.content_hash,
            )
            for item in indicators
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
            confidence=excluded.confidence
        """,
        [
            (
                "Macro Indicator",
                item.source_name,
                item.source_url,
                item.period,
                fetched_at,
                item.metric_name,
                "Macro/electricity background data. This is not company-level order, revenue, or customer evidence.",
                _snippet(item),
                _confidence(item.source_name),
                None,
                "macro_context",
                f"macro:{item.content_hash}",
            )
            for item in indicators
        ],
    )
    return len(indicators)


def list_latest_macro_indicators(
    conn: sqlite3.Connection,
    *,
    limit: int = 10,
    category: str | None = None,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = ""
    if category:
        where = "WHERE category = ?"
        params.append(category)
    params.append(limit)
    return conn.execute(
        f"""
        SELECT series_id, source_name, metric_name, category, geography,
               frequency, period, value, unit, source_url, fetched_at
        FROM macro_indicators
        {where}
        ORDER BY period DESC, fetched_at DESC, series_id
        LIMIT ?
        """,
        params,
    ).fetchall()


def _make_indicator(
    *,
    series_id: str,
    source_name: str,
    metric_name: str,
    category: str,
    geography: str | None,
    frequency: str | None,
    period: str,
    value: float | None,
    unit: str | None,
    source_url: str,
) -> MacroIndicator:
    if not series_id:
        raise ValueError("Macro indicator is missing series_id.")
    if not source_name:
        raise ValueError(f"Macro indicator {series_id} is missing source_name.")
    if not metric_name:
        raise ValueError(f"Macro indicator {series_id} is missing metric_name.")
    if not category:
        raise ValueError(f"Macro indicator {series_id} is missing category.")
    if not period:
        raise ValueError(f"Macro indicator {series_id} is missing period.")
    if value is None:
        raise ValueError(f"Macro indicator {series_id} is missing numeric value.")
    if not source_url:
        raise ValueError(f"Macro indicator {series_id} is missing source_url.")
    content_hash = hashlib.sha256(
        json.dumps(
            {
                "series_id": series_id,
                "source_name": source_name,
                "metric_name": metric_name,
                "category": category,
                "geography": geography,
                "frequency": frequency,
                "period": period,
                "value": value,
                "unit": unit,
                "source_url": source_url,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return MacroIndicator(
        series_id=series_id,
        source_name=source_name,
        metric_name=metric_name,
        category=category,
        geography=geography,
        frequency=frequency,
        period=period,
        value=value,
        unit=unit,
        source_url=source_url,
        content_hash=content_hash,
    )


def _fred_observations(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return []
    return [row for row in observations if isinstance(row, dict)]


def _eia_response_rows(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    response = payload.get("response")
    if not isinstance(response, dict):
        return []
    data = response.get("data")
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _fred_public_csv_rows(
    csv_text: str,
    *,
    series_id: str,
    observation_start: str | None,
) -> list[tuple[str, float]]:
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        return []
    date_field = _first_present_field(reader.fieldnames, ("observation_date", "DATE", "date"))
    value_field = _first_present_field(reader.fieldnames, (series_id, series_id.upper(), series_id.lower()))
    if not date_field or not value_field:
        raise ValueError(f"FRED public CSV for {series_id} is missing date or value columns.")
    rows: list[tuple[str, float]] = []
    for row in reader:
        period = str(row.get(date_field) or "").strip()
        value = _optional_number(row.get(value_field))
        if not period or value is None:
            continue
        if observation_start and period < observation_start:
            continue
        rows.append((period, value))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def _first_present_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found:
            return found
    return None


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _number(raw: str) -> float | None:
    return _optional_number(raw)


def _optional_number(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip().replace(",", "")
        if not value or value == ".":
            return None
    else:
        value = raw
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_fred_url(
    *,
    series_id: str,
    limit: int,
    observation_start: str | None,
) -> str:
    params: dict[str, str | int] = {
        "series_id": series_id,
        "api_key": "***",
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    if observation_start:
        params["observation_start"] = observation_start
    return f"{FRED_OBSERVATIONS_URL}?{urlencode(params)}"


def _fred_public_csv_url(*, series_id: str, observation_start: str | None) -> str:
    params = {"id": series_id.upper()}
    if observation_start:
        params["cosd"] = observation_start
    return f"{FRED_PUBLIC_CSV_URL}?{urlencode(params)}"


def _sanitize_eia_url(
    *,
    limit: int,
    frequency: str,
    stateid: str | None,
    sectorid: str | None,
) -> str:
    params: list[tuple[str, str | int]] = [
        ("api_key", "***"),
        ("frequency", frequency),
        ("data[0]", "sales"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("offset", 0),
        ("length", limit),
    ]
    if stateid:
        params.append(("facets[stateid][]", stateid))
    if sectorid:
        params.append(("facets[sectorid][]", sectorid))
    return f"{EIA_ELECTRICITY_RETAIL_SALES_URL}?{urlencode(params)}"


def _snippet(item: MacroIndicator) -> str:
    unit = f" {item.unit}" if item.unit else ""
    geography = f", geography={item.geography}" if item.geography else ""
    frequency = f", frequency={item.frequency}" if item.frequency else ""
    return (
        f"{item.series_id} {item.metric_name}: period={item.period}, "
        f"value={item.value:g}{unit}{geography}{frequency}"
    )


def _confidence(source_name: str) -> float:
    if source_name in {"FRED", "EIA Open Data"}:
        return 0.85
    return 0.65
