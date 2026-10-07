from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class RiskFlag:
    ticker: str
    risk_date: str | None
    source_type: str
    source_name: str
    source_url: str
    risk_type: str
    severity: str
    description: str
    confidence: float
    status: str
    content_hash: str


def import_risk_flags_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "source_url", "risk_type", "severity", "description", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Risk flags CSV missing required columns: {', '.join(sorted(missing))}")
        flags = [_flag_from_row(row, csv_path) for row in reader]
    return upsert_risk_flags(conn, flags)


def upsert_risk_flags(conn: sqlite3.Connection, flags: list[RiskFlag]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO risk_flags (
            ticker, risk_date, source_type, source_name, source_url,
            risk_type, severity, description, confidence, status,
            content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            risk_date=excluded.risk_date,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            risk_type=excluded.risk_type,
            severity=excluded.severity,
            description=excluded.description,
            confidence=excluded.confidence,
            status=excluded.status,
            updated_at=excluded.updated_at
        """,
        [
            (
                flag.ticker,
                flag.risk_date,
                flag.source_type,
                flag.source_name,
                flag.source_url,
                flag.risk_type,
                flag.severity,
                flag.description,
                flag.confidence,
                flag.status,
                flag.content_hash,
                now,
            )
            for flag in flags
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
                flag.source_type,
                flag.source_name,
                flag.source_url,
                flag.risk_date,
                now,
                flag.risk_type,
                flag.description,
                flag.description,
                flag.confidence,
                flag.ticker,
                "risk_flag",
                f"risk-flag:{flag.content_hash}",
            )
            for flag in flags
        ],
    )
    return len(flags)


def _flag_from_row(row: dict[str, str], csv_path: Path) -> RiskFlag:
    ticker = row.get("ticker", "").strip().upper()
    source_url = row.get("source_url", "").strip()
    risk_type = row.get("risk_type", "").strip().lower()
    severity = _normalize_severity(row.get("severity"))
    description = row.get("description", "").strip()
    confidence = _bounded_float(row.get("confidence"), default=0.5, minimum=0.0, maximum=1.0)
    status = _normalize_status(row.get("status"))
    risk_date = row.get("risk_date", "").strip() or row.get("signal_date", "").strip() or None
    source_type = row.get("source_type", "").strip() or "risk_flag_csv"
    source_name = row.get("source_name", "").strip() or csv_path.name
    if not ticker:
        raise ValueError("Risk flags CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Risk flags CSV row for {ticker} is missing source_url.")
    if not risk_type:
        raise ValueError(f"Risk flags CSV row for {ticker} is missing risk_type.")
    if not description:
        raise ValueError(f"Risk flags CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                risk_date or "",
                source_type,
                source_name,
                source_url,
                risk_type,
                severity,
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return RiskFlag(
        ticker=ticker,
        risk_date=risk_date,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        risk_type=risk_type,
        severity=severity,
        description=description,
        confidence=confidence,
        status=status,
        content_hash=content_hash,
    )


def _normalize_severity(raw: str | None) -> str:
    value = (raw or "medium").strip().lower()
    if value in {"low", "medium", "high", "critical"}:
        return value
    return "medium"


def _normalize_status(raw: str | None) -> str:
    value = (raw or "active").strip().lower()
    if value in {"active", "resolved", "watch"}:
        return value
    return "active"


def _bounded_float(raw: str | None, *, default: float, minimum: float, maximum: float) -> float:
    if raw is None or raw.strip() == "":
        value = default
    else:
        value = float(raw)
    return max(minimum, min(maximum, value))
