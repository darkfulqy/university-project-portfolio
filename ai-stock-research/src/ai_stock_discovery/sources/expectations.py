from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


LOCAL_EXPECTATION_GAP_SOURCE_NAME = "Local profile/AI tag expectation gap inference"
LOCAL_EXPECTATION_GAP_SOURCE_TYPE = "local_profile_ai_tag_expectation_gap_inference"

AI_EXPOSURE_TAGS = {
    "ai_compute",
    "automation",
    "cloud",
    "cooling",
    "cybersecurity",
    "data_center_infrastructure",
    "networking_connectivity",
    "power",
    "software_enterprise_ai",
}


@dataclass(frozen=True)
class ExpectationGapSignal:
    ticker: str
    signal_date: str | None
    source_type: str
    source_name: str
    source_url: str
    signal_type: str
    direction: str
    magnitude: float | None
    description: str
    confidence: float
    content_hash: str


def import_expectation_gap_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "source_url", "signal_type", "direction", "description", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Expectation gap CSV missing required columns: {', '.join(sorted(missing))}")
        signals = [_signal_from_row(row, csv_path) for row in reader]
    return upsert_expectation_gap_signals(conn, signals)


def infer_profile_ai_tag_expectation_gaps(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 500,
) -> list[ExpectationGapSignal]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    selected_tickers = [ticker.strip().upper() for ticker in tickers or [] if ticker.strip()]
    rows = _profile_ai_tag_rows(conn, tickers=selected_tickers or None, limit=limit)
    signals: list[ExpectationGapSignal] = []
    for row in rows:
        signal = _profile_ai_tag_signal(row)
        if signal:
            signals.append(signal)
    return signals


def upsert_expectation_gap_signals(
    conn: sqlite3.Connection,
    signals: list[ExpectationGapSignal],
) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO expectation_gap_signals (
            ticker, signal_date, source_type, source_name, source_url,
            signal_type, direction, magnitude, description, confidence,
            content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            signal_date=excluded.signal_date,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            signal_type=excluded.signal_type,
            direction=excluded.direction,
            magnitude=excluded.magnitude,
            description=excluded.description,
            confidence=excluded.confidence,
            updated_at=excluded.updated_at
        """,
        [
            (
                signal.ticker,
                signal.signal_date,
                signal.source_type,
                signal.source_name,
                signal.source_url,
                signal.signal_type,
                signal.direction,
                signal.magnitude,
                signal.description,
                signal.confidence,
                signal.content_hash,
                now,
            )
            for signal in signals
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
                signal.source_type,
                signal.source_name,
                signal.source_url,
                signal.signal_date,
                now,
                signal.signal_type,
                signal.description,
                signal.description,
                signal.confidence,
                signal.ticker,
                "expectation_gap",
                f"expectation-gap:{signal.content_hash}",
            )
            for signal in signals
        ],
    )
    return len(signals)


def _signal_from_row(row: dict[str, str], csv_path: Path) -> ExpectationGapSignal:
    ticker = row.get("ticker", "").strip().upper()
    source_url = row.get("source_url", "").strip()
    signal_type = row.get("signal_type", "").strip().lower()
    direction = _normalize_direction(row.get("direction"))
    description = row.get("description", "").strip()
    confidence = _bounded_float(row.get("confidence"), default=0.5, minimum=0.0, maximum=1.0)
    magnitude = _optional_float(row.get("magnitude"))
    signal_date = row.get("signal_date", "").strip() or None
    source_type = row.get("source_type", "").strip() or "expectation_gap_csv"
    source_name = row.get("source_name", "").strip() or csv_path.name
    if not ticker:
        raise ValueError("Expectation gap CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Expectation gap CSV row for {ticker} is missing source_url.")
    if not signal_type:
        raise ValueError(f"Expectation gap CSV row for {ticker} is missing signal_type.")
    if not description:
        raise ValueError(f"Expectation gap CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                signal_date or "",
                source_type,
                source_name,
                source_url,
                signal_type,
                direction,
                str(magnitude) if magnitude is not None else "",
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return ExpectationGapSignal(
        ticker=ticker,
        signal_date=signal_date,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        signal_type=signal_type,
        direction=direction,
        magnitude=magnitude,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _profile_ai_tag_rows(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None,
    limit: int,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = """
        WHERE COALESCE(p.ticker, '') != ''
          AND COALESCE(p.source, '') != ''
          AND COALESCE(t.tag, '') != ''
          AND COALESCE(t.source_url, '') != ''
          AND t.confidence >= 0.55
    """
    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        where += f" AND p.ticker IN ({placeholders})"
        params.extend(tickers)
    params.append(limit)
    return conn.execute(
        f"""
        SELECT
            p.ticker,
            COALESCE(p.sector, '') AS sector,
            COALESCE(p.industry, '') AS industry,
            p.source AS profile_source,
            GROUP_CONCAT(DISTINCT t.tag) AS tags,
            GROUP_CONCAT(DISTINCT t.source_url) AS tag_sources,
            MAX(t.confidence) AS max_tag_confidence,
            COUNT(DISTINCT t.tag) AS tag_count
        FROM company_profile p
        JOIN ai_industry_tags t ON t.ticker = p.ticker
        {where}
        GROUP BY p.ticker, p.sector, p.industry, p.source
        ORDER BY p.ticker
        LIMIT ?
        """,
        params,
    ).fetchall()


def _profile_ai_tag_signal(row: sqlite3.Row) -> ExpectationGapSignal | None:
    ticker = str(row["ticker"]).strip().upper()
    sector = str(row["sector"] or "").strip()
    industry = str(row["industry"] or "").strip()
    profile_source = str(row["profile_source"] or "").strip()
    raw_tags = str(row["tags"] or "")
    tags = sorted({tag.strip() for tag in raw_tags.split(",") if tag.strip() in AI_EXPOSURE_TAGS})
    if not ticker or not profile_source or not tags:
        return None
    label_reason = _legacy_label_reason(sector, industry)
    if not label_reason:
        return None
    tag_sources = _compact_sources(str(row["tag_sources"] or ""), limit=3)
    source_url = f"profile={profile_source}; ai_tag_sources={tag_sources}" if tag_sources else profile_source
    tag_text = ", ".join(tags[:6])
    if len(tags) > 6:
        tag_text += f", +{len(tags) - 6}"
    confidence = min(
        0.85,
        0.45 + min(len(tags), 5) * 0.06 + min(float(row["max_tag_confidence"] or 0), 1.0) * 0.12,
    )
    description = (
        f"Local source-backed profile labels {ticker} as sector='{sector or 'NA'}' "
        f"industry='{industry or 'NA'}' ({label_reason}), while local AI industry-chain tags include "
        f"{tag_text}. This is a legacy-market-label expectation-gap candidate for human review only; "
        "it does not prove market mispricing, customer orders, revenue impact, or investment merit."
    )
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                LOCAL_EXPECTATION_GAP_SOURCE_TYPE,
                "legacy_market_label",
                sector,
                industry,
                ",".join(tags),
                profile_source,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return ExpectationGapSignal(
        ticker=ticker,
        signal_date=utc_now_iso()[:10],
        source_type=LOCAL_EXPECTATION_GAP_SOURCE_TYPE,
        source_name=LOCAL_EXPECTATION_GAP_SOURCE_NAME,
        source_url=source_url,
        signal_type="legacy_market_label",
        direction="supports_gap",
        magnitude=float(len(tags)),
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _legacy_label_reason(sector: str, industry: str) -> str | None:
    sector_value = sector.strip().lower()
    industry_value = industry.strip().lower()
    if sector_value in {"industrials", "utilities"}:
        return f"{sector or 'sector'} profile may still be read as non-AI infrastructure"
    legacy_industry_keywords = (
        "computer hardware",
        "communication equipment",
        "electrical equipment",
        "industrial",
        "machinery",
        "renewable utilities",
    )
    if any(keyword in industry_value for keyword in legacy_industry_keywords):
        return f"{industry or 'industry'} profile may carry a legacy hardware/infrastructure label"
    return None


def _compact_sources(raw_sources: str, *, limit: int) -> str:
    sources = []
    for source in raw_sources.split(","):
        value = source.strip()
        if value and value not in sources:
            sources.append(value)
    sample = sources[:limit]
    suffix = f";+{len(sources) - limit}" if len(sources) > limit else ""
    return ";".join(sample) + suffix


def _normalize_direction(raw: str | None) -> str:
    value = (raw or "supports_gap").strip().lower()
    if value in {"supports_gap", "contradicts_gap", "neutral"}:
        return value
    return "neutral"


def _optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _bounded_float(raw: str | None, *, default: float, minimum: float, maximum: float) -> float:
    if raw is None or raw.strip() == "":
        value = default
    else:
        value = float(raw)
    return max(minimum, min(maximum, value))
