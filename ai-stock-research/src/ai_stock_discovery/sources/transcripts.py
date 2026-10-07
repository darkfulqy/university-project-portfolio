from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sqlite3

from ai_stock_discovery.analysis.keywords import scan_text, store_keyword_signals
from ai_stock_discovery.analysis.risk_extraction import scan_risk_text, store_extracted_risk_flags
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class TranscriptSnippet:
    ticker: str
    transcript_date: str | None
    fiscal_period: str | None
    event_type: str
    speaker: str | None
    source_type: str
    source_name: str
    source_url: str
    title: str | None
    transcript_excerpt: str
    confidence: float
    content_hash: str


@dataclass(frozen=True)
class TranscriptImportResult:
    snippets_imported: int
    ai_signals_stored: int
    risk_flags_stored: int


def import_transcript_snippets_csv(
    conn: sqlite3.Connection,
    csv_path: Path,
    *,
    scan_ai: bool = True,
    scan_risks: bool = True,
) -> TranscriptImportResult:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return TranscriptImportResult(0, 0, 0)
        required = {"ticker", "source_url", "transcript_excerpt", "confidence"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Transcript snippets CSV missing required columns: {', '.join(sorted(missing))}")
        snippets = [_snippet_from_row(row, csv_path) for row in reader]
    imported = upsert_transcript_snippets(conn, snippets)
    ai_count = 0
    risk_count = 0
    if scan_ai:
        ai_signals = []
        for snippet in snippets:
            ai_signals.extend(
                scan_text(
                    ticker=snippet.ticker,
                    text=snippet.transcript_excerpt,
                    source_type=snippet.source_type,
                    source_url=snippet.source_url,
                    signal_date=snippet.transcript_date,
                )
            )
        ai_count = store_keyword_signals(conn, ai_signals)
    if scan_risks:
        risk_flags = []
        for snippet in snippets:
            risk_flags.extend(
                scan_risk_text(
                    ticker=snippet.ticker,
                    text=snippet.transcript_excerpt,
                    source_type=snippet.source_type,
                    source_name=snippet.source_name,
                    source_url=snippet.source_url,
                    risk_date=snippet.transcript_date,
                    status="watch",
                )
            )
        risk_count = store_extracted_risk_flags(conn, risk_flags)
    return TranscriptImportResult(imported, ai_count, risk_count)


def upsert_transcript_snippets(conn: sqlite3.Connection, snippets: list[TranscriptSnippet]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO transcript_snippets (
            ticker, transcript_date, fiscal_period, event_type, speaker,
            source_type, source_name, source_url, title, transcript_excerpt,
            confidence, content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            transcript_date=excluded.transcript_date,
            fiscal_period=excluded.fiscal_period,
            event_type=excluded.event_type,
            speaker=excluded.speaker,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            title=excluded.title,
            transcript_excerpt=excluded.transcript_excerpt,
            confidence=excluded.confidence,
            updated_at=excluded.updated_at
        """,
        [
            (
                snippet.ticker,
                snippet.transcript_date,
                snippet.fiscal_period,
                snippet.event_type,
                snippet.speaker,
                snippet.source_type,
                snippet.source_name,
                snippet.source_url,
                snippet.title,
                snippet.transcript_excerpt,
                snippet.confidence,
                snippet.content_hash,
                now,
            )
            for snippet in snippets
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
                snippet.source_type,
                snippet.source_name,
                snippet.source_url,
                snippet.transcript_date,
                now,
                snippet.title or f"{snippet.ticker} transcript snippet",
                "Source-backed transcript snippet; review the original source before drawing company-level conclusions.",
                _evidence_snippet(snippet),
                snippet.confidence,
                snippet.ticker,
                "transcript_snippet",
                f"transcript-snippet:{snippet.content_hash}",
            )
            for snippet in snippets
        ],
    )
    return len(snippets)


def _snippet_from_row(row: dict[str, str], csv_path: Path) -> TranscriptSnippet:
    ticker = _field(row, "ticker").upper()
    source_url = _field(row, "source_url")
    transcript_excerpt = _compact_text(_field(row, "transcript_excerpt", "excerpt", "snippet"))
    confidence = _bounded_float(_field(row, "confidence"), default=0.5, minimum=0.0, maximum=1.0)
    transcript_date = _field(row, "transcript_date", "date", "published_at") or None
    fiscal_period = _field(row, "fiscal_period", "period") or None
    event_type = _normalize_event_type(_field(row, "event_type", "transcript_type"))
    speaker = _field(row, "speaker") or None
    source_type = _field(row, "source_type") or "transcript_snippet_csv"
    source_name = _field(row, "source_name") or csv_path.name
    title = _field(row, "title", "event_title") or None
    if not ticker:
        raise ValueError("Transcript snippets CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Transcript snippets CSV row for {ticker} is missing source_url.")
    if not transcript_excerpt:
        raise ValueError(f"Transcript snippets CSV row for {ticker} is missing transcript_excerpt.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                transcript_date or "",
                fiscal_period or "",
                event_type,
                speaker or "",
                source_type,
                source_name,
                source_url,
                title or "",
                transcript_excerpt,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return TranscriptSnippet(
        ticker=ticker,
        transcript_date=transcript_date,
        fiscal_period=fiscal_period,
        event_type=event_type,
        speaker=speaker,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        title=title,
        transcript_excerpt=transcript_excerpt,
        confidence=confidence,
        content_hash=content_hash,
    )


def _normalize_event_type(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "": "earnings_call",
        "earnings": "earnings_call",
        "earnings_call": "earnings_call",
        "conference_call": "earnings_call",
        "investor_day": "investor_day",
        "analyst_day": "investor_day",
        "conference": "conference",
        "company_presentation": "company_presentation",
        "presentation": "company_presentation",
        "fireside_chat": "conference",
        "other": "other",
    }
    return aliases.get(value, "other")


def _evidence_snippet(snippet: TranscriptSnippet) -> str:
    parts = [
        f"event={snippet.event_type}",
        f"period={snippet.fiscal_period or 'period NA'}",
    ]
    if snippet.speaker:
        parts.append(f"speaker={snippet.speaker}")
    parts.append(_truncate(snippet.transcript_excerpt, 1200))
    return "; ".join(parts)


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _compact_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 4].rstrip() + " ..."


def _bounded_float(raw: str, *, default: float, minimum: float, maximum: float) -> float:
    value = default if not raw else float(raw)
    return max(minimum, min(maximum, value))
