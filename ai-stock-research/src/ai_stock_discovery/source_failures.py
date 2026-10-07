from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class SourceFailureRow:
    source_name: str
    ticker: str
    endpoint: str
    failure_type: str
    status: str
    occurrences: int
    first_seen_at: str
    last_seen_at: str
    last_success_at: str
    reason: str
    source_url: str
    replacement_priority: str
    replacement_action: str
    replacement_command: str
    replacement_notes: str


def record_source_failure(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    reason: str,
    ticker: str | None = None,
    endpoint: str | None = None,
    source_url: str | None = None,
    failure_type: str | None = None,
) -> None:
    normalized_ticker = (ticker or "").strip().upper() or None
    normalized_endpoint = (endpoint or "").strip() or None
    normalized_reason = _shorten(reason.strip(), 1000)
    normalized_type = failure_type or classify_failure_type(normalized_reason)
    now = utc_now_iso()
    content_hash = _content_hash(
        source_name=source_name,
        ticker=normalized_ticker,
        endpoint=normalized_endpoint,
        failure_type=normalized_type,
        reason=normalized_reason,
    )
    conn.execute(
        """
        INSERT INTO source_failures (
            source_name, ticker, endpoint, failure_type, status, reason, source_url,
            first_seen_at, last_seen_at, occurrences, last_success_at, content_hash
        )
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, 1, NULL, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            status='open',
            reason=excluded.reason,
            source_url=excluded.source_url,
            last_seen_at=excluded.last_seen_at,
            occurrences=source_failures.occurrences + 1
        """,
        (
            source_name,
            normalized_ticker,
            normalized_endpoint,
            normalized_type,
            normalized_reason,
            source_url,
            now,
            now,
            content_hash,
        ),
    )


def mark_source_success(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    ticker: str | None = None,
    endpoint: str | None = None,
) -> int:
    normalized_ticker = (ticker or "").strip().upper() or None
    normalized_endpoint = (endpoint or "").strip() or None
    now = utc_now_iso()
    cursor = conn.execute(
        """
        UPDATE source_failures
        SET status='resolved',
            last_success_at=?,
            last_seen_at=last_seen_at
        WHERE status='open'
          AND source_name=?
          AND (ticker IS ? OR ticker = ?)
          AND (endpoint IS ? OR endpoint = ?)
        """,
        (
            now,
            source_name,
            normalized_ticker,
            normalized_ticker,
            normalized_endpoint,
            normalized_endpoint,
        ),
    )
    return int(cursor.rowcount or 0)


def classify_failure_type(reason: str) -> str:
    text = reason.lower()
    if "402" in text or "payment required" in text:
        return "http_402"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if "403" in text:
        return "http_403"
    if "404" in text:
        return "http_404"
    if "not configured" in text or "missing api" in text or "missing key" in text:
        return "missing_configuration"
    if "returned 0" in text or "no usable" in text or "no fmp" in text:
        return "empty_response"
    if "timeout" in text:
        return "timeout"
    return "source_error"


def build_source_failure_report(
    conn: sqlite3.Connection,
    *,
    status: str | None = "open",
    limit: int = 200,
) -> list[SourceFailureRow]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    where = ""
    params: list[object] = []
    if status and status != "all":
        where = "WHERE status = ?"
        params.append(status)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            source_name, ticker, endpoint, failure_type, status, occurrences,
            first_seen_at, last_seen_at, last_success_at, reason, source_url
        FROM source_failures
        {where}
        ORDER BY
            CASE status WHEN 'open' THEN 0 ELSE 1 END,
            last_seen_at DESC,
            source_name,
            ticker
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        _source_failure_row(
            source_name=str(row["source_name"]),
            ticker=str(row["ticker"] or ""),
            endpoint=str(row["endpoint"] or ""),
            failure_type=str(row["failure_type"]),
            status=str(row["status"]),
            occurrences=int(row["occurrences"] or 0),
            first_seen_at=str(row["first_seen_at"] or ""),
            last_seen_at=str(row["last_seen_at"] or ""),
            last_success_at=str(row["last_success_at"] or ""),
            reason=str(row["reason"] or ""),
            source_url=str(row["source_url"] or ""),
        )
        for row in rows
    ]


def write_source_failure_csv(rows: list[SourceFailureRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_name",
                "ticker",
                "endpoint",
                "failure_type",
                "status",
                "occurrences",
                "first_seen_at",
                "last_seen_at",
                "last_success_at",
                "reason",
                "source_url",
                "replacement_priority",
                "replacement_action",
                "replacement_command",
                "replacement_notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.source_name,
                    row.ticker,
                    row.endpoint,
                    row.failure_type,
                    row.status,
                    row.occurrences,
                    row.first_seen_at,
                    row.last_seen_at,
                    row.last_success_at,
                    row.reason,
                    row.source_url,
                    row.replacement_priority,
                    row.replacement_action,
                    row.replacement_command,
                    row.replacement_notes,
                ]
            )


def _source_failure_row(
    *,
    source_name: str,
    ticker: str,
    endpoint: str,
    failure_type: str,
    status: str,
    occurrences: int,
    first_seen_at: str,
    last_seen_at: str,
    last_success_at: str,
    reason: str,
    source_url: str,
) -> SourceFailureRow:
    replacement_priority, replacement_action, replacement_command, replacement_notes = _replacement_plan(
        source_name=source_name,
        ticker=ticker,
        endpoint=endpoint,
        failure_type=failure_type,
    )
    return SourceFailureRow(
        source_name=source_name,
        ticker=ticker,
        endpoint=endpoint,
        failure_type=failure_type,
        status=status,
        occurrences=occurrences,
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        last_success_at=last_success_at,
        reason=reason,
        source_url=source_url,
        replacement_priority=replacement_priority,
        replacement_action=replacement_action,
        replacement_command=replacement_command,
        replacement_notes=replacement_notes,
    )


def _replacement_plan(
    *,
    source_name: str,
    ticker: str,
    endpoint: str,
    failure_type: str,
) -> tuple[str, str, str, str]:
    normalized_source = source_name.lower()
    normalized_endpoint = endpoint.lower()
    normalized_type = failure_type.lower()
    ticker_arg = ticker.upper() if ticker else "TICKER"

    if "financial modeling prep" in normalized_source and normalized_endpoint == "historical-price-eod":
        return (
            "P0",
            "fetch_market_bars_from_fallback_or_local_source",
            f"python -m ai_stock_discovery.cli fetch-yahoo-market-bars --ticker {ticker_arg} --benchmark QQQ --lookback-days 90",
            "FMP EOD is entitlement-gated; Yahoo chart is prototype-only. Prefer local broker/API CSV when available, then run detect-market-anomalies.",
        )
    if "financial modeling prep" in normalized_source and normalized_endpoint in {
        "income-statement",
        "financial-statements",
        "balance-sheet-statement",
        "cash-flow-statement",
    }:
        return (
            "P0",
            "backfill_sec_companyfacts",
            f"python -m ai_stock_discovery.cli backfill-sec-companyfacts --ticker {ticker_arg}",
            "SEC companyfacts is the preferred official fallback for fundamentals; requires SEC_USER_AGENT in .env.",
        )
    if source_name == "SEC companyfacts" and normalized_type == "missing_configuration":
        return (
            "P0",
            "configure_sec_user_agent_then_backfill",
            f"python -m ai_stock_discovery.cli backfill-sec-companyfacts --ticker {ticker_arg}",
            "Set SEC_USER_AGENT in .env with a real contact identity before fetching SEC data.",
        )
    if "financial modeling prep" in normalized_source and normalized_endpoint == "quote":
        return (
            "P1",
            "fetch_quote_fallback_or_import_manual_snapshot",
            f"python -m ai_stock_discovery.cli fetch-eastmoney-quote --ticker {ticker_arg}",
            "FMP quote is entitlement-gated; Eastmoney public US quote is fallback valuation price context only. Alternatively import valuation_snapshots.csv with source URLs.",
        )
    if "financial modeling prep" in normalized_source and normalized_endpoint == "analyst-estimates":
        return (
            "P1",
            "use_source_backed_analyst_or_expectation_gap_template",
            "python -m ai_stock_discovery.cli build-source-input-templates --template analyst_estimate_events",
            "Do not infer consensus. Fill analyst estimate events only from original analyst/API/news/IR sources with URLs.",
        )
    return (
        "P2",
        "review_source_and_choose_adapter",
        "python -m ai_stock_discovery.cli build-source-input-plan --only-readiness-gaps",
        "Review the original source failure and use a source-backed local CSV/API adapter; do not generate synthetic rows.",
    )


def _content_hash(
    *,
    source_name: str,
    ticker: str | None,
    endpoint: str | None,
    failure_type: str,
    reason: str,
) -> str:
    payload = {
        "source_name": source_name,
        "ticker": ticker,
        "endpoint": endpoint,
        "failure_type": failure_type,
        "reason": reason,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
