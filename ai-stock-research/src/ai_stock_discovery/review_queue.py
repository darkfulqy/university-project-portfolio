from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.evidence_audit import build_evidence_audit


@dataclass(frozen=True)
class ReviewQueueRow:
    ticker: str
    company_name: str | None
    score_total: float
    research_status: str
    evidence_coverage_score: float
    review_priority: str
    review_bucket: str
    tracking_frequency: str
    next_review_due_days: int
    required_review_checks: str
    invalidating_conditions: str
    source_link_count: int
    latest_evidence_at: str | None
    card_path: str
    queue_notes: str


def build_review_queue(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    cards_dir: Path = Path("reports/cards"),
) -> list[ReviewQueueRow]:
    audit_rows = build_evidence_audit(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
    )
    rows: list[ReviewQueueRow] = []
    for audit in audit_rows:
        score = score_ticker(conn, audit.ticker)
        bucket = _review_bucket(score.score_total, audit.review_priority)
        frequency, due_days = _tracking_frequency(bucket)
        rows.append(
            ReviewQueueRow(
                ticker=audit.ticker,
                company_name=audit.company_name,
                score_total=score.score_total,
                research_status=score.status,
                evidence_coverage_score=audit.evidence_coverage_score,
                review_priority=audit.review_priority,
                review_bucket=bucket,
                tracking_frequency=frequency,
                next_review_due_days=due_days,
                required_review_checks=_required_review_checks(audit.missing_core_evidence),
                invalidating_conditions=score.invalidating_conditions,
                source_link_count=audit.source_link_count,
                latest_evidence_at=audit.latest_evidence_at,
                card_path=str(cards_dir / f"{audit.ticker}.md"),
                queue_notes=_queue_notes(bucket, audit.review_priority),
            )
        )
    rows.sort(
        key=lambda row: (
            _bucket_rank(row.review_bucket),
            row.review_priority != "ready_for_human_review",
            -row.score_total,
            -row.evidence_coverage_score,
            row.ticker,
        )
    )
    return rows[:limit]


def write_review_queue_csv(rows: list[ReviewQueueRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReviewQueueRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _review_bucket(score_total: float, review_priority: str) -> str:
    if review_priority != "ready_for_human_review":
        return "evidence_gap_review"
    if score_total >= 80:
        return "priority_research"
    if score_total >= 65:
        return "watch_observe"
    if score_total >= 50:
        return "light_monitor"
    return "archive_or_event_watch"


def _tracking_frequency(bucket: str) -> tuple[str, int]:
    if bucket == "priority_research":
        return "weekly", 7
    if bucket == "watch_observe":
        return "biweekly", 14
    if bucket == "light_monitor":
        return "monthly", 30
    if bucket == "evidence_gap_review":
        return "when_missing_sources_are_available", 21
    return "quarterly_or_on_new_catalyst", 90


def _required_review_checks(missing_core_evidence: str) -> str:
    missing = [item for item in missing_core_evidence.split(";") if item]
    checks = ["verify_original_source_links"]
    mapping = {
        "ai_relevance_or_industry_tag": "verify_ai_relevance_maps_to_business_impact",
        "sec_financial_facts": "load_or_refresh_sec_companyfacts_financials",
        "valuation_snapshot_or_peer_comparison": "load_valuation_snapshot_or_peer_comparison",
        "expectation_gap_signal": "load_expectation_gap_or_analyst_coverage_evidence",
        "catalyst_record": "add_dated_source_backed_catalyst",
        "risk_review_evidence": "complete_risk_review",
    }
    checks.extend(mapping[item] for item in missing if item in mapping)
    if len(checks) == 1:
        checks.extend(
            [
                "review_thesis_against_ai_business_impact",
                "review_valuation_catalysts_and_risks",
                "review_invalidating_conditions",
            ]
        )
    return ";".join(checks)


def _queue_notes(bucket: str, review_priority: str) -> str:
    if bucket == "evidence_gap_review":
        return (
            f"Evidence gap queue item: {review_priority}. Do not upgrade conviction "
            "until missing source categories are addressed."
        )
    if bucket == "priority_research":
        return "High-score item ready for human review; verify original sources before any conclusion."
    if bucket == "watch_observe":
        return "Observation item; wait for stronger financial, valuation, or catalyst confirmation."
    if bucket == "light_monitor":
        return "Light monitoring item; keep only if new source-backed evidence appears."
    return "Archive or event-watch item; revisit only on new source-backed catalyst or evidence."


def _bucket_rank(bucket: str) -> int:
    order = {
        "priority_research": 0,
        "watch_observe": 1,
        "light_monitor": 2,
        "evidence_gap_review": 3,
        "archive_or_event_watch": 4,
    }
    return order.get(bucket, 9)
