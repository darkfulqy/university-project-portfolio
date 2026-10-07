from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.review_queue import ReviewQueueRow, build_review_queue
from ai_stock_discovery.thesis_checklist import ThesisChecklistRow, build_thesis_checklist


@dataclass(frozen=True)
class ResearchPoolRow:
    ticker: str
    company_name: str | None
    sector: str | None
    industry: str | None
    score_total: float
    score_layer: str
    research_pool_status: str
    thesis_gate: str
    review_bucket: str
    review_priority: str
    evidence_coverage_score: float
    source_link_count: int
    tracking_frequency: str
    card_path: str
    next_action: str
    pool_notes: str


def build_research_pool(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    cards_dir: Path = Path("reports/cards"),
) -> list[ResearchPoolRow]:
    checklist_rows = build_thesis_checklist(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
    )
    review_rows = build_review_queue(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
        cards_dir=cards_dir,
    )
    checklist_by_ticker = {row.ticker: row for row in checklist_rows}
    review_by_ticker = {row.ticker: row for row in review_rows}
    ordered_tickers = sorted(
        checklist_by_ticker.keys() | review_by_ticker.keys(),
        key=lambda ticker: (
            min(
                _index_or_default(checklist_rows, ticker),
                _index_or_default(review_rows, ticker),
            ),
            ticker,
        ),
    )

    rows: list[ResearchPoolRow] = []
    for ticker in ordered_tickers:
        checklist = checklist_by_ticker.get(ticker)
        review = review_by_ticker.get(ticker)
        if checklist is None or review is None:
            continue
        sector, industry = _profile_context(conn, ticker)
        score_layer = _score_layer(checklist.score_total)
        status = _research_pool_status(score_layer, checklist.thesis_gate)
        rows.append(
            ResearchPoolRow(
                ticker=ticker,
                company_name=checklist.company_name or review.company_name,
                sector=sector,
                industry=industry,
                score_total=checklist.score_total,
                score_layer=score_layer,
                research_pool_status=status,
                thesis_gate=checklist.thesis_gate,
                review_bucket=review.review_bucket,
                review_priority=review.review_priority,
                evidence_coverage_score=checklist.evidence_coverage_score,
                source_link_count=checklist.source_link_count,
                tracking_frequency=review.tracking_frequency,
                card_path=review.card_path,
                next_action=_next_action(score_layer, status, checklist),
                pool_notes=_pool_notes(score_layer, status),
            )
        )
    rows.sort(
        key=lambda row: (
            _status_rank(row.research_pool_status),
            _score_layer_rank(row.score_layer),
            -row.score_total,
            -row.evidence_coverage_score,
            row.ticker,
        )
    )
    return rows[:limit]


def write_research_pool_csv(rows: list[ResearchPoolRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ResearchPoolRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _score_layer(score_total: float) -> str:
    if score_total >= 80:
        return "priority_research_pool"
    if score_total >= 65:
        return "watch_pool"
    if score_total >= 50:
        return "confirmation_waitlist"
    return "archive_or_event_watch"


def _research_pool_status(score_layer: str, thesis_gate: str) -> str:
    if thesis_gate.endswith("no_company_thesis"):
        return "source_blocked"
    if thesis_gate == "partial_thesis_needs_manual_review":
        return "manual_thesis_review"
    return f"eligible_{score_layer}"


def _next_action(score_layer: str, status: str, checklist: ThesisChecklistRow) -> str:
    if status == "source_blocked":
        return checklist.next_source_actions
    if status == "manual_thesis_review":
        return "manual_verify_partial_thesis_chain;verify_original_source_links"
    if score_layer == "priority_research_pool":
        return "refresh_research_card;maintain_weekly_tracking"
    if score_layer == "watch_pool":
        return "wait_for_financial_valuation_or_catalyst_confirmation"
    if score_layer == "confirmation_waitlist":
        return "light_monitor_for_new_source_backed_evidence"
    return "archive_until_new_source_backed_catalyst"


def _pool_notes(score_layer: str, status: str) -> str:
    if status == "source_blocked":
        return (
            "Score layer is shown for triage only; do not form or upgrade a company-level thesis "
            "until the missing source chain is complete."
        )
    if status == "manual_thesis_review":
        return (
            "Some thesis links are only partial; review original sources before changing pool status. "
            "Not investment advice."
        )
    if score_layer == "priority_research_pool":
        return "Design score layer 80+; eligible for full human research card maintenance. Not investment advice."
    if score_layer == "watch_pool":
        return "Design score layer 65-80; observe until financial, valuation, or catalyst confirmation improves."
    if score_layer == "confirmation_waitlist":
        return "Design score layer 50-65; keep lightweight monitoring only."
    return "Design score layer below 50; revisit only on new source-backed catalyst or evidence."


def _profile_context(conn: sqlite3.Connection, ticker: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        """
        SELECT sector, industry
        FROM company_profile
        WHERE ticker = ?
        """,
        (ticker,),
    ).fetchone()
    if not row:
        return None, None
    return row["sector"], row["industry"]


def _index_or_default(rows: list[ThesisChecklistRow] | list[ReviewQueueRow], ticker: str) -> int:
    for index, row in enumerate(rows):
        if row.ticker == ticker:
            return index
    return len(rows) + 1


def _status_rank(status: str) -> int:
    order = {
        "eligible_priority_research_pool": 0,
        "eligible_watch_pool": 1,
        "manual_thesis_review": 2,
        "eligible_confirmation_waitlist": 3,
        "source_blocked": 4,
        "eligible_archive_or_event_watch": 5,
    }
    return order.get(status, 9)


def _score_layer_rank(score_layer: str) -> int:
    order = {
        "priority_research_pool": 0,
        "watch_pool": 1,
        "confirmation_waitlist": 2,
        "archive_or_event_watch": 3,
    }
    return order.get(score_layer, 9)
