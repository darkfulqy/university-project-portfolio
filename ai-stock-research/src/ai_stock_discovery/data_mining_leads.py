from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.refresh_plan import RefreshPlanRow, build_refresh_plan
from ai_stock_discovery.score_provenance import ScoreProvenanceRow, build_score_provenance
from ai_stock_discovery.source_failures import SourceFailureRow, build_source_failure_report


FREE_SOURCE_ACTIONS = {"refresh_short_sale_context", "search_news_for_catalysts"}


@dataclass(frozen=True)
class DataMiningLeadRow:
    ticker: str
    company_name: str | None
    score_total: float
    lead_type: str
    mining_priority: str
    research_pool_status: str
    score_trace_status: str
    thesis_gate: str
    evidence_coverage_score: float
    source_link_count: int
    evidence_modules: str
    latest_evidence_at: str | None
    open_source_failures: int
    source_blockers: str
    refresh_actions: str
    recommended_commands: str
    lead_notes: str


def build_data_mining_leads(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    command_limit: int = 5,
) -> list[DataMiningLeadRow]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if command_limit <= 0:
        raise ValueError("command_limit must be positive.")

    provenance_rows = build_score_provenance(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
    )
    ticker_set = {row.ticker for row in provenance_rows}
    refresh_rows = _refresh_rows_by_ticker(
        build_refresh_plan(
            conn,
            tickers=tickers,
            limit=max(limit * 8, limit),
            min_score=min_score,
        )
    )
    source_failures = _source_failures_by_ticker(conn, ticker_set=ticker_set)

    rows: list[DataMiningLeadRow] = []
    for provenance in provenance_rows:
        failures = source_failures.get(provenance.ticker, [])
        actions = refresh_rows.get(provenance.ticker, [])
        evidence_modules, latest_evidence_at = _evidence_context(conn, provenance.ticker)
        lead_type = _lead_type(provenance, failures, actions)
        priority = _mining_priority(lead_type, failures)
        commands = _recommended_commands(actions, failures, command_limit=command_limit)
        rows.append(
            DataMiningLeadRow(
                ticker=provenance.ticker,
                company_name=provenance.company_name,
                score_total=provenance.score_total,
                lead_type=lead_type,
                mining_priority=priority,
                research_pool_status=provenance.research_pool_status,
                score_trace_status=provenance.score_trace_status,
                thesis_gate=provenance.thesis_gate,
                evidence_coverage_score=provenance.evidence_coverage_score,
                source_link_count=provenance.source_link_count,
                evidence_modules=evidence_modules,
                latest_evidence_at=latest_evidence_at,
                open_source_failures=len(failures),
                source_blockers=_source_blockers(failures),
                refresh_actions=_refresh_actions(actions),
                recommended_commands=commands,
                lead_notes=_lead_notes(lead_type),
            )
        )
    rows.sort(
        key=lambda row: (
            _priority_rank(row.mining_priority),
            -row.score_total,
            -row.evidence_coverage_score,
            -row.source_link_count,
            row.ticker,
        )
    )
    return rows[:limit]


def write_data_mining_leads_csv(rows: list[DataMiningLeadRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DataMiningLeadRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _refresh_rows_by_ticker(rows: list[RefreshPlanRow]) -> dict[str, list[RefreshPlanRow]]:
    grouped: dict[str, list[RefreshPlanRow]] = {}
    for row in rows:
        grouped.setdefault(row.ticker, []).append(row)
    return grouped


def _source_failures_by_ticker(
    conn: sqlite3.Connection,
    *,
    ticker_set: set[str],
) -> dict[str, list[SourceFailureRow]]:
    rows = build_source_failure_report(conn, status="open", limit=10000)
    grouped: dict[str, list[SourceFailureRow]] = {}
    for row in rows:
        ticker = row.ticker.upper()
        if not ticker or (ticker_set and ticker not in ticker_set):
            continue
        grouped.setdefault(ticker, []).append(row)
    return grouped


def _evidence_context(conn: sqlite3.Connection, ticker: str) -> tuple[str, str | None]:
    module_rows = conn.execute(
        """
        SELECT related_module, COUNT(*) AS count
        FROM evidence_items
        WHERE related_ticker = ?
        GROUP BY related_module
        ORDER BY related_module
        """,
        (ticker,),
    ).fetchall()
    modules = ";".join(f"{row['related_module']}={row['count']}" for row in module_rows)
    latest = conn.execute(
        """
        SELECT MAX(COALESCE(published_at, fetched_at)) AS latest_evidence_at
        FROM evidence_items
        WHERE related_ticker = ?
        """,
        (ticker,),
    ).fetchone()
    return modules, latest["latest_evidence_at"] if latest else None


def _lead_type(
    provenance: ScoreProvenanceRow,
    failures: list[SourceFailureRow],
    refresh_rows: list[RefreshPlanRow],
) -> str:
    if any(failure.replacement_priority == "P0" for failure in failures):
        return "p0_source_failure_recovery"
    if any(row.action_type in FREE_SOURCE_ACTIONS for row in refresh_rows):
        return "free_public_source_mining"
    if provenance.research_pool_status == "source_blocked" or provenance.thesis_gate.endswith("no_company_thesis"):
        return "source_chain_gap"
    if failures:
        return "source_failure_review"
    if provenance.score_trace_status in {"component_source_link_gap_review", "thesis_source_gap"}:
        return "source_chain_gap"
    if provenance.source_link_count > 0:
        return "source_backed_review_lead"
    return "monitor_only"


def _mining_priority(lead_type: str, failures: list[SourceFailureRow]) -> str:
    if lead_type == "p0_source_failure_recovery":
        return "P0"
    if lead_type == "free_public_source_mining":
        return "P1"
    if lead_type == "source_backed_review_lead":
        return "P2"
    if failures or lead_type in {"source_chain_gap", "source_failure_review"}:
        return "P3"
    return "P4"


def _source_blockers(failures: list[SourceFailureRow]) -> str:
    return ";".join(
        f"{failure.replacement_priority}:{failure.source_name}:{failure.endpoint}:{failure.failure_type}"
        for failure in failures
    )


def _refresh_actions(rows: list[RefreshPlanRow]) -> str:
    return ";".join(_ordered_unique(f"{row.action_type}:{row.source_name}" for row in rows))


def _recommended_commands(
    refresh_rows: list[RefreshPlanRow],
    failures: list[SourceFailureRow],
    *,
    command_limit: int,
) -> str:
    p0_failure_commands = [
        failure.replacement_command
        for failure in failures
        if failure.replacement_priority == "P0" and failure.replacement_command
    ]
    other_failure_commands = [
        failure.replacement_command
        for failure in failures
        if failure.replacement_priority != "P0" and failure.replacement_command
    ]
    commands = _ordered_unique(
        [
            *p0_failure_commands,
            *(row.recommended_command for row in refresh_rows if row.recommended_command),
            *other_failure_commands,
        ]
    )
    return "; ".join(commands[:command_limit])


def _lead_notes(lead_type: str) -> str:
    if lead_type == "p0_source_failure_recovery":
        return "Primary source is blocked; run the recommended replacement only with real source access and no synthetic rows."
    if lead_type == "free_public_source_mining":
        return "Run free/public source refresh to add review context; news metadata and FINRA short-sale volume are not company conclusions."
    if lead_type == "source_backed_review_lead":
        return "Local source links support human review; verify original sources before any company-level thesis."
    if lead_type == "source_chain_gap":
        return "Evidence chain is incomplete; keep the ticker in data-mining triage until missing source categories are filled."
    if lead_type == "source_failure_review":
        return "Open source failures need adapter, key, or local-source replacement review."
    return "Monitor only; revisit when new source-backed evidence appears."


def _priority_rank(priority: str) -> int:
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
    return order.get(priority, 9)


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
