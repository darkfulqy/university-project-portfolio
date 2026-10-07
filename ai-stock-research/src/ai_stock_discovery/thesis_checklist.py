from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.evidence_audit import EvidenceAuditRow, build_evidence_audit


@dataclass(frozen=True)
class ThesisChecklistRow:
    ticker: str
    company_name: str | None
    score_total: float
    research_status: str
    evidence_coverage_score: float
    review_priority: str
    thesis_gate: str
    ai_business_impact_check: str
    fundamentals_check: str
    expectation_gap_check: str
    catalyst_check: str
    risk_invalidation_check: str
    source_link_count: int
    missing_core_evidence: str
    next_source_actions: str
    checklist_notes: str


def build_thesis_checklist(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
) -> list[ThesisChecklistRow]:
    audit_rows = build_evidence_audit(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
    )
    rows = [_checklist_row(audit) for audit in audit_rows]
    rows.sort(
        key=lambda row: (
            _gate_rank(row.thesis_gate),
            -row.score_total,
            -row.evidence_coverage_score,
            row.ticker,
        )
    )
    return rows[:limit]


def write_thesis_checklist_csv(rows: list[ThesisChecklistRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ThesisChecklistRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _checklist_row(audit: EvidenceAuditRow) -> ThesisChecklistRow:
    ai_check = _ai_business_impact_check(audit)
    fundamentals_check = _fundamentals_check(audit)
    expectation_gap_check = _expectation_gap_check(audit)
    catalyst_check = _catalyst_check(audit)
    risk_check = _risk_invalidation_check(audit)
    checks = [
        ai_check,
        fundamentals_check,
        expectation_gap_check,
        catalyst_check,
        risk_check,
    ]
    gate = _thesis_gate(audit, checks)
    return ThesisChecklistRow(
        ticker=audit.ticker,
        company_name=audit.company_name,
        score_total=audit.score_total,
        research_status=audit.research_status,
        evidence_coverage_score=audit.evidence_coverage_score,
        review_priority=audit.review_priority,
        thesis_gate=gate,
        ai_business_impact_check=ai_check,
        fundamentals_check=fundamentals_check,
        expectation_gap_check=expectation_gap_check,
        catalyst_check=catalyst_check,
        risk_invalidation_check=risk_check,
        source_link_count=audit.source_link_count,
        missing_core_evidence=audit.missing_core_evidence,
        next_source_actions=_next_source_actions(audit),
        checklist_notes=_checklist_notes(gate),
    )


def _ai_business_impact_check(audit: EvidenceAuditRow) -> str:
    if audit.ai_signal_count and audit.ai_tag_count:
        return "present:ai_context_and_chain_tag_loaded"
    if audit.ai_signal_count:
        return "partial:ai_context_loaded_tag_missing"
    if audit.ai_tag_count:
        return "partial:ai_chain_tag_loaded_context_missing"
    return "missing:load_sec_ir_or_announcement_ai_business_context"


def _fundamentals_check(audit: EvidenceAuditRow) -> str:
    if audit.financial_period_count >= 2:
        return "present:sec_financial_trend_context_loaded"
    if audit.financial_period_count == 1:
        return "partial:single_sec_financial_period_loaded"
    if audit.guidance_event_count:
        return "partial:guidance_loaded_without_sec_financial_facts"
    return "missing:load_sec_companyfacts_or_source_backed_guidance"


def _expectation_gap_check(audit: EvidenceAuditRow) -> str:
    if audit.expectation_gap_count and audit.analyst_estimate_event_count:
        return "present:gap_and_analyst_context_loaded"
    if audit.expectation_gap_count:
        return "present:expectation_gap_signal_loaded"
    if audit.analyst_estimate_event_count:
        return "partial:analyst_event_loaded_gap_context_missing"
    if audit.peer_valuation_count:
        return "partial:peer_valuation_context_only"
    return "missing:load_expectation_gap_or_coverage_evidence"


def _catalyst_check(audit: EvidenceAuditRow) -> str:
    if audit.catalyst_count:
        return "present:dated_or_source_backed_catalyst_loaded"
    return "missing:add_dated_source_backed_catalyst"


def _risk_invalidation_check(audit: EvidenceAuditRow) -> str:
    if audit.risk_flag_count and audit.insider_transaction_count:
        return "present:risk_flags_and_insider_context_loaded"
    if audit.risk_flag_count:
        return "present:risk_flags_loaded"
    if audit.financial_period_count or audit.insider_transaction_count:
        return "partial:some_invalidation_context_loaded"
    return "missing:complete_risk_and_invalidation_review"


def _thesis_gate(audit: EvidenceAuditRow, checks: list[str]) -> str:
    if any(check.startswith("missing:") for check in checks):
        return "evidence_gap_no_company_thesis"
    if audit.source_link_count == 0:
        return "source_links_missing_no_company_thesis"
    if any(check.startswith("partial:") for check in checks):
        return "partial_thesis_needs_manual_review"
    if audit.score_total >= 80:
        return "priority_thesis_review_ready"
    if audit.score_total >= 65:
        return "watchlist_thesis_review_ready"
    return "source_chain_complete_but_low_score"


def _next_source_actions(audit: EvidenceAuditRow) -> str:
    actions: list[str] = []
    missing = {item for item in audit.missing_core_evidence.split(";") if item}
    if "ai_relevance_or_industry_tag" in missing:
        actions.append("verify_ai_business_context_from_sec_ir_or_announcements")
    if "sec_financial_facts" in missing:
        actions.append("load_sec_companyfacts")
    if "valuation_snapshot_or_peer_comparison" in missing:
        actions.append("load_valuation_snapshot_or_peer_comparison")
    if "expectation_gap_signal" in missing:
        actions.append("load_expectation_gap_or_analyst_context")
    if "catalyst_record" in missing:
        actions.append("add_source_backed_catalyst")
    if "risk_review_evidence" in missing:
        actions.append("complete_risk_flag_or_invalidation_review")
    if not actions:
        actions.append("verify_original_source_links_before_any_conclusion")
    return ";".join(actions)


def _checklist_notes(gate: str) -> str:
    if gate.endswith("no_company_thesis"):
        return "Do not form a company-level thesis until missing source categories are addressed."
    if gate == "partial_thesis_needs_manual_review":
        return "Some thesis chain links are partial; review original sources before upgrading status."
    if gate in {"priority_thesis_review_ready", "watchlist_thesis_review_ready"}:
        return "Core thesis chain is source-covered for human review; this is not investment advice."
    return "Source chain is covered, but score/status does not justify priority research by itself."


def _gate_rank(gate: str) -> int:
    order = {
        "priority_thesis_review_ready": 0,
        "watchlist_thesis_review_ready": 1,
        "partial_thesis_needs_manual_review": 2,
        "source_chain_complete_but_low_score": 3,
        "source_links_missing_no_company_thesis": 4,
        "evidence_gap_no_company_thesis": 5,
    }
    return order.get(gate, 9)
