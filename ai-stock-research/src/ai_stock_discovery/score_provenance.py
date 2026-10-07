from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.analysis.scoring import ScoreResult, score_ticker
from ai_stock_discovery.research_pool import build_research_pool


@dataclass(frozen=True)
class ScoreProvenanceRow:
    ticker: str
    company_name: str | None
    score_total: float
    research_status: str
    score_trace_status: str
    thesis_gate: str
    research_pool_status: str
    score_layer: str
    evidence_coverage_score: float
    source_link_count: int
    ai_relevance_score: float
    ai_trace_status: str
    fundamental_score: float
    fundamental_trace_status: str
    valuation_score: float
    valuation_trace_status: str
    expectation_gap_score: float
    expectation_gap_trace_status: str
    catalyst_score: float
    catalyst_trace_status: str
    market_confirmation_score: float
    market_confirmation_trace_status: str
    risk_penalty: float
    risk_trace_status: str
    source_modules: str
    score_source_links: str
    score_notes: str
    provenance_notes: str


def build_score_provenance(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    source_link_limit: int = 12,
) -> list[ScoreProvenanceRow]:
    pool_rows = build_research_pool(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
    )
    rows: list[ScoreProvenanceRow] = []
    for pool in pool_rows:
        score = score_ticker(conn, pool.ticker)
        component_links = _component_source_links(conn, pool.ticker)
        component_statuses = _component_trace_statuses(score, component_links)
        score_links = _ordered_unique(
            link
            for component in (
                "ai",
                "fundamental",
                "valuation",
                "expectation_gap",
                "catalyst",
                "market_confirmation",
                "risk",
            )
            for link in component_links[component]
        )
        trace_status = _score_trace_status(pool.thesis_gate, pool.source_link_count, component_statuses, score)
        rows.append(
            ScoreProvenanceRow(
                ticker=pool.ticker,
                company_name=pool.company_name,
                score_total=score.score_total,
                research_status=score.status,
                score_trace_status=trace_status,
                thesis_gate=pool.thesis_gate,
                research_pool_status=pool.research_pool_status,
                score_layer=pool.score_layer,
                evidence_coverage_score=pool.evidence_coverage_score,
                source_link_count=pool.source_link_count,
                ai_relevance_score=score.ai_relevance_score,
                ai_trace_status=component_statuses["ai"],
                fundamental_score=score.fundamental_score,
                fundamental_trace_status=component_statuses["fundamental"],
                valuation_score=score.valuation_score,
                valuation_trace_status=component_statuses["valuation"],
                expectation_gap_score=score.expectation_gap_score,
                expectation_gap_trace_status=component_statuses["expectation_gap"],
                catalyst_score=score.catalyst_score,
                catalyst_trace_status=component_statuses["catalyst"],
                market_confirmation_score=score.market_confirmation_score,
                market_confirmation_trace_status=component_statuses["market_confirmation"],
                risk_penalty=score.risk_penalty,
                risk_trace_status=component_statuses["risk"],
                source_modules=_source_modules(component_links),
                score_source_links=";".join(score_links[:source_link_limit]),
                score_notes=" | ".join(score.notes),
                provenance_notes=_provenance_notes(trace_status),
            )
        )
    rows.sort(
        key=lambda row: (
            _trace_rank(row.score_trace_status),
            -row.score_total,
            -row.evidence_coverage_score,
            row.ticker,
        )
    )
    return rows[:limit]


def write_score_provenance_csv(rows: list[ScoreProvenanceRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ScoreProvenanceRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _component_trace_statuses(
    score: ScoreResult,
    links: dict[str, list[str]],
) -> dict[str, str]:
    return {
        "ai": _positive_component_status(score.ai_relevance_score, links["ai"]),
        "fundamental": _positive_component_status(score.fundamental_score, links["fundamental"]),
        "valuation": _positive_component_status(score.valuation_score, links["valuation"]),
        "expectation_gap": _positive_component_status(score.expectation_gap_score, links["expectation_gap"]),
        "catalyst": _positive_component_status(score.catalyst_score, links["catalyst"]),
        "market_confirmation": _positive_component_status(score.market_confirmation_score, links["market_confirmation"]),
        "risk": _risk_component_status(score.risk_penalty, links["risk"]),
    }


def _positive_component_status(score_value: float, links: list[str]) -> str:
    if score_value > 0 and links:
        return "traced_score"
    if score_value > 0:
        return "score_without_source_link_review"
    if links:
        return "zero_score_with_sources"
    return "zero_score_no_sources"


def _risk_component_status(risk_penalty: float, links: list[str]) -> str:
    if risk_penalty < 0 and links:
        return "traced_penalty"
    if risk_penalty < 0:
        return "penalty_without_source_link_review"
    if links:
        return "no_penalty_with_sources"
    return "no_penalty_no_sources"


def _score_trace_status(
    thesis_gate: str,
    source_link_count: int,
    component_statuses: dict[str, str],
    score: ScoreResult,
) -> str:
    if source_link_count == 0 and score.score_total > 0:
        return "score_without_any_source_link_blocked"
    if any(status.endswith("without_source_link_review") for status in component_statuses.values()):
        return "component_source_link_gap_review"
    if source_link_count == 0:
        return "no_source_links_no_company_conclusion"
    if thesis_gate.endswith("no_company_thesis"):
        return "thesis_source_gap"
    if thesis_gate == "partial_thesis_needs_manual_review":
        return "partial_source_chain_manual_review"
    return "source_backed_for_human_review"


def _provenance_notes(trace_status: str) -> str:
    if trace_status == "source_backed_for_human_review":
        return "Score components have local source links and thesis gate permits human review; not investment advice."
    if trace_status == "partial_source_chain_manual_review":
        return "Some source-chain links are partial; review original sources before relying on the score."
    if trace_status == "thesis_source_gap":
        return "Score has some source context, but core thesis source categories are still missing."
    if trace_status == "component_source_link_gap_review":
        return "At least one non-zero score component lacks an expected source link and needs review."
    if trace_status == "score_without_any_source_link_blocked":
        return "Non-zero score exists without any source links; block company-level conclusions until source links are fixed."
    return "No source links are loaded; do not form company-level conclusions."


def _source_modules(component_links: dict[str, list[str]]) -> str:
    return ";".join(f"{module}={len(links)}" for module, links in component_links.items())


def _component_source_links(conn: sqlite3.Connection, ticker: str) -> dict[str, list[str]]:
    return {
        "ai": _links(
            conn,
            ticker,
            [
                ("ai_relevance_signals", "source_url", ""),
                ("ai_industry_tags", "source_url", ""),
            ],
        ),
        "fundamental": _links(
            conn,
            ticker,
            [
                ("financial_facts", "source", ""),
                ("guidance_events", "source_url", ""),
            ],
        ),
        "valuation": _links(
            conn,
            ticker,
            [
                ("valuation_snapshots", "source", ""),
                ("peer_valuation_comparisons", "source_url", ""),
            ],
        ),
        "expectation_gap": _links(
            conn,
            ticker,
            [
                ("expectation_gap_signals", "source_url", ""),
                ("analyst_estimate_events", "source_url", ""),
            ],
        ),
        "catalyst": _links(conn, ticker, [("catalysts", "source_url", "")]),
        "market_confirmation": _links(
            conn,
            ticker,
            [
                ("market_confirmation_signals", "source_path", ""),
                ("institutional_holding_events", "source_url", ""),
            ],
        ),
        "risk": _links(
            conn,
            ticker,
            [
                ("risk_flags", "source_url", "AND status IN ('active', 'watch')"),
                ("financial_facts", "source", ""),
            ],
        ),
    }


def _links(
    conn: sqlite3.Connection,
    ticker: str,
    sources: list[tuple[str, str, str]],
) -> list[str]:
    values: list[str] = []
    for table, column, extra in sources:
        rows = conn.execute(
            f"""
            SELECT DISTINCT {column} AS source_link
            FROM {table}
            WHERE ticker = ?
              AND {column} IS NOT NULL
              AND {column} != ''
              {extra}
            ORDER BY {column}
            """,
            (ticker,),
        ).fetchall()
        values.extend(str(row["source_link"]) for row in rows if row["source_link"])
    return _ordered_unique(values)


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _trace_rank(trace_status: str) -> int:
    order = {
        "score_without_any_source_link_blocked": 0,
        "component_source_link_gap_review": 1,
        "thesis_source_gap": 2,
        "partial_source_chain_manual_review": 3,
        "no_source_links_no_company_conclusion": 4,
        "source_backed_for_human_review": 5,
    }
    return order.get(trace_status, 9)
