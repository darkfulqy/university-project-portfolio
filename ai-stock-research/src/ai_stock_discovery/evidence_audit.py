from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.analysis.scoring import score_ticker


@dataclass(frozen=True)
class EvidenceAuditRow:
    ticker: str
    company_name: str | None
    score_total: float
    research_status: str
    evidence_coverage_score: float
    review_priority: str
    ai_signal_count: int
    ai_tag_count: int
    financial_period_count: int
    guidance_event_count: int
    transcript_snippet_count: int
    valuation_snapshot_count: int
    peer_valuation_count: int
    expectation_gap_count: int
    analyst_estimate_event_count: int
    catalyst_count: int
    risk_flag_count: int
    insider_transaction_count: int
    institutional_holding_event_count: int
    market_confirmation_count: int
    evidence_item_count: int
    source_link_count: int
    latest_evidence_at: str | None
    missing_core_evidence: str
    audit_notes: str


def build_evidence_audit(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
) -> list[EvidenceAuditRow]:
    selected_tickers = [ticker.upper() for ticker in tickers] if tickers else _candidate_tickers(conn, limit=limit)
    rows: list[EvidenceAuditRow] = []
    for ticker in selected_tickers[:limit]:
        score = score_ticker(conn, ticker)
        if min_score is not None and score.score_total < min_score:
            continue
        counts = _counts_for_ticker(conn, ticker)
        company_name = _company_name(conn, ticker)
        coverage, missing = _coverage(counts)
        rows.append(
            EvidenceAuditRow(
                ticker=ticker,
                company_name=company_name,
                score_total=score.score_total,
                research_status=score.status,
                evidence_coverage_score=coverage,
                review_priority=_review_priority(counts, missing),
                ai_signal_count=counts["ai_signal"],
                ai_tag_count=counts["ai_tag"],
                financial_period_count=counts["financial"],
                guidance_event_count=counts["guidance"],
                transcript_snippet_count=counts["transcript"],
                valuation_snapshot_count=counts["valuation"],
                peer_valuation_count=counts["peer_valuation"],
                expectation_gap_count=counts["expectation_gap"],
                analyst_estimate_event_count=counts["analyst_estimate"],
                catalyst_count=counts["catalyst"],
                risk_flag_count=counts["risk_flag"],
                insider_transaction_count=counts["insider_transaction"],
                institutional_holding_event_count=counts["institutional_holding"],
                market_confirmation_count=counts["market_confirmation"],
                evidence_item_count=counts["evidence_item"],
                source_link_count=counts["source_link"],
                latest_evidence_at=_latest_evidence_at(conn, ticker),
                missing_core_evidence=";".join(missing),
                audit_notes=_audit_notes(counts, missing),
            )
        )
    rows.sort(
        key=lambda row: (
            row.review_priority != "ready_for_human_review",
            -row.score_total,
            -row.evidence_coverage_score,
            row.ticker,
        )
    )
    return rows


def write_evidence_audit_csv(rows: list[EvidenceAuditRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvidenceAuditRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _candidate_tickers(conn: sqlite3.Connection, *, limit: int) -> list[str]:
    universe_count = int(
        conn.execute("SELECT COUNT(*) AS count FROM universe").fetchone()["count"] or 0
    )
    profile_fallback = (
        "UNION SELECT ticker, 18 AS priority FROM company_profile"
        if universe_count == 0
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT ticker FROM (
            SELECT ticker, 1 AS priority FROM research_cards
            UNION
            SELECT ticker, 2 AS priority FROM ai_relevance_signals
            UNION
            SELECT ticker, 3 AS priority FROM ai_industry_tags
            UNION
            SELECT ticker, 4 AS priority FROM catalysts
            UNION
            SELECT ticker, 5 AS priority FROM expectation_gap_signals
            UNION
            SELECT ticker, 6 AS priority FROM analyst_estimate_events
            UNION
            SELECT ticker, 7 AS priority FROM financial_facts
            UNION
            SELECT ticker, 8 AS priority FROM guidance_events
            UNION
            SELECT ticker, 9 AS priority FROM transcript_snippets
            UNION
            SELECT ticker, 10 AS priority FROM valuation_snapshots
            UNION
            SELECT ticker, 11 AS priority FROM peer_valuation_comparisons
            UNION
            SELECT ticker, 12 AS priority FROM market_confirmation_signals
            UNION
            SELECT ticker, 13 AS priority FROM risk_flags
            UNION
            SELECT ticker, 14 AS priority FROM insider_transactions
            UNION
            SELECT ticker, 15 AS priority FROM institutional_holding_events
            UNION
            SELECT related_ticker AS ticker, 0 AS priority FROM evidence_items
            WHERE related_module = 'external_research_monitor'
              AND related_ticker IS NOT NULL
              AND related_ticker != ''
            UNION
            SELECT ticker, 17 AS priority FROM universe
            WHERE is_etf = 0 AND is_preferred = 0 AND is_unit = 0 AND is_active = 1
            UNION
            SELECT ticker, 18 AS priority FROM company_profile
            WHERE ticker NOT IN (SELECT ticker FROM universe)
              AND ticker IN (
                  SELECT ticker FROM ai_relevance_signals
                  UNION SELECT ticker FROM ai_industry_tags
                  UNION SELECT ticker FROM catalysts
                  UNION SELECT ticker FROM expectation_gap_signals
                  UNION SELECT ticker FROM analyst_estimate_events
                  UNION SELECT ticker FROM financial_facts
                  UNION SELECT ticker FROM guidance_events
                  UNION SELECT ticker FROM transcript_snippets
                  UNION SELECT ticker FROM valuation_snapshots
                  UNION SELECT ticker FROM peer_valuation_comparisons
                  UNION SELECT ticker FROM market_confirmation_signals
                  UNION SELECT ticker FROM risk_flags
                  UNION SELECT ticker FROM insider_transactions
                  UNION SELECT ticker FROM institutional_holding_events
                  UNION SELECT related_ticker FROM evidence_items
                  WHERE related_module = 'external_research_monitor'
                    AND related_ticker IS NOT NULL
                    AND related_ticker != ''
            )
            {profile_fallback}
        ) AS candidate
        WHERE ticker IS NOT NULL AND ticker != ''
          AND NOT EXISTS (
              SELECT 1
              FROM universe_exclusions AS exclusion
              WHERE exclusion.ticker = candidate.ticker
                AND exclusion.is_active = 1
                AND exclusion.severity = 'exclude'
          )
        GROUP BY ticker
        ORDER BY MIN(priority), ticker
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row["ticker"] for row in rows]


def _counts_for_ticker(conn: sqlite3.Connection, ticker: str) -> dict[str, int]:
    return {
        "ai_signal": _count(conn, "ai_relevance_signals", ticker),
        "ai_tag": _count(conn, "ai_industry_tags", ticker),
        "financial": _count(conn, "financial_facts", ticker),
        "guidance": _count(conn, "guidance_events", ticker),
        "transcript": _count(conn, "transcript_snippets", ticker),
        "valuation": _count(conn, "valuation_snapshots", ticker),
        "peer_valuation": _count(conn, "peer_valuation_comparisons", ticker),
        "expectation_gap": _count(conn, "expectation_gap_signals", ticker),
        "analyst_estimate": _count(conn, "analyst_estimate_events", ticker),
        "catalyst": _count(conn, "catalysts", ticker),
        "risk_flag": _count(conn, "risk_flags", ticker, extra="AND status IN ('active', 'watch')"),
        "insider_transaction": _count(conn, "insider_transactions", ticker),
        "institutional_holding": _count(conn, "institutional_holding_events", ticker),
        "market_confirmation": _count(conn, "market_confirmation_signals", ticker),
        "evidence_item": _count(conn, "evidence_items", ticker, column="related_ticker"),
        "source_link": _source_link_count(conn, ticker),
    }


def _count(
    conn: sqlite3.Connection,
    table: str,
    ticker: str,
    *,
    column: str = "ticker",
    extra: str = "",
) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {column} = ? {extra}",
        (ticker,),
    ).fetchone()
    return int(row["count"] or 0)


def _source_link_count(conn: sqlite3.Connection, ticker: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT url) AS count
        FROM (
            SELECT source_url AS url FROM ai_relevance_signals WHERE ticker = ?
            UNION
            SELECT source FROM financial_facts WHERE ticker = ?
            UNION
            SELECT source_url FROM guidance_events WHERE ticker = ?
            UNION
            SELECT source_url FROM transcript_snippets WHERE ticker = ?
            UNION
            SELECT source FROM valuation_snapshots WHERE ticker = ?
            UNION
            SELECT source_url FROM peer_valuation_comparisons WHERE ticker = ?
            UNION
            SELECT source_url FROM ai_industry_tags WHERE ticker = ?
            UNION
            SELECT source_url FROM catalysts WHERE ticker = ?
            UNION
            SELECT source_url FROM expectation_gap_signals WHERE ticker = ?
            UNION
            SELECT source_url FROM analyst_estimate_events WHERE ticker = ?
            UNION
            SELECT source_url FROM risk_flags WHERE ticker = ?
            UNION
            SELECT source_path FROM market_confirmation_signals WHERE ticker = ?
            UNION
            SELECT source_url FROM insider_transactions WHERE ticker = ?
            UNION
            SELECT source_url FROM institutional_holding_events WHERE ticker = ?
            UNION
            SELECT source_url FROM short_sale_volume WHERE ticker = ?
            UNION
            SELECT url FROM evidence_items WHERE related_ticker = ?
        )
        WHERE url IS NOT NULL AND url != ''
        """,
        (ticker,) * 16,
    ).fetchone()
    return int(row["count"] or 0)


def _company_name(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        """
        SELECT company_name FROM company_profile WHERE ticker = ?
        UNION
        SELECT company_name FROM universe WHERE ticker = ?
        LIMIT 1
        """,
        (ticker, ticker),
    ).fetchone()
    return row["company_name"] if row and row["company_name"] else None


def _latest_evidence_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        """
        SELECT MAX(value) AS latest_at
        FROM (
            SELECT updated_at AS value FROM ai_relevance_signals WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM ai_industry_tags WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM financial_facts WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM guidance_events WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM transcript_snippets WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM valuation_snapshots WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM peer_valuation_comparisons WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM catalysts WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM expectation_gap_signals WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM analyst_estimate_events WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM risk_flags WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM insider_transactions WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM institutional_holding_events WHERE ticker = ?
            UNION ALL
            SELECT updated_at FROM market_confirmation_signals WHERE ticker = ?
            UNION ALL
            SELECT fetched_at FROM evidence_items WHERE related_ticker = ?
        )
        """,
        (ticker,) * 15,
    ).fetchone()
    return row["latest_at"] if row and row["latest_at"] else None


def _coverage(counts: dict[str, int]) -> tuple[float, list[str]]:
    score = 0.0
    missing: list[str] = []
    if counts["ai_signal"] or counts["ai_tag"]:
        score += 25.0
    else:
        missing.append("ai_relevance_or_industry_tag")
    if counts["financial"]:
        score += 20.0
    elif counts["guidance"]:
        score += 8.0
        missing.append("sec_financial_facts")
    else:
        missing.append("sec_financial_facts")
    if counts["valuation"] or counts["peer_valuation"]:
        score += 15.0
    else:
        missing.append("valuation_snapshot_or_peer_comparison")
    if counts["expectation_gap"] or counts["analyst_estimate"]:
        score += 15.0
    else:
        missing.append("expectation_gap_signal")
    if counts["catalyst"]:
        score += 10.0
    else:
        missing.append("catalyst_record")
    if counts["risk_flag"] or counts["financial"] or counts["insider_transaction"]:
        score += 10.0
    else:
        missing.append("risk_review_evidence")
    if counts["market_confirmation"] or counts["institutional_holding"]:
        score += 5.0
    return score, missing


def _review_priority(counts: dict[str, int], missing: list[str]) -> str:
    if not counts["ai_signal"] and not counts["ai_tag"]:
        return "needs_ai_evidence"
    if not counts["financial"]:
        return "needs_financial_facts"
    if not counts["valuation"] and not counts["peer_valuation"]:
        return "needs_valuation_snapshot"
    if not counts["expectation_gap"] and not counts["analyst_estimate"] and not counts["catalyst"]:
        return "needs_gap_or_catalyst_evidence"
    if "risk_review_evidence" in missing:
        return "needs_risk_review"
    return "ready_for_human_review"


def _audit_notes(counts: dict[str, int], missing: list[str]) -> str:
    if not missing:
        return "Core evidence categories are present; still verify original sources before any conclusion."
    return (
        "Missing or thin evidence categories: "
        + ", ".join(missing)
        + ". Audit counts are for source coverage only, not investment merit."
    )
