from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.analysis.scoring import score_ticker, upsert_research_card
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class WatchlistRow:
    ticker: str
    company_name: str | None
    sector: str | None
    industry: str | None
    score_total: float
    status: str
    ai_relevance_score: float
    expectation_gap_score: float
    fundamental_score: float
    valuation_score: float
    catalyst_score: float
    market_confirmation_score: float
    risk_penalty: float
    ai_tag_count: int
    catalyst_count: int
    evidence_count: int
    last_reviewed_at: str


def build_watchlist(
    conn: sqlite3.Connection,
    *,
    limit: int = 200,
    min_score: float | None = None,
) -> list[WatchlistRow]:
    tickers = _candidate_tickers(conn, limit=limit)
    rows: list[WatchlistRow] = []
    now = utc_now_iso()
    for ticker in tickers:
        score = score_ticker(conn, ticker)
        upsert_research_card(conn, score)
        profile = conn.execute(
            """
            SELECT company_name, sector, industry
            FROM company_profile
            WHERE ticker = ?
            """,
            (ticker,),
        ).fetchone()
        if min_score is not None and score.score_total < min_score:
            continue
        rows.append(
            WatchlistRow(
                ticker=ticker,
                company_name=profile["company_name"] if profile else None,
                sector=profile["sector"] if profile else None,
                industry=profile["industry"] if profile else None,
                score_total=score.score_total,
                status=score.status,
                ai_relevance_score=score.ai_relevance_score,
                expectation_gap_score=score.expectation_gap_score,
                fundamental_score=score.fundamental_score,
                valuation_score=score.valuation_score,
                catalyst_score=score.catalyst_score,
                market_confirmation_score=score.market_confirmation_score,
                risk_penalty=score.risk_penalty,
                ai_tag_count=_count(conn, "ai_industry_tags", ticker),
                catalyst_count=_count(conn, "catalysts", ticker),
                evidence_count=_count(conn, "evidence_items", ticker, column="related_ticker"),
                last_reviewed_at=now,
            )
        )
    rows.sort(key=lambda row: (row.score_total, row.ai_relevance_score, row.catalyst_score), reverse=True)
    return rows


def write_watchlist_csv(rows: list[WatchlistRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WatchlistRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _candidate_tickers(conn: sqlite3.Connection, *, limit: int) -> list[str]:
    universe_count = int(
        conn.execute("SELECT COUNT(*) AS count FROM universe").fetchone()["count"] or 0
    )
    profile_fallback = (
        "UNION SELECT ticker, 13 AS priority FROM company_profile"
        if universe_count == 0
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT ticker FROM (
            SELECT ticker, 1 AS priority FROM ai_relevance_signals
            UNION
            SELECT ticker, 2 AS priority FROM ai_industry_tags
            UNION
            SELECT ticker, 3 AS priority FROM catalysts
            UNION
            SELECT ticker, 4 AS priority FROM analyst_estimate_events
            UNION
            SELECT ticker, 5 AS priority FROM valuation_snapshots
            UNION
            SELECT ticker, 6 AS priority FROM peer_valuation_comparisons
            UNION
            SELECT ticker, 7 AS priority FROM guidance_events
            UNION
            SELECT ticker, 8 AS priority FROM transcript_snippets
            UNION
            SELECT ticker, 9 AS priority FROM financial_facts
            UNION
            SELECT ticker, 10 AS priority FROM institutional_holding_events
            UNION
            SELECT related_ticker AS ticker, 0 AS priority FROM evidence_items
            WHERE related_module = 'external_research_monitor'
              AND related_ticker IS NOT NULL
              AND related_ticker != ''
            UNION
            SELECT ticker, 12 AS priority FROM universe
            WHERE is_etf = 0 AND is_preferred = 0 AND is_unit = 0 AND is_active = 1
            UNION
            SELECT ticker, 13 AS priority FROM company_profile
            WHERE ticker NOT IN (SELECT ticker FROM universe)
              AND ticker IN (
                  SELECT ticker FROM ai_relevance_signals
                  UNION SELECT ticker FROM ai_industry_tags
                  UNION SELECT ticker FROM catalysts
                  UNION SELECT ticker FROM analyst_estimate_events
                  UNION SELECT ticker FROM valuation_snapshots
                  UNION SELECT ticker FROM peer_valuation_comparisons
                  UNION SELECT ticker FROM guidance_events
                  UNION SELECT ticker FROM transcript_snippets
                  UNION SELECT ticker FROM financial_facts
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


def _count(
    conn: sqlite3.Connection,
    table: str,
    ticker: str,
    *,
    column: str = "ticker",
) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {column} = ?",
        (ticker,),
    ).fetchone()
    return int(row["count"] or 0)
