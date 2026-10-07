from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.review_queue import build_review_queue
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class CardBuildRow:
    ticker: str
    company_name: str | None
    score_total: float
    research_status: str
    review_bucket: str
    review_priority: str
    evidence_coverage_score: float
    source_link_count: int
    card_path: str
    generated_at: str
    build_status: str
    notes: str


def build_research_cards(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    output_dir: Path = Path("reports/cards"),
) -> list[CardBuildRow]:
    generated_at = utc_now_iso()
    review_rows = build_review_queue(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
        cards_dir=output_dir,
    )
    rows: list[CardBuildRow] = []
    for review in review_rows[:limit]:
        generate_card(conn, review.ticker, output_dir=output_dir)
        card_path = output_dir / f"{review.ticker}.md"
        rows.append(
            CardBuildRow(
                ticker=review.ticker,
                company_name=review.company_name,
                score_total=review.score_total,
                research_status=review.research_status,
                review_bucket=review.review_bucket,
                review_priority=review.review_priority,
                evidence_coverage_score=review.evidence_coverage_score,
                source_link_count=review.source_link_count,
                card_path=str(card_path),
                generated_at=generated_at,
                build_status="generated",
                notes="Research card generated from local source-backed evidence only; not investment advice.",
            )
        )
    return rows


def write_card_index_csv(rows: list[CardBuildRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CardBuildRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
