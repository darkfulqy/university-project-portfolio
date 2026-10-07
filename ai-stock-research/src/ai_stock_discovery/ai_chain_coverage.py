from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.research_pool import ResearchPoolRow, build_research_pool


CHAIN_CATEGORY_TAGS: dict[str, tuple[str, ...]] = {
    "ai_compute": ("ai_compute",),
    "semiconductors": ("semiconductors",),
    "semiconductor_equipment": ("semiconductor_equipment",),
    "materials": ("materials",),
    "data_center_infrastructure": ("data_center_infrastructure",),
    "power": ("power",),
    "cooling": ("cooling",),
    "networking_connectivity": ("networking_connectivity",),
    "cloud": ("cloud",),
    "software_enterprise_ai": ("software_enterprise_ai",),
    "cybersecurity": ("cybersecurity",),
    "automation": ("automation",),
    "vertical_applications": ("vertical_applications",),
}
UNTAGGED_CATEGORY = "unmapped_ai_chain"


@dataclass(frozen=True)
class AiChainCoverageRow:
    chain_category: str
    mapped_tags: str
    candidate_count: int
    source_backed_tag_count: int
    priority_research_count: int
    watch_pool_count: int
    confirmation_waitlist_count: int
    archive_or_event_watch_count: int
    source_blocked_count: int
    manual_review_count: int
    eligible_count: int
    avg_score: float
    max_score: float
    top_tickers: str
    source_blocked_tickers: str
    next_action: str
    coverage_notes: str


def build_ai_chain_coverage(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    top_limit: int = 8,
) -> list[AiChainCoverageRow]:
    pool_rows = build_research_pool(
        conn,
        tickers=tickers,
        limit=limit,
        min_score=min_score,
    )
    tag_map = _tag_map(conn, [row.ticker for row in pool_rows])
    rows: list[AiChainCoverageRow] = []
    for category, mapped_tags in CHAIN_CATEGORY_TAGS.items():
        matched = [
            row
            for row in pool_rows
            if set(mapped_tags).intersection({tag for tag, _source_count in tag_map.get(row.ticker, [])})
        ]
        rows.append(_coverage_row(category, mapped_tags, matched, tag_map, top_limit=top_limit))
    untagged = [row for row in pool_rows if not tag_map.get(row.ticker)]
    rows.append(_coverage_row(UNTAGGED_CATEGORY, (), untagged, tag_map, top_limit=top_limit))
    rows.sort(key=lambda row: (_category_rank(row.chain_category), -row.candidate_count, row.chain_category))
    return rows


def write_ai_chain_coverage_csv(rows: list[AiChainCoverageRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AiChainCoverageRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _coverage_row(
    category: str,
    mapped_tags: tuple[str, ...],
    candidates: list[ResearchPoolRow],
    tag_map: dict[str, list[tuple[str, int]]],
    *,
    top_limit: int,
) -> AiChainCoverageRow:
    sorted_candidates = sorted(candidates, key=lambda row: (-row.score_total, row.ticker))
    scores = [row.score_total for row in candidates]
    source_backed_tag_count = _source_backed_tag_count(candidates, mapped_tags, tag_map)
    source_blocked = [row for row in candidates if row.research_pool_status == "source_blocked"]
    manual_review = [row for row in candidates if row.research_pool_status == "manual_thesis_review"]
    eligible = [row for row in candidates if row.research_pool_status.startswith("eligible_")]
    return AiChainCoverageRow(
        chain_category=category,
        mapped_tags=";".join(mapped_tags),
        candidate_count=len(candidates),
        source_backed_tag_count=source_backed_tag_count,
        priority_research_count=sum(1 for row in candidates if row.score_layer == "priority_research_pool"),
        watch_pool_count=sum(1 for row in candidates if row.score_layer == "watch_pool"),
        confirmation_waitlist_count=sum(1 for row in candidates if row.score_layer == "confirmation_waitlist"),
        archive_or_event_watch_count=sum(1 for row in candidates if row.score_layer == "archive_or_event_watch"),
        source_blocked_count=len(source_blocked),
        manual_review_count=len(manual_review),
        eligible_count=len(eligible),
        avg_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
        max_score=max(scores) if scores else 0.0,
        top_tickers=";".join(row.ticker for row in sorted_candidates[:top_limit]),
        source_blocked_tickers=";".join(row.ticker for row in source_blocked[:top_limit]),
        next_action=_next_action(category, candidates, source_blocked),
        coverage_notes=_coverage_notes(category, candidates, source_blocked),
    )


def _source_backed_tag_count(
    candidates: list[ResearchPoolRow],
    mapped_tags: tuple[str, ...],
    tag_map: dict[str, list[tuple[str, int]]],
) -> int:
    count = 0
    mapped = set(mapped_tags)
    for row in candidates:
        tags = tag_map.get(row.ticker, [])
        if mapped:
            count += sum(1 for tag, source_count in tags if tag in mapped and source_count > 0)
        else:
            count += sum(1 for _tag, source_count in tags if source_count > 0)
    return count


def _tag_map(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, list[tuple[str, int]]]:
    if not tickers:
        return {}
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"""
        SELECT ticker, tag, COUNT(DISTINCT source_url) AS source_count
        FROM ai_industry_tags
        WHERE ticker IN ({placeholders})
        GROUP BY ticker, tag
        ORDER BY ticker, tag
        """,
        tuple(tickers),
    ).fetchall()
    tag_map: dict[str, list[tuple[str, int]]] = {}
    for row in rows:
        tag_map.setdefault(row["ticker"], []).append((row["tag"], int(row["source_count"] or 0)))
    return tag_map


def _next_action(
    category: str,
    candidates: list[ResearchPoolRow],
    source_blocked: list[ResearchPoolRow],
) -> str:
    if category == UNTAGGED_CATEGORY:
        return "tag_ai_chain_or_import_manual_source_backed_tags"
    if not candidates:
        return "seed_source_backed_candidates_for_chain_category"
    if source_blocked:
        return "complete_thesis_source_chain_for_tagged_candidates"
    return "review_top_tickers_and_refresh_source_links"


def _coverage_notes(
    category: str,
    candidates: list[ResearchPoolRow],
    source_blocked: list[ResearchPoolRow],
) -> str:
    if category == UNTAGGED_CATEGORY:
        return "Candidates without AI industry-chain tags; tag source-backed exposure before interpreting chain coverage."
    if not candidates:
        return "No local candidates are mapped to this chain category yet; this is a coverage gap, not a market conclusion."
    if source_blocked:
        return "Some tagged candidates are source-blocked by thesis gates; complete SEC/IR/API/news evidence before thesis review."
    return "Coverage is based only on local AI industry-chain tags and research-pool state; verify original sources before conclusions."


def _category_rank(category: str) -> int:
    if category == UNTAGGED_CATEGORY:
        return len(CHAIN_CATEGORY_TAGS) + 1
    for index, key in enumerate(CHAIN_CATEGORY_TAGS):
        if key == category:
            return index
    return len(CHAIN_CATEGORY_TAGS)
