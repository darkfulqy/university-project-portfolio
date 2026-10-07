from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class SnapshotChangeRow:
    ticker: str
    company_name: str | None
    latest_snapshot_at: str
    previous_snapshot_at: str | None
    latest_run_id: int | None
    previous_run_id: int | None
    latest_score_total: float
    previous_score_total: float | None
    score_delta: float | None
    latest_research_status: str
    previous_research_status: str | None
    latest_review_bucket: str
    previous_review_bucket: str | None
    latest_review_priority: str
    previous_review_priority: str | None
    latest_evidence_coverage_score: float
    previous_evidence_coverage_score: float | None
    evidence_coverage_delta: float | None
    latest_source_link_count: int
    previous_source_link_count: int | None
    source_link_count_delta: int | None
    latest_evidence_at: str | None
    change_type: str
    attention_level: str
    change_summary: str
    report_notes: str


def build_snapshot_change_report(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score_delta: float = 0.0,
    changed_only: bool = False,
) -> list[SnapshotChangeRow]:
    snapshots = _latest_two_snapshots(conn, tickers=tickers)
    rows: list[SnapshotChangeRow] = []
    for ticker, pair in snapshots.items():
        latest = pair[0]
        previous = pair[1] if len(pair) > 1 else None
        row = _build_row(conn, latest=latest, previous=previous)
        if changed_only and row.change_type == "no_material_change":
            continue
        if (
            min_score_delta > 0
            and row.score_delta is not None
            and abs(row.score_delta) < min_score_delta
            and row.change_type in {"score_changed", "no_material_change"}
        ):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _attention_rank(row.attention_level),
            -(abs(row.score_delta) if row.score_delta is not None else 0.0),
            -abs(row.evidence_coverage_delta) if row.evidence_coverage_delta is not None else 0.0,
            row.ticker,
        )
    )
    return rows[:limit]


def write_snapshot_change_csv(rows: list[SnapshotChangeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SnapshotChangeRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _latest_two_snapshots(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None,
) -> dict[str, list[sqlite3.Row]]:
    params: list[object] = []
    where = ""
    if tickers:
        normalized = [ticker.upper() for ticker in tickers]
        placeholders = ",".join("?" for _ in normalized)
        where = f"WHERE ticker IN ({placeholders})"
        params.extend(normalized)
    rows = conn.execute(
        f"""
        SELECT ticker, snapshot_at, run_id, score_total, research_status,
               evidence_coverage_score, review_priority, review_bucket,
               tracking_frequency, source_link_count, latest_evidence_at
        FROM score_snapshots
        {where}
        ORDER BY ticker, snapshot_at DESC, id DESC
        """,
        params,
    ).fetchall()
    snapshots: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        bucket = snapshots.setdefault(row["ticker"], [])
        if len(bucket) < 2:
            bucket.append(row)
    return snapshots


def _build_row(
    conn: sqlite3.Connection,
    *,
    latest: sqlite3.Row,
    previous: sqlite3.Row | None,
) -> SnapshotChangeRow:
    score_delta = _delta(latest["score_total"], previous["score_total"] if previous else None)
    coverage_delta = _delta(
        latest["evidence_coverage_score"],
        previous["evidence_coverage_score"] if previous else None,
    )
    source_delta = _int_delta(
        latest["source_link_count"],
        previous["source_link_count"] if previous else None,
    )
    change_type, attention_level, summary = _change_classification(
        latest=latest,
        previous=previous,
        score_delta=score_delta,
        coverage_delta=coverage_delta,
        source_delta=source_delta,
    )
    return SnapshotChangeRow(
        ticker=latest["ticker"],
        company_name=_company_name(conn, latest["ticker"]),
        latest_snapshot_at=latest["snapshot_at"],
        previous_snapshot_at=previous["snapshot_at"] if previous else None,
        latest_run_id=latest["run_id"],
        previous_run_id=previous["run_id"] if previous else None,
        latest_score_total=float(latest["score_total"] or 0.0),
        previous_score_total=float(previous["score_total"]) if previous else None,
        score_delta=score_delta,
        latest_research_status=latest["research_status"],
        previous_research_status=previous["research_status"] if previous else None,
        latest_review_bucket=latest["review_bucket"],
        previous_review_bucket=previous["review_bucket"] if previous else None,
        latest_review_priority=latest["review_priority"],
        previous_review_priority=previous["review_priority"] if previous else None,
        latest_evidence_coverage_score=float(latest["evidence_coverage_score"] or 0.0),
        previous_evidence_coverage_score=(
            float(previous["evidence_coverage_score"]) if previous else None
        ),
        evidence_coverage_delta=coverage_delta,
        latest_source_link_count=int(latest["source_link_count"] or 0),
        previous_source_link_count=(
            int(previous["source_link_count"]) if previous else None
        ),
        source_link_count_delta=source_delta,
        latest_evidence_at=latest["latest_evidence_at"],
        change_type=change_type,
        attention_level=attention_level,
        change_summary=summary,
        report_notes="Snapshot change report only; verify original sources before drawing conclusions and do not treat this as investment advice.",
    )


def _change_classification(
    *,
    latest: sqlite3.Row,
    previous: sqlite3.Row | None,
    score_delta: float | None,
    coverage_delta: float | None,
    source_delta: int | None,
) -> tuple[str, str, str]:
    if previous is None:
        return (
            "new_snapshot_no_prior",
            "review_new_snapshot",
            "Only one local score snapshot exists for this ticker; no trend is inferred.",
        )

    latest_bucket = latest["review_bucket"]
    previous_bucket = previous["review_bucket"]
    if latest_bucket != previous_bucket:
        direction = _bucket_direction(previous_bucket, latest_bucket)
        return (
            "review_bucket_changed",
            f"bucket_{direction}_review",
            f"Review bucket changed from {previous_bucket} to {latest_bucket}; verify which source-backed evidence changed.",
        )

    latest_priority = latest["review_priority"]
    previous_priority = previous["review_priority"]
    if latest_priority != previous_priority:
        return (
            "review_priority_changed",
            "priority_change_review",
            f"Review priority changed from {previous_priority} to {latest_priority}; inspect evidence gaps before acting.",
        )

    if score_delta is not None and abs(score_delta) >= 10.0:
        direction = "improved" if score_delta > 0 else "declined"
        return (
            "score_changed",
            f"score_{direction}_review",
            f"Score changed by {score_delta:g} points; verify component evidence and source links.",
        )

    if coverage_delta is not None and abs(coverage_delta) >= 20.0:
        direction = "improved" if coverage_delta > 0 else "declined"
        return (
            "evidence_coverage_changed",
            f"evidence_coverage_{direction}_review",
            f"Evidence coverage changed by {coverage_delta:g} points; review source inventory and missing categories.",
        )

    if source_delta is not None and source_delta > 0:
        return (
            "source_links_increased",
            "new_sources_review",
            f"Source link count increased by {source_delta}; inspect new links before upgrading conviction.",
        )

    if source_delta is not None and source_delta < 0:
        return (
            "source_links_decreased",
            "source_inventory_declined_review",
            f"Source link count decreased by {abs(source_delta)}; review whether evidence was removed or replaced.",
        )

    return (
        "no_material_change",
        "no_material_change",
        "Latest two local score snapshots show no material score, bucket, priority, coverage, or source-link change.",
    )


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


def _delta(latest: object, previous: object | None) -> float | None:
    if previous is None:
        return None
    return round(float(latest or 0.0) - float(previous or 0.0), 2)


def _int_delta(latest: object, previous: object | None) -> int | None:
    if previous is None:
        return None
    return int(latest or 0) - int(previous or 0)


def _bucket_direction(previous_bucket: str, latest_bucket: str) -> str:
    previous_rank = _bucket_rank(previous_bucket)
    latest_rank = _bucket_rank(latest_bucket)
    if latest_rank < previous_rank:
        return "upgrade"
    if latest_rank > previous_rank:
        return "downgrade"
    return "change"


def _bucket_rank(bucket: str) -> int:
    order = {
        "priority_research": 0,
        "watch_observe": 1,
        "light_monitor": 2,
        "evidence_gap_review": 3,
        "archive_or_event_watch": 4,
    }
    return order.get(bucket, 9)


def _attention_rank(attention_level: str) -> int:
    order = {
        "bucket_upgrade_review": 0,
        "bucket_downgrade_review": 1,
        "priority_change_review": 2,
        "score_improved_review": 3,
        "score_declined_review": 4,
        "evidence_coverage_improved_review": 5,
        "evidence_coverage_declined_review": 6,
        "new_sources_review": 7,
        "source_inventory_declined_review": 8,
        "review_new_snapshot": 9,
        "no_material_change": 10,
    }
    return order.get(attention_level, 99)
