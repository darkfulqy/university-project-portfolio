from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3

from ai_stock_discovery.review_queue import build_review_queue
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class ScoreSnapshotResult:
    snapshots_written: int
    tickers_snapshotted: int
    snapshot_at: str


def snapshot_scores(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    run_id: int | None = None,
    snapshot_at: str | None = None,
) -> ScoreSnapshotResult:
    timestamp = snapshot_at or utc_now_iso()
    rows = build_review_queue(conn, tickers=tickers, limit=limit, min_score=min_score)
    payload = []
    for row in rows:
        content_hash = _content_hash(
            ticker=row.ticker,
            snapshot_at=timestamp,
            run_id=run_id,
            score_total=row.score_total,
            research_status=row.research_status,
            evidence_coverage_score=row.evidence_coverage_score,
            review_priority=row.review_priority,
            review_bucket=row.review_bucket,
            tracking_frequency=row.tracking_frequency,
            source_link_count=row.source_link_count,
            latest_evidence_at=row.latest_evidence_at,
            required_review_checks=row.required_review_checks,
            invalidating_conditions=row.invalidating_conditions,
        )
        payload.append(
            (
                row.ticker,
                timestamp,
                run_id,
                row.score_total,
                row.research_status,
                row.evidence_coverage_score,
                row.review_priority,
                row.review_bucket,
                row.tracking_frequency,
                row.source_link_count,
                row.latest_evidence_at,
                row.required_review_checks,
                row.invalidating_conditions,
                content_hash,
            )
        )
    before_count = _snapshot_count(conn)
    conn.executemany(
        """
        INSERT INTO score_snapshots (
            ticker, snapshot_at, run_id, score_total, research_status,
            evidence_coverage_score, review_priority, review_bucket,
            tracking_frequency, source_link_count, latest_evidence_at,
            required_review_checks, invalidating_conditions, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO NOTHING
        """,
        payload,
    )
    after_count = _snapshot_count(conn)
    return ScoreSnapshotResult(
        snapshots_written=after_count - before_count,
        tickers_snapshotted=len(rows),
        snapshot_at=timestamp,
    )


def list_score_snapshots(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    limit: int = 20,
) -> list[sqlite3.Row]:
    if ticker:
        return conn.execute(
            """
            SELECT ticker, snapshot_at, run_id, score_total, research_status,
                   evidence_coverage_score, review_priority, review_bucket,
                   tracking_frequency, source_link_count, latest_evidence_at
            FROM score_snapshots
            WHERE ticker = ?
            ORDER BY snapshot_at DESC, id DESC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        ).fetchall()
    return conn.execute(
        """
        SELECT ticker, snapshot_at, run_id, score_total, research_status,
               evidence_coverage_score, review_priority, review_bucket,
               tracking_frequency, source_link_count, latest_evidence_at
        FROM score_snapshots
        ORDER BY snapshot_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _snapshot_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM score_snapshots").fetchone()
    return int(row["count"] or 0)


def _content_hash(**values: object) -> str:
    ordered = [
        str(values.get("ticker") or ""),
        str(values.get("snapshot_at") or ""),
        str(values.get("run_id") or ""),
        str(values.get("score_total") or ""),
        str(values.get("research_status") or ""),
        str(values.get("evidence_coverage_score") or ""),
        str(values.get("review_priority") or ""),
        str(values.get("review_bucket") or ""),
        str(values.get("tracking_frequency") or ""),
        str(values.get("source_link_count") or ""),
        str(values.get("latest_evidence_at") or ""),
        str(values.get("required_review_checks") or ""),
        str(values.get("invalidating_conditions") or ""),
    ]
    return hashlib.sha256("|".join(ordered).encode("utf-8")).hexdigest()
