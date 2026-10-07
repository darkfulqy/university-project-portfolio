from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.change_report import build_snapshot_change_report, write_snapshot_change_csv
from ai_stock_discovery.database import init_db, open_db


class SnapshotChangeReportTests(unittest.TestCase):
    def test_change_report_compares_latest_two_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "snapshot_changes.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', '2026-06-01')
                    """
                )
                _insert_snapshot(
                    conn,
                    ticker="TEST",
                    snapshot_at="2026-06-01T00:00:00Z",
                    run_id=1,
                    score_total=50,
                    evidence_coverage_score=40,
                    review_priority="needs_valuation_snapshot",
                    review_bucket="evidence_gap_review",
                    source_link_count=2,
                    content_hash="test-old",
                )
                _insert_snapshot(
                    conn,
                    ticker="TEST",
                    snapshot_at="2026-06-02T00:00:00Z",
                    run_id=2,
                    score_total=82,
                    evidence_coverage_score=100,
                    review_priority="ready_for_human_review",
                    review_bucket="priority_research",
                    source_link_count=5,
                    content_hash="test-new",
                )
                _insert_snapshot(
                    conn,
                    ticker="NEW",
                    snapshot_at="2026-06-02T00:00:00Z",
                    run_id=2,
                    score_total=0,
                    evidence_coverage_score=0,
                    review_priority="needs_ai_evidence",
                    review_bucket="evidence_gap_review",
                    source_link_count=0,
                    content_hash="new-only",
                )
                rows = build_snapshot_change_report(conn, tickers=["TEST", "NEW"])
            write_snapshot_change_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        by_ticker = {row.ticker: row for row in rows}
        self.assertEqual(by_ticker["TEST"].change_type, "review_bucket_changed")
        self.assertEqual(by_ticker["TEST"].attention_level, "bucket_upgrade_review")
        self.assertEqual(by_ticker["TEST"].score_delta, 32.0)
        self.assertEqual(by_ticker["TEST"].evidence_coverage_delta, 60.0)
        self.assertEqual(by_ticker["TEST"].source_link_count_delta, 3)
        self.assertIn("verify original sources", by_ticker["TEST"].report_notes)
        self.assertEqual(by_ticker["NEW"].change_type, "new_snapshot_no_prior")
        self.assertIsNone(by_ticker["NEW"].previous_snapshot_at)
        self.assertIn("ticker,company_name", text)
        self.assertIn("bucket_upgrade_review", text)

    def test_changed_only_filters_no_material_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_snapshot(
                    conn,
                    ticker="SAME",
                    snapshot_at="2026-06-01T00:00:00Z",
                    run_id=1,
                    score_total=10,
                    evidence_coverage_score=20,
                    review_priority="needs_ai_evidence",
                    review_bucket="evidence_gap_review",
                    source_link_count=1,
                    content_hash="same-old",
                )
                _insert_snapshot(
                    conn,
                    ticker="SAME",
                    snapshot_at="2026-06-02T00:00:00Z",
                    run_id=2,
                    score_total=10,
                    evidence_coverage_score=20,
                    review_priority="needs_ai_evidence",
                    review_bucket="evidence_gap_review",
                    source_link_count=1,
                    content_hash="same-new",
                )
                all_rows = build_snapshot_change_report(conn, tickers=["SAME"])
                changed_rows = build_snapshot_change_report(conn, tickers=["SAME"], changed_only=True)

        self.assertEqual(all_rows[0].change_type, "no_material_change")
        self.assertEqual(changed_rows, [])


def _insert_snapshot(
    conn,
    *,
    ticker: str,
    snapshot_at: str,
    run_id: int,
    score_total: float,
    evidence_coverage_score: float,
    review_priority: str,
    review_bucket: str,
    source_link_count: int,
    content_hash: str,
) -> None:
    conn.execute(
        """
        INSERT INTO score_snapshots (
            ticker, snapshot_at, run_id, score_total, research_status,
            evidence_coverage_score, review_priority, review_bucket,
            tracking_frequency, source_link_count, latest_evidence_at,
            required_review_checks, invalidating_conditions, content_hash
        )
        VALUES (?, ?, ?, ?, 'watchlist_candidate', ?, ?, ?, 'weekly', ?, ?, '', '', ?)
        """,
        (
            ticker,
            snapshot_at,
            run_id,
            score_total,
            evidence_coverage_score,
            review_priority,
            review_bucket,
            source_link_count,
            snapshot_at,
            content_hash,
        ),
    )


if __name__ == "__main__":
    unittest.main()
