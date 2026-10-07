from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.tracking import list_score_snapshots, snapshot_scores


class TrackingTests(unittest.TestCase):
    def test_snapshot_scores_persists_review_state_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', 'now')
                    """
                )
                result = snapshot_scores(
                    conn,
                    tickers=["TEST"],
                    run_id=7,
                    snapshot_at="2026-06-01T00:00:00Z",
                )
                duplicate = snapshot_scores(
                    conn,
                    tickers=["TEST"],
                    run_id=7,
                    snapshot_at="2026-06-01T00:00:00Z",
                )
                rows = list_score_snapshots(conn, ticker="TEST")

        self.assertEqual(result.tickers_snapshotted, 1)
        self.assertEqual(result.snapshots_written, 1)
        self.assertEqual(duplicate.snapshots_written, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "TEST")
        self.assertEqual(rows[0]["run_id"], 7)
        self.assertEqual(rows[0]["review_bucket"], "evidence_gap_review")
        self.assertEqual(rows[0]["review_priority"], "needs_ai_evidence")

    def test_list_score_snapshots_returns_recent_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', 'now')
                    """
                )
                snapshot_scores(conn, tickers=["TEST"], snapshot_at="2026-06-01T00:00:00Z")
                snapshot_scores(conn, tickers=["TEST"], snapshot_at="2026-06-02T00:00:00Z")
                rows = list_score_snapshots(conn, ticker="TEST", limit=2)

        self.assertEqual([row["snapshot_at"] for row in rows], ["2026-06-02T00:00:00Z", "2026-06-01T00:00:00Z"])


if __name__ == "__main__":
    unittest.main()
