from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.backtest import build_snapshot_backtest, write_backtest_csv
from ai_stock_discovery.database import init_db, open_db


class BacktestTests(unittest.TestCase):
    def test_build_snapshot_backtest_uses_local_price_bars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_snapshot(conn, ticker="TEST", snapshot_at="2026-06-01T00:00:00Z")
                _insert_bar(conn, "TEST", "2026-06-01", 10, 100)
                _insert_bar(conn, "TEST", "2026-06-02", 11, 101)
                _insert_bar(conn, "TEST", "2026-06-03", 12, 103)
                rows = build_snapshot_backtest(conn, tickers=["TEST"], horizon_bars=2)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].outcome_status, "complete")
        self.assertEqual(rows[0].entry_date, "2026-06-01")
        self.assertEqual(rows[0].exit_date, "2026-06-03")
        self.assertEqual(rows[0].forward_return_pct, 20.0)
        self.assertEqual(rows[0].benchmark_return_pct, 3.0)
        self.assertEqual(rows[0].excess_return_pct, 17.0)
        self.assertIn("not investment advice", rows[0].notes)

    def test_backtest_report_keeps_incomplete_outcomes_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "backtest.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_snapshot(conn, ticker="TEST", snapshot_at="2026-06-01T00:00:00Z")
                _insert_bar(conn, "TEST", "2026-06-01", 10, None)
                rows = build_snapshot_backtest(conn, tickers=["TEST"], horizon_bars=2)
            write_backtest_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(rows[0].outcome_status, "insufficient_forward_bars")
        self.assertIn("ticker,snapshot_at", text)
        self.assertIn("insufficient_forward_bars", text)

    def test_complete_only_filters_incomplete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_snapshot(conn, ticker="TEST", snapshot_at="2026-06-01T00:00:00Z")
                rows = build_snapshot_backtest(conn, tickers=["TEST"], include_incomplete=False)

        self.assertEqual(rows, [])


def _insert_snapshot(
    conn,
    *,
    ticker: str,
    snapshot_at: str,
    score_total: float = 70.0,
    review_bucket: str = "watch_observe",
) -> None:
    conn.execute(
        """
        INSERT INTO score_snapshots (
            ticker, snapshot_at, run_id, score_total, research_status,
            evidence_coverage_score, review_priority, review_bucket,
            tracking_frequency, source_link_count, latest_evidence_at,
            required_review_checks, invalidating_conditions, content_hash
        )
        VALUES (?, ?, 1, ?, 'watchlist_candidate', 80, 'ready_for_human_review',
                ?, 'biweekly', 3, '2026-06-01T00:00:00Z',
                'verify_original_source_links', 'Source-backed invalidating conditions.',
                ?)
        """,
        (ticker, snapshot_at, score_total, review_bucket, f"{ticker}-{snapshot_at}-{review_bucket}"),
    )


def _insert_bar(
    conn,
    ticker: str,
    bar_date: str,
    close: float,
    benchmark_close: float | None,
) -> None:
    conn.execute(
        """
        INSERT INTO market_price_bars (
            ticker, bar_date, close, volume, benchmark_close, source_type,
            source_name, source_path, content_hash, updated_at
        )
        VALUES (?, ?, ?, 1000, ?, 'unit', 'unit bars', 'bars.csv', ?, 'now')
        """,
        (ticker, bar_date, close, benchmark_close, f"{ticker}-{bar_date}"),
    )


if __name__ == "__main__":
    unittest.main()
