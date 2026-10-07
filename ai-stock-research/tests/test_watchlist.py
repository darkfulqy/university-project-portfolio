from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.watchlist import build_watchlist, write_watchlist_csv


class WatchlistTests(unittest.TestCase):
    def test_build_watchlist_scores_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, sector, industry, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'Technology', 'Software', 'unit', 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example',
                        'artificial intelligence',
                        'Artificial intelligence revenue contribution was discussed.',
                        2, 0.6, 'now'
                    )
                    """
                )
                rows = build_watchlist(conn, limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ticker, "TEST")
        self.assertGreater(rows[0].score_total, 0)
        self.assertEqual(rows[0].evidence_count, 0)

    def test_write_watchlist_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "watchlist.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', 'now')
                    """
                )
                rows = build_watchlist(conn, limit=10)
            write_watchlist_csv(rows, output)
            text = output.read_text(encoding="utf-8")
        self.assertIn("ticker,company_name", text)
        self.assertIn("TEST,Test Corp", text)

    def test_watchlist_uses_clean_universe_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES (?, ?, ?, 0, 0, 1, 'now')
                    """,
                    [
                        ("AAA", "Common Corp", 0),
                        ("ETF", "ETF Corp", 1),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES (?, ?, 'unit', 'now')
                    """,
                    [
                        ("AAA", "Common Corp"),
                        ("ETF", "ETF Corp"),
                    ],
                )
                rows = build_watchlist(conn, limit=10)
        self.assertEqual([row.ticker for row in rows], ["AAA"])


if __name__ == "__main__":
    unittest.main()
