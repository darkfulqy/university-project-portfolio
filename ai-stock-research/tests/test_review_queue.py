from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.review_queue import build_review_queue, write_review_queue_csv


class ReviewQueueTests(unittest.TestCase):
    def test_review_queue_turns_missing_evidence_into_required_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 0, 0, 0, 1, 'now')
                    """
                )
                rows = build_review_queue(conn, tickers=["TEST"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].review_bucket, "evidence_gap_review")
        self.assertEqual(rows[0].tracking_frequency, "when_missing_sources_are_available")
        self.assertIn("verify_original_source_links", rows[0].required_review_checks)
        self.assertIn("verify_ai_relevance_maps_to_business_impact", rows[0].required_review_checks)
        self.assertIn("load_or_refresh_sec_companyfacts_financials", rows[0].required_review_checks)
        self.assertIn("Thesis should fail", rows[0].invalidating_conditions)

    def test_review_queue_assigns_tracking_cadence_for_review_ready_candidate(self) -> None:
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
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example/test',
                        'artificial intelligence', 'AI revenue and backlog context.', 4, 0.9, 'now'
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO financial_facts (
                        ticker, period, fiscal_year, fiscal_quarter, revenue,
                        gross_profit, operating_income, free_cash_flow, source, updated_at
                    )
                    VALUES ('TEST', ?, ?, 'Q1', ?, ?, ?, ?, 'https://sec.example/facts', 'now')
                    """,
                    [
                        ("2025-03-31", 2025, 100, 40, 10, 5),
                        ("2026-03-31", 2026, 140, 60, 20, 12),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO valuation_snapshots (
                        ticker, date, price, market_cap, enterprise_value, ev_sales, sector_percentile,
                        source, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 10, 1000000000, 900000000, 3.2, 20,
                        'https://valuation.example/test', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO expectation_gap_signals (
                        ticker, signal_date, source_type, source_name, source_url,
                        signal_type, direction, description, confidence, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'manual', 'unit', 'https://gap.example/test',
                        'legacy_market_label', 'supports_gap', 'Legacy label evidence.', 0.9, 'gap-hash', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO catalysts (
                        ticker, catalyst_type, catalyst_date, description, source_url,
                        confidence, status, updated_at
                    )
                    VALUES (
                        'TEST', 'earnings_or_filing', '2026-08-01', 'Source-backed earnings catalyst.',
                        'https://catalyst.example/test', 0.9, 'scheduled', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO market_confirmation_signals (
                        ticker, signal_date, source_type, source_name, source_path,
                        signal_type, direction, description, confidence, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'local', 'unit', 'local.csv',
                        'relative_strength', 'bullish', 'Source-backed confirmation.', 0.8,
                        'market-hash', 'now'
                    )
                    """
                )
                rows = build_review_queue(conn, tickers=["TEST"], cards_dir=Path("cards"))

        self.assertEqual(rows[0].review_priority, "ready_for_human_review")
        self.assertIn(rows[0].review_bucket, {"watch_observe", "priority_research"})
        self.assertIn(rows[0].tracking_frequency, {"weekly", "biweekly"})
        self.assertIn("review_invalidating_conditions", rows[0].required_review_checks)
        self.assertEqual(rows[0].card_path, str(Path("cards") / "TEST.md"))

    def test_write_review_queue_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "review_queue.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', 'now')
                    """
                )
                rows = build_review_queue(conn, tickers=["TEST"])
            write_review_queue_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("ticker,company_name", text)
        self.assertIn("TEST,Test Corp", text)


if __name__ == "__main__":
    unittest.main()
