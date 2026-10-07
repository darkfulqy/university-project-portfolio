from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.research_pool import build_research_pool, write_research_pool_csv


class ResearchPoolTests(unittest.TestCase):
    def test_research_pool_blocks_source_gap_items_from_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, security_type,
                        is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('MISS', 'Missing Sources Inc.', 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, '2026-06-01')
                    """
                )
                rows = build_research_pool(conn, tickers=["MISS"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].score_layer, "archive_or_event_watch")
        self.assertEqual(rows[0].research_pool_status, "source_blocked")
        self.assertEqual(rows[0].thesis_gate, "evidence_gap_no_company_thesis")
        self.assertIn("load_sec_companyfacts", rows[0].next_action)
        self.assertIn("do not form or upgrade", rows[0].pool_notes)

    def test_research_pool_combines_score_layer_thesis_gate_and_review_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "research_pool.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_complete_source_chain(conn)
                rows = build_research_pool(conn, tickers=["TEST"], cards_dir=Path("cards"))
            write_research_pool_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.ticker, "TEST")
        self.assertEqual(row.sector, "Technology")
        self.assertEqual(row.industry, "AI Infrastructure")
        self.assertIn(row.score_layer, {"priority_research_pool", "watch_pool", "confirmation_waitlist", "archive_or_event_watch"})
        self.assertNotEqual(row.research_pool_status, "source_blocked")
        self.assertIn(
            row.thesis_gate,
            {
                "priority_thesis_review_ready",
                "watchlist_thesis_review_ready",
                "source_chain_complete_but_low_score",
            },
        )
        self.assertEqual(Path(row.card_path).name, "TEST.md")
        self.assertIn("ticker,company_name,sector,industry", text)
        self.assertIn("TEST,Test Corp,Technology,AI Infrastructure", text)


def _insert_complete_source_chain(conn) -> None:
    conn.execute(
        """
        INSERT INTO company_profile (
            ticker, company_name, sector, industry, source, updated_at
        )
        VALUES ('TEST', 'Test Corp', 'Technology', 'AI Infrastructure', 'unit', '2026-06-01')
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
            'artificial intelligence', 'AI revenue and backlog context.', 4, 0.9, '2026-05-31'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO ai_industry_tags (
            ticker, tag, confidence, source_type, source_url, evidence_snippet, method, updated_at
        )
        VALUES (
            'TEST', 'ai_infrastructure', 0.9, 'profile', 'https://profile.example/test',
            'Company supplies AI infrastructure.', 'manual', '2026-05-31'
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO financial_facts (
            ticker, period, fiscal_year, fiscal_quarter, revenue,
            gross_profit, operating_income, free_cash_flow, source, updated_at
        )
        VALUES ('TEST', ?, ?, 'Q1', ?, ?, ?, ?, 'https://sec.example/facts', '2026-05-31')
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
            'TEST', '2026-05-31', 10, 1000000000, 900000000, 3.2, 25,
            'https://valuation.example/test', '2026-05-31'
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
            'legacy_market_label', 'supports_gap', 'Legacy label evidence.', 0.9, 'gap-hash', '2026-05-31'
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
            'https://catalyst.example/test', 0.9, 'scheduled', '2026-05-31'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO risk_flags (
            ticker, risk_date, source_type, source_name, source_url, risk_type,
            severity, description, confidence, status, content_hash, updated_at
        )
        VALUES (
            'TEST', '2026-05-31', 'SEC 10-K', 'unit filing',
            'https://risk.example/test', 'customer_concentration', 'medium',
            'Source-backed risk flag.', 0.8, 'watch', 'risk-hash', '2026-05-31'
        )
        """
    )


if __name__ == "__main__":
    unittest.main()
