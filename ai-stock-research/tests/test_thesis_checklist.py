from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.thesis_checklist import build_thesis_checklist, write_thesis_checklist_csv


class ThesisChecklistTests(unittest.TestCase):
    def test_checklist_blocks_company_thesis_when_core_sources_are_missing(self) -> None:
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
                rows = build_thesis_checklist(conn, tickers=["MISS"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].thesis_gate, "evidence_gap_no_company_thesis")
        self.assertIn("missing:load_sec_ir_or_announcement_ai_business_context", rows[0].ai_business_impact_check)
        self.assertIn("load_sec_companyfacts", rows[0].next_source_actions)
        self.assertIn("Do not form a company-level thesis", rows[0].checklist_notes)

    def test_checklist_accepts_complete_source_chain_without_claiming_investment_merit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "thesis_checklist.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_complete_source_chain(conn)
                rows = build_thesis_checklist(conn, tickers=["TEST"])
            write_thesis_checklist_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0].thesis_gate, {
            "priority_thesis_review_ready",
            "watchlist_thesis_review_ready",
            "source_chain_complete_but_low_score",
        })
        self.assertEqual(rows[0].ai_business_impact_check, "present:ai_context_and_chain_tag_loaded")
        self.assertEqual(rows[0].fundamentals_check, "present:sec_financial_trend_context_loaded")
        self.assertEqual(rows[0].expectation_gap_check, "present:expectation_gap_signal_loaded")
        self.assertEqual(rows[0].catalyst_check, "present:dated_or_source_backed_catalyst_loaded")
        self.assertEqual(rows[0].risk_invalidation_check, "present:risk_flags_loaded")
        self.assertIn("verify_original_source_links_before_any_conclusion", rows[0].next_source_actions)
        self.assertIn("ticker,company_name", text)
        self.assertIn("TEST,Test Corp", text)


def _insert_complete_source_chain(conn) -> None:
    conn.execute(
        """
        INSERT INTO company_profile (ticker, company_name, source, updated_at)
        VALUES ('TEST', 'Test Corp', 'unit', '2026-06-01')
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
