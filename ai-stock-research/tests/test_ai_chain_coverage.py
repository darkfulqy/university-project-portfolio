from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.ai_chain_coverage import build_ai_chain_coverage, write_ai_chain_coverage_csv
from ai_stock_discovery.database import init_db, open_db


class AiChainCoverageTests(unittest.TestCase):
    def test_coverage_reports_untagged_candidates_as_mapping_gap(self) -> None:
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
                rows = build_ai_chain_coverage(conn, tickers=["MISS"])

        untagged = [row for row in rows if row.chain_category == "unmapped_ai_chain"][0]
        self.assertEqual(untagged.candidate_count, 1)
        self.assertEqual(untagged.source_blocked_count, 1)
        self.assertEqual(untagged.top_tickers, "MISS")
        self.assertEqual(untagged.next_action, "tag_ai_chain_or_import_manual_source_backed_tags")
        self.assertIn("without AI industry-chain tags", untagged.coverage_notes)

    def test_coverage_groups_source_backed_tags_by_chain_category(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "ai_chain_coverage.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_industry_tags (
                        ticker, tag, confidence, source_type, source_url, evidence_snippet, method, updated_at
                    )
                    VALUES (
                        'TEST', 'cooling', 0.8, 'manual', 'https://source.example/cooling',
                        'Source-backed liquid cooling exposure.', 'manual', '2026-06-01'
                    )
                    """
                )
                rows = build_ai_chain_coverage(conn, tickers=["TEST"], top_limit=3)
            write_ai_chain_coverage_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        cooling = [row for row in rows if row.chain_category == "cooling"][0]
        self.assertEqual(cooling.candidate_count, 1)
        self.assertEqual(cooling.source_backed_tag_count, 1)
        self.assertEqual(cooling.source_blocked_count, 1)
        self.assertEqual(cooling.top_tickers, "TEST")
        self.assertEqual(cooling.next_action, "complete_thesis_source_chain_for_tagged_candidates")
        self.assertIn("chain_category,mapped_tags", text)
        self.assertIn("cooling,cooling,1,1", text)


if __name__ == "__main__":
    unittest.main()
