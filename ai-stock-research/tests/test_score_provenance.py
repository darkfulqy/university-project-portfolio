from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.score_provenance import build_score_provenance, write_score_provenance_csv


class ScoreProvenanceTests(unittest.TestCase):
    def test_score_provenance_blocks_rows_without_source_links(self) -> None:
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
                rows = build_score_provenance(conn, tickers=["MISS"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].score_trace_status, "no_source_links_no_company_conclusion")
        self.assertEqual(rows[0].ai_trace_status, "zero_score_no_sources")
        self.assertEqual(rows[0].thesis_gate, "evidence_gap_no_company_thesis")
        self.assertEqual(rows[0].score_source_links, "")
        self.assertIn("do not form company-level conclusions", rows[0].provenance_notes)

    def test_score_provenance_traces_nonzero_component_to_source_link(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "score_provenance.csv"
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
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example/test',
                        'artificial intelligence', 'AI context from a filing.', 3, 0.8, '2026-05-31'
                    )
                    """
                )
                rows = build_score_provenance(conn, tickers=["TEST"])
            write_score_provenance_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertGreater(row.ai_relevance_score, 0)
        self.assertEqual(row.ai_trace_status, "traced_score")
        self.assertEqual(row.score_trace_status, "thesis_source_gap")
        self.assertIn("https://sec.example/test", row.score_source_links)
        self.assertIn("ai=1", row.source_modules)
        self.assertIn("ticker,company_name,score_total", text)
        self.assertIn("TEST,Test Corp", text)


if __name__ == "__main__":
    unittest.main()
