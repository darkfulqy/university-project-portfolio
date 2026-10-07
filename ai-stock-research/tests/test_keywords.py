import unittest
from pathlib import Path
import tempfile

from ai_stock_discovery.analysis.keywords import (
    TAG_SIGNAL_SOURCE_TYPE,
    scan_local_ai_context,
    scan_text,
)
from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db


class KeywordTests(unittest.TestCase):
    def test_scan_text_keeps_source_backed_context_conservative(self) -> None:
        text = "The company discussed artificial intelligence products, but did not quantify revenue."
        signals = scan_text(ticker="TEST", text=text, source_type="unit", source_url="https://example.test")
        self.assertTrue(signals)
        self.assertEqual(signals[0].ai_relevance_level, 2)
        self.assertLess(signals[0].confidence, 1)

    def test_supply_chain_term_requires_ai_context(self) -> None:
        text = "The company sells transformers to utilities. No special end market was discussed."
        signals = scan_text(ticker="TEST", text=text, source_type="unit", source_url="https://example.test")
        self.assertEqual(signals, [])

    def test_risk_only_ai_context_stays_low_confidence(self) -> None:
        text = "Artificial intelligence may increase cybersecurity risks and phishing threats."
        signals = scan_text(ticker="TEST", text=text, source_type="unit", source_url="https://example.test")
        self.assertTrue(signals)
        self.assertEqual(signals[0].ai_relevance_level, 1)
        self.assertLess(signals[0].confidence, 0.45)

    def test_scan_local_company_profile_writes_scoreable_ai_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, description, source, updated_at
                    )
                    VALUES (
                        'TEST', 'Test Compute Inc.', 'Technology', 'Infrastructure',
                        'The company sells GPU systems for AI workloads and data center deployments.',
                        'https://example.test/profile', 'now'
                    )
                    """
                )
                result = scan_local_ai_context(conn, tickers=["TEST"])
                rows = conn.execute(
                    """
                    SELECT keyword, source_type, source_url
                    FROM ai_relevance_signals
                    WHERE ticker = 'TEST'
                    ORDER BY keyword
                    """
                ).fetchall()
                score = score_ticker(conn, "TEST")

        self.assertEqual(result.profiles_scanned, 1)
        self.assertGreater(result.profile_signals, 0)
        self.assertTrue(any(row["keyword"] == "gpu" for row in rows))
        self.assertTrue(all(row["source_url"] == "https://example.test/profile" for row in rows))
        self.assertGreater(score.ai_relevance_score, 0)

    def test_scan_local_ai_tags_writes_conservative_idempotent_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Power Inc.', 'unit', 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_industry_tags (
                        ticker, tag, confidence, source_type, source_url,
                        evidence_snippet, method, updated_at
                    )
                    VALUES (
                        'TEST', 'data_center_infrastructure', 0.7, 'company_profile',
                        'https://example.test/profile',
                        'Provides power distribution equipment for data center customers.',
                        'rule', 'now'
                    )
                    """
                )
                first = scan_local_ai_context(conn, tickers=["TEST"])
                second = scan_local_ai_context(conn, tickers=["TEST"])
                rows = conn.execute(
                    """
                    SELECT keyword, source_type, ai_relevance_level, confidence, context_snippet
                    FROM ai_relevance_signals
                    WHERE ticker = 'TEST'
                    """
                ).fetchall()

        self.assertEqual(first.tag_signals, 1)
        self.assertEqual(second.tag_signals, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_type"], TAG_SIGNAL_SOURCE_TYPE)
        self.assertEqual(rows[0]["ai_relevance_level"], 1)
        self.assertLessEqual(rows[0]["confidence"], 0.55)
        self.assertIn("does not prove revenue", rows[0]["context_snippet"])


if __name__ == "__main__":
    unittest.main()
