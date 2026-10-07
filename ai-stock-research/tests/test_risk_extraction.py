from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.risk_extraction import (
    extract_risk_flags_from_evidence,
    scan_risk_text,
    store_extracted_risk_flags,
)
from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db


class RiskExtractionTests(unittest.TestCase):
    def test_scan_risk_text_finds_specific_sec_risk_candidates(self) -> None:
        text = (
            "Risk Factors: A limited number of customers account for a substantial portion "
            "of our revenue. Export controls may also limit sales to China."
        )
        flags = scan_risk_text(
            ticker="TEST",
            text=text,
            source_type="SEC 10-K",
            source_name="SEC filing documents",
            source_url="https://www.sec.gov/test",
            risk_date="2026-05-31",
        )
        risk_types = {flag.risk_type for flag in flags}
        self.assertIn("customer_concentration", risk_types)
        self.assertIn("export_control", risk_types)
        self.assertTrue(all(flag.status == "watch" for flag in flags))
        self.assertTrue(all(flag.source_url == "https://www.sec.gov/test" for flag in flags))

    def test_scan_risk_text_ignores_generic_risk_language(self) -> None:
        flags = scan_risk_text(
            ticker="TEST",
            text="The company discusses risks and uncertainties in ordinary course disclosures.",
            source_type="SEC 10-K",
            source_name="SEC filing documents",
            source_url="https://www.sec.gov/test",
        )
        self.assertEqual(flags, [])

    def test_extract_from_evidence_feeds_risk_penalty_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        source_type, source_name, url, published_at, fetched_at,
                        raw_title, summary, evidence_snippet, confidence,
                        related_ticker, related_module, content_hash
                    )
                    VALUES (
                        'SEC 10-K', 'SEC filing documents', 'https://www.sec.gov/test',
                        '2026-05-31', 'now', 'Risk Factors',
                        'Risk factor excerpt',
                        'The company depends on a single source supplier and may face component shortages.',
                        0.7, 'TEST', 'filing_monitor', 'risk-source-1'
                    )
                    """
                )
                flags = extract_risk_flags_from_evidence(conn, ticker="TEST")
                count = store_extracted_risk_flags(conn, flags)
                score = score_ticker(conn, "TEST")
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'risk_flag'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertLess(score.risk_penalty, 0)
        self.assertIn("supply_chain", score.risk_summary)
        self.assertEqual(evidence["url"], "https://www.sec.gov/test")

    def test_extract_from_evidence_ignores_ai_relevance_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        source_type, source_name, url, published_at, fetched_at,
                        raw_title, summary, evidence_snippet, confidence,
                        related_ticker, related_module, content_hash
                    )
                    VALUES (
                        'SEC 10-K', 'SEC filing documents', 'https://www.sec.gov/test',
                        '2026-05-31', 'now', 'AI keyword',
                        'Automated keyword context candidate',
                        'Artificial intelligence disclosure mentions geopolitical tensions.',
                        0.35, 'TEST', 'ai_relevance', 'risk-source-ai'
                    )
                    """
                )
                flags = extract_risk_flags_from_evidence(conn, ticker="TEST")

        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main()
