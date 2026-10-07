from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.expectations import (
    infer_profile_ai_tag_expectation_gaps,
    import_expectation_gap_csv,
    upsert_expectation_gap_signals,
)


class ExpectationGapTests(unittest.TestCase):
    def test_import_expectation_gap_csv_feeds_score_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "expectation_gap.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,signal_date,source_type,source_name,source_url,signal_type,direction,magnitude,description,confidence",
                        "TEST,2026-05-31,manual_research,unit source,https://example.test/research,legacy_market_label,supports_gap,,Company still categorized as traditional industrial despite source-backed AI capex exposure,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_expectation_gap_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                signal = conn.execute(
                    """
                    SELECT signal_type, direction
                    FROM expectation_gap_signals
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'expectation_gap'
                    """
                ).fetchone()
        self.assertEqual(count, 1)
        self.assertGreater(score.expectation_gap_score, 0)
        self.assertEqual(signal["signal_type"], "legacy_market_label")
        self.assertEqual(signal["direction"], "supports_gap")
        self.assertEqual(evidence["url"], "https://example.test/research")

    def test_contradicting_expectation_gap_signal_does_not_create_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "expectation_gap.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,source_url,signal_type,direction,description,confidence",
                        "TEST,https://example.test/research,valuation_multiple_lag,contradicts_gap,Source says valuation already reflects expected AI growth,0.9",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_expectation_gap_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
        self.assertEqual(score.expectation_gap_score, 0)

    def test_infer_profile_ai_tag_expectation_gap_from_legacy_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, source, updated_at
                    )
                    VALUES (
                        'TEST', 'Test Power Inc.', 'Industrials', 'Electrical Equipment & Parts',
                        'https://example.test/profile', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_industry_tags (
                        ticker, tag, confidence, source_type, source_url,
                        evidence_snippet, updated_at
                    )
                    VALUES (
                        'TEST', 'power', 0.7, 'company_profile',
                        'https://example.test/profile', 'Power exposure for data centers.', 'now'
                    )
                    """
                )
                signals = infer_profile_ai_tag_expectation_gaps(conn, tickers=["TEST"])
                count = upsert_expectation_gap_signals(conn, signals)
                score = score_ticker(conn, "TEST")
                signal = conn.execute(
                    """
                    SELECT signal_type, direction, source_name, source_url, description
                    FROM expectation_gap_signals
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()

        self.assertEqual(len(signals), 1)
        self.assertEqual(count, 1)
        self.assertGreater(score.expectation_gap_score, 0)
        self.assertEqual(signal["signal_type"], "legacy_market_label")
        self.assertEqual(signal["direction"], "supports_gap")
        self.assertIn("Local profile/AI tag", signal["source_name"])
        self.assertIn("https://example.test/profile", signal["source_url"])
        self.assertIn("does not prove market mispricing", signal["description"])

    def test_infer_profile_ai_tag_expectation_gap_skips_non_legacy_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, source, updated_at
                    )
                    VALUES (
                        'CHIP', 'Chip Inc.', 'Technology', 'Semiconductors',
                        'https://example.test/profile', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_industry_tags (
                        ticker, tag, confidence, source_type, source_url,
                        evidence_snippet, updated_at
                    )
                    VALUES (
                        'CHIP', 'semiconductors', 0.8, 'company_profile',
                        'https://example.test/profile', 'Semiconductor exposure.', 'now'
                    )
                    """
                )
                signals = infer_profile_ai_tag_expectation_gaps(conn, tickers=["CHIP"])

        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
