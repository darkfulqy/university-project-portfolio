from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.industry_tags import (
    infer_ai_chain_tags,
    import_ai_tags_csv,
    infer_tags_for_ticker,
    replace_inferred_tags_for_ticker,
    store_ai_industry_tags,
)
from ai_stock_discovery.database import init_db, open_db


class IndustryTagTests(unittest.TestCase):
    def test_infer_tags_from_profile_description(self) -> None:
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
                        'TEST', 'Test Corp', 'Technology', 'Semiconductors',
                        'Supplier of data center GPUs and advanced packaging tools.',
                        'unit-profile', 'now'
                    )
                    """
                )
                tags = infer_tags_for_ticker(conn, "TEST")
        tag_names = {tag.tag for tag in tags}
        self.assertIn("ai_compute", tag_names)
        self.assertIn("semiconductors", tag_names)
        self.assertIn("data_center_infrastructure", tag_names)

    def test_store_tags_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                tags = infer_tags_for_ticker(conn, "TEST")
                self.assertEqual(tags, [])
                count = import_ai_tags_csv(conn, _write_tags_csv(Path(temp_dir)))
                evidence_count = conn.execute(
                    "SELECT COUNT(*) AS count FROM evidence_items WHERE related_module = 'ai_industry_tag'"
                ).fetchone()["count"]
        self.assertEqual(count, 1)
        self.assertEqual(evidence_count, 1)

    def test_store_inferred_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example',
                        'liquid cooling', 'Liquid cooling for AI data center workloads.',
                        2, 0.6, 'now'
                    )
                    """
                )
                count = store_ai_industry_tags(conn, infer_tags_for_ticker(conn, "TEST"))
                rows = conn.execute(
                    "SELECT tag FROM ai_industry_tags WHERE ticker = 'TEST'"
                ).fetchall()
        self.assertGreater(count, 0)
        self.assertIn("cooling", {row["tag"] for row in rows})

    def test_low_confidence_risk_signal_does_not_create_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-Q', 'https://sec.example',
                        'artificial intelligence',
                        'Artificial intelligence may increase cybersecurity risks.',
                        1, 0.35, 'now'
                    )
                    """
                )
                tags = infer_tags_for_ticker(conn, "TEST")
        self.assertEqual(tags, [])

    def test_replace_inferred_tags_preserves_manual_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_ai_tags_csv(conn, _write_tags_csv(Path(temp_dir)))
                replace_inferred_tags_for_ticker(conn, "TEST", [])
                rows = conn.execute(
                    "SELECT tag, method FROM ai_industry_tags WHERE ticker = 'TEST'"
                ).fetchall()
        self.assertEqual([(row["tag"], row["method"]) for row in rows], [("cooling", "manual")])

    def test_batch_inference_uses_clean_universe_and_preserves_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES (?, ?, 0, 0, 0, 1, 'now')
                    """,
                    [
                        ("GOOD", "Good Compute Inc."),
                        ("SPAC", "Example Acquisition Corp Class A"),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, description, source, updated_at
                    )
                    VALUES (?, ?, 'Industrials', 'Thermal Management', ?, 'unit-profile', 'now')
                    """,
                    [
                        ("GOOD", "Good Compute Inc.", "Liquid cooling systems for AI data center workloads."),
                        ("SPAC", "Example Acquisition Corp Class A", "Liquid cooling systems."),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO universe_exclusions (
                        ticker, reason_type, reason, source_name, source_url,
                        severity, method, is_active, detected_at, content_hash
                    )
                    VALUES (
                        'SPAC', 'blank_check_or_spac', 'Excluded blank-check company',
                        'unit', 'unit', 'exclude', 'rule', 1, 'now', 'spac-exclusion'
                    )
                    """
                )
                result = infer_ai_chain_tags(conn, limit=10)
                rows = conn.execute(
                    "SELECT ticker, tag FROM ai_industry_tags ORDER BY ticker, tag"
                ).fetchall()

        self.assertEqual(result.tickers_considered, 1)
        self.assertEqual(result.tickers_tagged, 1)
        self.assertGreater(result.tags_written, 0)
        self.assertIn(("GOOD", "cooling"), [(row["ticker"], row["tag"]) for row in rows])
        self.assertNotIn("SPAC", {row["ticker"] for row in rows})

    def test_batch_inference_includes_source_backed_profiles_outside_universe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('GOOD', 'Good Compute Inc.', 0, 0, 0, 1, 'now')
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, description, source, updated_at
                    )
                    VALUES (?, ?, 'Technology', 'Semiconductors', ?, ?, 'now')
                    """,
                    [
                        ("GOOD", "Good Compute Inc.", "Liquid cooling systems for AI data center workloads.", "unit-profile"),
                        ("DISC", "Discovery Semiconductor Inc.", "Semiconductor power devices for data center power systems.", "https://financialmodelingprep.com/stable/profile?symbol=DISC&apikey=***"),
                        ("MAP", "Mapping Only Inc.", "Semiconductor mapping placeholder.", "SEC company_tickers"),
                    ],
                )
                result = infer_ai_chain_tags(conn, limit=10)
                rows = conn.execute(
                    "SELECT ticker, tag FROM ai_industry_tags ORDER BY ticker, tag"
                ).fetchall()

        tickers = {row["ticker"] for row in rows}
        self.assertEqual(result.tickers_considered, 2)
        self.assertIn("GOOD", tickers)
        self.assertIn("DISC", tickers)
        self.assertNotIn("MAP", tickers)


def _write_tags_csv(base: Path) -> Path:
    csv_path = base / "tags.csv"
    csv_path.write_text(
        "ticker,tag,confidence,source_url,evidence_snippet\n"
        "TEST,cooling,0.8,manual-source,Manual source says liquid cooling exposure\n",
        encoding="utf-8",
    )
    return csv_path


if __name__ == "__main__":
    unittest.main()
