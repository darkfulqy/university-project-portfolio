from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.catalysts import (
    extract_catalysts_from_evidence,
    store_catalysts,
)
from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db


class CatalystTests(unittest.TestCase):
    def test_extract_and_store_catalysts_from_evidence(self) -> None:
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
                        'Company IR', 'IR press release', 'https://example.test/pr',
                        '2026-05-31', '2026-05-31', 'New product launch',
                        'Company announces new data center liquid cooling product.',
                        'The launch targets AI data center demand.', 0.8,
                        'TEST', 'ir_monitor', 'hash-1'
                    )
                    """
                )
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")
                inserted = store_catalysts(conn, candidates)
                score = score_ticker(conn, "TEST")
                rows = conn.execute(
                    "SELECT catalyst_type, source_url FROM catalysts WHERE ticker = 'TEST'"
                ).fetchall()
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(inserted, len(candidates))
        self.assertGreater(score.catalyst_score, 0)
        self.assertIn("product_launch", {row["catalyst_type"] for row in rows})
        self.assertIn("data_center_project", {row["catalyst_type"] for row in rows})

    def test_extract_catalysts_requires_related_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        source_type, source_name, url, fetched_at, raw_title, summary,
                        evidence_snippet, confidence, related_ticker, related_module, content_hash
                    )
                    VALUES (
                        'SEC RSS', 'SEC feed', 'https://example.test', 'now',
                        '8-K filing', 'Potential catalyst', '8-K filed', 0.9,
                        NULL, 'filing_monitor', 'hash-2'
                    )
                    """
                )
                candidates = extract_catalysts_from_evidence(conn)
        self.assertEqual(candidates, [])

    def test_extract_catalysts_ignores_ai_relevance_snippets(self) -> None:
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
                        'SEC 10-Q', 'SEC 10-Q', 'https://example.test/10q',
                        '2026-05-31', '2026-05-31', 'AI keyword',
                        'Automated keyword context candidate',
                        'SEC 10-Q artificial intelligence risk snippet', 0.6,
                        'TEST', 'ai_relevance', 'hash-3'
                    )
                    """
                )
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")
        self.assertEqual(candidates, [])

    def test_extract_catalysts_ignores_ir_page_snapshots(self) -> None:
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
                        'Company IR', 'Company IR page', 'https://example.test/investors',
                        NULL, '2026-05-31', 'Supermicro Data Center Server',
                        'IR page hash check. Full page content is not stored.',
                        'IR page observed with title: Supermicro Data Center Server',
                        0.7, 'TEST', 'ir_monitor', 'hash-ir-page'
                    )
                    """
                )
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")
        self.assertEqual(candidates, [])

    def test_extract_catalysts_does_not_treat_annual_meeting_announcement_as_product_launch(self) -> None:
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
                        'News RSS', 'GlobeNewswire Press Releases', 'https://example.test/annual-meeting',
                        'Mon, 01 Jun 2026 20:30 GMT', '2026-06-02',
                        'Test Corp Announces its 2026 Annual Meeting to be Held on July 6, 2026',
                        'Test Corp (Nasdaq: TEST) announces that the 2026 Annual Meeting will be held virtually.',
                        'The notice of meeting and proxy statement were distributed to stockholders.',
                        0.65, 'TEST', 'news_monitor', 'hash-annual-meeting'
                    )
                    """
                )
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")

        self.assertEqual(candidates, [])

    def test_extract_catalysts_from_sec_filing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO filings (
                        ticker, cik, form, filed_at, accession_number, filing_url,
                        document_url, parsed_status, updated_at
                    )
                    VALUES (
                        'TEST', '0000000001', '10-Q', '2026-05-31',
                        '0000000001-26-000001',
                        'https://sec.example/index.htm',
                        'https://sec.example/10q.htm',
                        'new', 'now'
                    )
                    """
                )
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")
                inserted = store_catalysts(conn, candidates)
                row = conn.execute(
                    """
                    SELECT catalyst_type, catalyst_date, source_url, description
                    FROM catalysts
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()

        self.assertEqual(inserted, 1)
        self.assertEqual(row["catalyst_type"], "earnings_or_filing")
        self.assertEqual(row["catalyst_date"], "2026-05-31")
        self.assertEqual(row["source_url"], "https://sec.example/10q.htm")
        self.assertIn("filing-review catalyst candidate only", row["description"])


if __name__ == "__main__":
    unittest.main()
