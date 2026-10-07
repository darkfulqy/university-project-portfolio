from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources import ir


class FakeHttpClient:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def get_text(self, url: str, accept: str | None = None) -> str:
        if url not in self.pages:
            raise ValueError(f"missing page: {url}")
        return self.pages[url]


class IrTests(unittest.TestCase):
    def test_upsert_ir_url_updates_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                ir.upsert_ir_url(conn, ticker="TEST", url="https://ir.example.test")
                row = conn.execute(
                    "SELECT ir_url FROM company_profile WHERE ticker = 'TEST'"
                ).fetchone()
                page = conn.execute(
                    "SELECT url FROM ir_pages WHERE ticker = 'TEST'"
                ).fetchone()
        self.assertEqual(row["ir_url"], "https://ir.example.test")
        self.assertEqual(page["url"], "https://ir.example.test")

    def test_upsert_ir_url_preserves_existing_profile_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, website, source, updated_at
                    )
                    VALUES (
                        'TEST', 'Test Corp', 'https://www.test.test',
                        'https://financialmodelingprep.com/stable/profile?symbol=TEST&apikey=***',
                        'now'
                    )
                    """
                )
                ir.upsert_ir_url(
                    conn,
                    ticker="TEST",
                    url="https://ir.test.test",
                    source="Company website IR discovery",
                )
                row = conn.execute(
                    "SELECT ir_url, source FROM company_profile WHERE ticker = 'TEST'"
                ).fetchone()
                page = conn.execute(
                    "SELECT source FROM ir_pages WHERE ticker = 'TEST'"
                ).fetchone()
        self.assertEqual(row["ir_url"], "https://ir.test.test")
        self.assertIn("financialmodelingprep", row["source"])
        self.assertEqual(page["source"], "Company website IR discovery")

    def test_import_ir_urls_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "ir.csv"
            csv_path.write_text("ticker,ir_url\nAAA,https://ir.aaa.test\n", encoding="utf-8")
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = ir.import_ir_urls(conn, csv_path)
        self.assertEqual(count, 1)

    def test_batch_check_ir_pages_tracks_changes_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                ir.upsert_ir_url(conn, ticker="AAA", url="https://ir.aaa.test")
                ir.upsert_ir_url(conn, ticker="ERR", url="https://ir.err.test")
                first = ir.check_ir_pages(
                    conn,
                    FakeHttpClient(
                        {
                            "https://ir.aaa.test": "<html><title>AAA IR</title></html>",
                        }
                    ),
                    limit=10,
                )
                second = ir.check_ir_pages(
                    conn,
                    FakeHttpClient(
                        {
                            "https://ir.aaa.test": "<html><title>AAA Investor News</title></html>",
                        }
                    ),
                    ticker="AAA",
                    limit=10,
                )
                page = conn.execute(
                    """
                    SELECT last_title, last_error
                    FROM ir_pages
                    WHERE ticker = 'AAA'
                    """
                ).fetchone()
                error_page = conn.execute(
                    """
                    SELECT last_error
                    FROM ir_pages
                    WHERE ticker = 'ERR'
                    """
                ).fetchone()
                evidence_count = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM evidence_items
                    WHERE related_ticker = 'AAA' AND related_module = 'ir_monitor'
                    """
                ).fetchone()["count"]
        self.assertEqual(first.checked, 1)
        self.assertEqual(first.changed, 0)
        self.assertEqual(len(first.errors), 1)
        self.assertEqual(second.checked, 1)
        self.assertEqual(second.changed, 1)
        self.assertEqual(page["last_title"], "AAA Investor News")
        self.assertIsNone(page["last_error"])
        self.assertIn("missing page", error_page["last_error"])
        self.assertEqual(evidence_count, 2)

    def test_discover_ir_pages_from_source_backed_profile_website(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, website, source, updated_at
                    )
                    VALUES (
                        'TEST', 'Test Corp', 'https://www.test.test',
                        'https://financialmodelingprep.com/stable/profile?symbol=TEST&apikey=***',
                        'now'
                    )
                    """
                )
                result = ir.discover_ir_pages_from_profiles(
                    conn,
                    FakeHttpClient(
                        {
                            "https://www.test.test": """
                                <html><body>
                                  <a href="https://ir.test.test">Investor Relations</a>
                                </body></html>
                            """,
                            "https://ir.test.test": """
                                <html><title>Test Corp Investor Relations</title>
                                <body>SEC filings, annual report, quarterly results.</body></html>
                            """,
                        }
                    ),
                    tickers=["TEST"],
                    limit=5,
                )
                row = conn.execute(
                    "SELECT ir_url, source FROM company_profile WHERE ticker = 'TEST'"
                ).fetchone()
                page = conn.execute(
                    """
                    SELECT source, last_title
                    FROM ir_pages
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()
                evidence_count = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'ir_monitor'
                    """
                ).fetchone()["count"]
        self.assertEqual(result.candidates, 1)
        self.assertEqual(result.discovered, 1)
        self.assertEqual(result.checked, 1)
        self.assertEqual(row["ir_url"], "https://ir.test.test")
        self.assertIn("financialmodelingprep", row["source"])
        self.assertEqual(page["source"], "Company website IR discovery")
        self.assertEqual(page["last_title"], "Test Corp Investor Relations")
        self.assertEqual(evidence_count, 1)


if __name__ == "__main__":
    unittest.main()
