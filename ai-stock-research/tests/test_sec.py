import unittest
from pathlib import Path
import tempfile

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sec_companyfacts_backfill import (
    backfill_sec_companyfacts,
    record_sec_companyfacts_backfill_unavailable,
    select_sec_companyfacts_backfill_tickers,
)
from ai_stock_discovery.source_failures import record_source_failure
from ai_stock_discovery.sources import financials
from ai_stock_discovery.sources import sec
from ai_stock_discovery.sources.sec import resolve_primary_document_url


SUBMISSIONS_SAMPLE = {
    "filings": {
        "recent": {
            "form": ["10-K", "4"],
            "accessionNumber": ["0000000001-26-000001", "0000000001-26-000002"],
            "filingDate": ["2026-05-30", "2026-05-31"],
            "reportDate": ["2026-03-31", ""],
            "primaryDocument": ["test-20260331.htm", "ownership.xml"],
        }
    }
}


class SecTests(unittest.TestCase):
    def test_backfill_companyfacts_selects_fmp_financial_failures_and_writes_facts(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                if "/api/xbrl/companyfacts/" not in url:
                    raise AssertionError(f"Unexpected URL: {url}")
                return {
                    "facts": {
                        "us-gaap": {
                            "Revenues": {
                                "units": {
                                    "USD": [
                                        {
                                            "end": "2026-03-31",
                                            "fy": 2026,
                                            "fp": "Q1",
                                            "form": "10-Q",
                                            "val": 1000,
                                        }
                                    ]
                                }
                            },
                            "NetIncomeLoss": {
                                "units": {
                                    "USD": [
                                        {
                                            "end": "2026-03-31",
                                            "fy": 2026,
                                            "fp": "Q1",
                                            "form": "10-Q",
                                            "val": 120,
                                        }
                                    ]
                                }
                            },
                        }
                    }
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('TEST', '0000000001', 'Test Corp', 'unit', 'now')
                    """
                )
                record_source_failure(
                    conn,
                    source_name=financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,
                    ticker="TEST",
                    endpoint="financial-statements",
                    reason="TEST FMP financial statements: HTTP 402 while fetching https://example.test?apikey=***",
                    failure_type="http_402",
                )
                selected = select_sec_companyfacts_backfill_tickers(conn, limit=10)
                result = backfill_sec_companyfacts(conn, FakeClient(), tickers=selected)  # type: ignore[arg-type]
                fact_count = conn.execute(
                    "SELECT COUNT(*) AS count FROM financial_facts WHERE ticker = 'TEST'"
                ).fetchone()["count"]
                sec_status = conn.execute(
                    "SELECT status FROM data_source_status WHERE source_name = 'SEC companyfacts'"
                ).fetchone()["status"]
                fmp_failure_status = conn.execute(
                    """
                    SELECT status
                    FROM source_failures
                    WHERE source_name = ?
                      AND ticker = 'TEST'
                      AND endpoint = 'financial-statements'
                    """,
                    (financials.FMP_FINANCIAL_STATEMENTS_SOURCE_NAME,),
                ).fetchone()["status"]

        self.assertEqual(selected, ["TEST"])
        self.assertEqual(result.tickers_considered, 1)
        self.assertEqual(result.financial_fact_periods, 1)
        self.assertEqual(result.successes, ("TEST",))
        self.assertEqual(result.failures, ())
        self.assertEqual(fact_count, 1)
        self.assertEqual(sec_status, "ok")
        self.assertEqual(fmp_failure_status, "resolved")

    def test_companyfacts_backfill_without_user_agent_records_ticker_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('TEST', '0000000001', 'Test Corp', 'unit', 'now')
                    """
                )
                count = record_sec_companyfacts_backfill_unavailable(conn, tickers=["TEST"])
                row = conn.execute(
                    """
                    SELECT ticker, failure_type, status, reason, source_url
                    FROM source_failures
                    WHERE source_name = 'SEC companyfacts'
                      AND endpoint = 'companyfacts'
                    """
                ).fetchone()
                source_status = conn.execute(
                    "SELECT status FROM data_source_status WHERE source_name = 'SEC companyfacts'"
                ).fetchone()["status"]

        self.assertEqual(count, 1)
        self.assertEqual(row["ticker"], "TEST")
        self.assertEqual(row["failure_type"], "missing_configuration")
        self.assertEqual(row["status"], "open")
        self.assertIn("SEC_USER_AGENT is not configured", row["reason"])
        self.assertIn("CIK0000000001.json", row["source_url"])
        self.assertEqual(source_status, "unavailable")

    def test_resolve_primary_document_url_from_index(self) -> None:
        class FakeClient:
            def get_text(self, url: str, *, accept: str = "*/*") -> str:
                return """
                <html><body>
                  <a href="/Archives/edgar/data/123456/000012345626000001/0000123456-26-000001-index.htm">index</a>
                  <a href="/Archives/edgar/data/123456/000012345626000001/primary.htm">primary</a>
                </body></html>
                """

        url = resolve_primary_document_url(
            FakeClient(),  # type: ignore[arg-type]
            filing_url="https://www.sec.gov/Archives/edgar/data/123456/000012345626000001/0000123456-26-000001-index.htm",
            accession_number="0000123456-26-000001",
        )
        self.assertEqual(
            url,
            "https://www.sec.gov/Archives/edgar/data/123456/000012345626000001/primary.htm",
        )

    def test_enqueue_recent_filings_from_submissions_writes_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                filings = sec.list_recent_filings(
                    ticker="TEST",
                    cik="1",
                    submissions=SUBMISSIONS_SAMPLE,
                    forms={"4"},
                )
                count = sec.enqueue_recent_filings(conn, filings, source="SEC submissions")
                row = conn.execute(
                    """
                    SELECT ticker, cik, form, accession_number, document_url, source, status
                    FROM filing_queue
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertEqual(row["cik"], "0000000001")
        self.assertEqual(row["form"], "4")
        self.assertTrue(row["document_url"].endswith("/ownership.xml"))
        self.assertEqual(row["source"], "SEC submissions")
        self.assertEqual(row["status"], "queued")

    def test_enqueue_recent_filings_does_not_reset_processed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                filings = sec.list_recent_filings(
                    ticker="TEST",
                    cik="1",
                    submissions=SUBMISSIONS_SAMPLE,
                    forms={"10-K"},
                )
                sec.enqueue_recent_filings(conn, filings, source="SEC submissions")
                conn.execute("UPDATE filing_queue SET status = 'processed' WHERE ticker = 'TEST'")
                sec.enqueue_recent_filings(conn, filings, source="SEC submissions")
                row = conn.execute(
                    "SELECT status, document_url FROM filing_queue WHERE ticker = 'TEST'"
                ).fetchone()

        self.assertEqual(row["status"], "processed")
        self.assertTrue(row["document_url"].endswith("/test-20260331.htm"))


if __name__ == "__main__":
    unittest.main()
