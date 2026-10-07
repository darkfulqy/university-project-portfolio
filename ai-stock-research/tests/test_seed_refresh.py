from pathlib import Path
from urllib.parse import parse_qs, urlparse
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.http import FetchError
from ai_stock_discovery.seed_refresh import refresh_ai_seed_data, select_ai_seed_tickers


class FakeSeedClient:
    def __init__(self, *, fail_quote: bool = False) -> None:
        self.json_urls: list[str] = []
        self.text_urls: list[str] = []
        self.fail_quote = fail_quote

    def get_json(self, url: str) -> object:
        self.json_urls.append(url)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        symbol = (params.get("symbol") or ["TEST"])[0]
        if "/stable/profile" in url:
            return [
                {
                    "symbol": symbol,
                    "companyName": f"{symbol} Corp",
                    "sector": "Technology",
                    "industry": "Semiconductors",
                    "marketCap": 123456,
                    "website": "https://example.test",
                    "description": "Supplier of data center GPUs and accelerated computing platforms.",
                }
            ]
        if "/stable/quote" in url:
            if self.fail_quote:
                raise FetchError("HTTP 402 for secret-key")
            return [
                {
                    "symbol": symbol,
                    "price": 12.5,
                    "marketCap": 123456,
                    "pe": 18,
                }
            ]
        if "/stable/analyst-estimates" in url:
            return [
                {
                    "symbol": symbol,
                    "date": "2026-12-31",
                    "revenueAvg": 1000,
                    "epsAvg": 1.2,
                    "numAnalystsRevenue": 2,
                    "numAnalystsEps": 4,
                }
            ]
        if "/stable/income-statement" in url:
            return [
                {
                    "symbol": symbol,
                    "date": "2026-03-31",
                    "fiscalYear": 2026,
                    "period": "Q1",
                    "revenue": 110,
                    "grossProfit": 50,
                    "operatingIncome": 20,
                    "netIncome": 13,
                    "epsDiluted": 1.3,
                    "weightedAverageShsOutDil": 10,
                },
                {
                    "symbol": symbol,
                    "date": "2025-03-31",
                    "fiscalYear": 2025,
                    "period": "Q1",
                    "revenue": 85,
                    "grossProfit": 40,
                    "operatingIncome": 15,
                    "netIncome": 9,
                    "epsDiluted": 0.9,
                    "weightedAverageShsOutDil": 10,
                },
            ]
        if "/stable/balance-sheet-statement" in url:
            return [
                {
                    "symbol": symbol,
                    "date": "2026-03-31",
                    "fiscalYear": 2026,
                    "period": "Q1",
                    "cashAndCashEquivalents": 1000,
                    "totalDebt": 400,
                },
                {
                    "symbol": symbol,
                    "date": "2025-03-31",
                    "fiscalYear": 2025,
                    "period": "Q1",
                    "cashAndCashEquivalents": 800,
                    "totalDebt": 500,
                },
            ]
        if "/stable/cash-flow-statement" in url:
            return [
                {
                    "symbol": symbol,
                    "date": "2026-03-31",
                    "fiscalYear": 2026,
                    "period": "Q1",
                    "operatingCashFlow": 30,
                    "capitalExpenditure": -8,
                    "freeCashFlow": 22,
                },
                {
                    "symbol": symbol,
                    "date": "2025-03-31",
                    "fiscalYear": 2025,
                    "period": "Q1",
                    "operatingCashFlow": 20,
                    "capitalExpenditure": -7,
                    "freeCashFlow": 13,
                },
            ]
        if "/api/xbrl/companyfacts/" in url:
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
                                        "val": 100,
                                    },
                                    {
                                        "end": "2025-03-31",
                                        "fy": 2025,
                                        "fp": "Q1",
                                        "form": "10-Q",
                                        "val": 80,
                                    },
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
                                        "val": 12,
                                    }
                                ]
                            }
                        },
                    }
                }
            }
        raise AssertionError(f"Unexpected JSON URL: {url}")

    def get_text(self, url: str, *, accept: str = "*/*") -> str:
        self.text_urls.append(url)
        return "\n".join(
            [
                "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                "20260529|TEST|400|10|1000|Q",
            ]
        )


class SeedRefreshTests(unittest.TestCase):
    def test_select_ai_seed_tickers_uses_existing_ai_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ai_industry_tags (
                        ticker, tag, confidence, source_type, source_url,
                        evidence_snippet, method, updated_at
                    )
                    VALUES ('TEST', 'ai_compute', 0.8, 'unit', 'https://example.test', 'GPU evidence', 'manual', 'now')
                    """
                )
                tickers = select_ai_seed_tickers(conn, limit=10)

        self.assertEqual(tickers, ["TEST"])

    def test_refresh_ai_seed_data_writes_source_backed_rows_and_redacts_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            client = FakeSeedClient()
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('TEST', '0000000001', 'Test Corp', 'unit', 'now')
                    """
                )
                result = refresh_ai_seed_data(
                    conn,
                    client,  # type: ignore[arg-type]
                    tickers=["TEST"],
                    fmp_api_key="secret-key",
                    analyst_limit=1,
                )
                counts = {
                    table: conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
                    for table in (
                        "company_profile",
                        "financial_facts",
                        "valuation_snapshots",
                        "analyst_estimate_events",
                        "short_sale_volume",
                        "ai_industry_tags",
                    )
                }
                source_urls = [
                    row["url"]
                    for row in conn.execute(
                        "SELECT url FROM evidence_items WHERE url LIKE '%financialmodelingprep%'"
                    ).fetchall()
                ]
                financial_sources = [
                    row["source"]
                    for row in conn.execute("SELECT source FROM financial_facts").fetchall()
                ]
                statuses = {
                    row["source_name"]: row["status"]
                    for row in conn.execute("SELECT source_name, status FROM data_source_status").fetchall()
                }

        self.assertEqual(result.tickers_considered, 1)
        self.assertEqual(result.financial_fact_periods, 4)
        self.assertEqual(result.profile_records, 1)
        self.assertGreaterEqual(result.valuation_snapshots, 1)
        self.assertGreaterEqual(result.analyst_events, 2)
        self.assertEqual(result.finra_records, 1)
        self.assertGreaterEqual(result.ai_tags_written, 2)
        self.assertEqual(counts["company_profile"], 1)
        self.assertEqual(counts["financial_facts"], 4)
        self.assertGreaterEqual(counts["valuation_snapshots"], 1)
        self.assertGreaterEqual(counts["analyst_estimate_events"], 2)
        self.assertEqual(counts["short_sale_volume"], 1)
        self.assertGreaterEqual(counts["ai_industry_tags"], 2)
        self.assertTrue(source_urls)
        self.assertFalse(any("secret-key" in url for url in source_urls))
        self.assertTrue(financial_sources)
        self.assertFalse(any("secret-key" in source for source in financial_sources))
        self.assertIn("secret-key", " ".join(client.json_urls))
        self.assertEqual(statuses["Financial Modeling Prep profile API"], "ok")
        self.assertEqual(statuses["Financial Modeling Prep quote API"], "ok")
        self.assertEqual(statuses["Financial Modeling Prep analyst estimates API"], "ok")
        self.assertEqual(statuses["Financial Modeling Prep financial statements API"], "ok")
        self.assertEqual(statuses["SEC companyfacts"], "ok")
        self.assertEqual(statuses["FINRA Daily Short Sale Volume"], "ok")

    def test_profile_valuation_fallback_does_not_mark_quote_source_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            client = FakeSeedClient(fail_quote=True)
            with open_db(db_path) as conn:
                result = refresh_ai_seed_data(
                    conn,
                    client,  # type: ignore[arg-type]
                    tickers=["TEST"],
                    fmp_api_key="secret-key",
                    include_analyst=False,
                    include_finra=False,
                    include_sec_facts=False,
                    include_fmp_financials=False,
                )
                statuses = {
                    row["source_name"]: row
                    for row in conn.execute("SELECT source_name, status, reason FROM data_source_status").fetchall()
                }
                failure_rows = conn.execute(
                    """
                    SELECT ticker, endpoint, failure_type, status, reason, source_url
                    FROM source_failures
                    WHERE source_name = 'Financial Modeling Prep quote API'
                    """
                ).fetchall()

        self.assertEqual(result.profile_records, 1)
        self.assertEqual(result.valuation_snapshots, 1)
        self.assertEqual(statuses["Financial Modeling Prep profile API"]["status"], "ok")
        self.assertEqual(statuses["Financial Modeling Prep quote API"]["status"], "degraded")
        self.assertIn("HTTP 402", statuses["Financial Modeling Prep quote API"]["reason"])
        self.assertNotIn("secret-key", statuses["Financial Modeling Prep quote API"]["reason"])
        self.assertEqual(len(failure_rows), 1)
        self.assertEqual(failure_rows[0]["ticker"], "TEST")
        self.assertEqual(failure_rows[0]["endpoint"], "quote")
        self.assertEqual(failure_rows[0]["failure_type"], "http_402")
        self.assertEqual(failure_rows[0]["status"], "open")
        self.assertIn("HTTP 402", failure_rows[0]["reason"])
        self.assertNotIn("secret-key", failure_rows[0]["reason"])
        self.assertIn("apikey=***", failure_rows[0]["source_url"])

    def test_successful_seed_refresh_resolves_prior_source_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            client = FakeSeedClient(fail_quote=True)
            with open_db(db_path) as conn:
                refresh_ai_seed_data(
                    conn,
                    client,  # type: ignore[arg-type]
                    tickers=["TEST"],
                    fmp_api_key="secret-key",
                    include_profile=False,
                    include_analyst=False,
                    include_finra=False,
                    include_sec_facts=False,
                    include_fmp_financials=False,
                )
                client.fail_quote = False
                refresh_ai_seed_data(
                    conn,
                    client,  # type: ignore[arg-type]
                    tickers=["TEST"],
                    fmp_api_key="secret-key",
                    include_profile=False,
                    include_analyst=False,
                    include_finra=False,
                    include_sec_facts=False,
                    include_fmp_financials=False,
                )
                row = conn.execute(
                    """
                    SELECT status, last_success_at
                    FROM source_failures
                    WHERE source_name = 'Financial Modeling Prep quote API'
                      AND ticker = 'TEST'
                      AND endpoint = 'quote'
                    """
                ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "resolved")
        self.assertTrue(row["last_success_at"])

    def test_sec_only_refresh_does_not_require_fmp_key_when_fmp_steps_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            client = FakeSeedClient()
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('TEST', '0000000001', 'Test Corp', 'unit', 'now')
                    """
                )
                result = refresh_ai_seed_data(
                    conn,
                    client,  # type: ignore[arg-type]
                    tickers=["TEST"],
                    fmp_api_key=None,
                    include_profile=False,
                    include_quote=False,
                    include_analyst=False,
                    include_finra=False,
                    include_fmp_financials=False,
                )

        self.assertEqual(result.financial_fact_periods, 2)
        self.assertEqual(result.failures, ())


if __name__ == "__main__":
    unittest.main()
