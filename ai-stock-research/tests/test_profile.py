from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.http import FetchError
from ai_stock_discovery.sources.profile import (
    CompanyProfile,
    enrich_fmp_profile_candidates,
    fetch_fmp_profile,
    import_company_profiles_csv,
    select_ai_profile_enrichment_candidates,
    upsert_company_profiles,
)


class ProfileTests(unittest.TestCase):
    def test_import_company_profiles_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "profiles.csv"
            csv_path.write_text(
                "ticker,company_name,sector,industry,market_cap,website\n"
                "TEST,Test Corp,Technology,Semiconductors,123456,https://example.test\n",
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_company_profiles_csv(conn, csv_path)
                row = conn.execute(
                    "SELECT company_name, sector, industry, market_cap FROM company_profile WHERE ticker = 'TEST'"
                ).fetchone()
        self.assertEqual(count, 1)
        self.assertEqual(row["company_name"], "Test Corp")
        self.assertEqual(row["sector"], "Technology")
        self.assertEqual(row["industry"], "Semiconductors")
        self.assertEqual(row["market_cap"], 123456)

    def test_profile_upsert_preserves_existing_cik_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                upsert_company_profiles(
                    conn,
                    [CompanyProfile(ticker="TEST", cik="0000000001", company_name="Old")],
                )
                upsert_company_profiles(
                    conn,
                    [CompanyProfile(ticker="TEST", sector="Technology", company_name="New")],
                )
                row = conn.execute(
                    "SELECT cik, company_name, sector FROM company_profile WHERE ticker = 'TEST'"
                ).fetchone()
        self.assertEqual(row["cik"], "0000000001")
        self.assertEqual(row["company_name"], "New")
        self.assertEqual(row["sector"], "Technology")

    def test_fmp_profile_sanitizes_api_key(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                self.url = url
                return [
                    {
                        "companyName": "Test Corp",
                        "sector": "Technology",
                        "industry": "Software",
                        "mktCap": 1000,
                        "website": "https://example.test",
                    }
                ]

        client = FakeClient()
        result = fetch_fmp_profile(client, "TEST", "secret-key")  # type: ignore[arg-type]
        self.assertEqual(result.company_name, "Test Corp")
        self.assertEqual(result.market_cap, 1000)
        self.assertNotIn("secret-key", result.source)
        self.assertIn("apikey=***", result.source)

    def test_fmp_profile_error_redacts_api_key(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                raise FetchError(f"HTTP 403 while fetching {url}")

        with self.assertRaises(FetchError) as raised:
            fetch_fmp_profile(FakeClient(), "TEST", "secret-key")  # type: ignore[arg-type]

        self.assertNotIn("secret-key", str(raised.exception))
        self.assertIn("apikey=***", str(raised.exception))

    def test_select_ai_profile_enrichment_candidates_from_source_backed_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO company_profile (ticker, company_name, description, source, updated_at)
                    VALUES (?, ?, ?, 'SEC company_tickers', 'now')
                    """,
                    [
                        ("AOSL", "ALPHA & OMEGA SEMICONDUCTOR Ltd", None),
                        ("HUBC", "Hub Cyber Security Ltd.", None),
                        ("HUBCW", "Hub Cyber Security Ltd.", None),
                        ("SPAC", "Example Energy Acquisition Corp", None),
                        ("DONE", "Done Data Systems Inc", "Already has a description.",),
                        ("FOOD", "Food Retail Corp", None),
                    ],
                )
                candidates = select_ai_profile_enrichment_candidates(conn, limit=10)

        self.assertEqual([candidate.ticker for candidate in candidates], ["HUBC", "AOSL"])
        self.assertNotIn("HUBCW", {candidate.ticker for candidate in candidates})
        terms_by_ticker = {candidate.ticker: candidate.matched_terms for candidate in candidates}
        self.assertIn("semiconductor", terms_by_ticker["AOSL"])

    def test_enrich_fmp_profile_candidates_writes_profile_valuation_and_failures(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                if "FAIL" in url:
                    raise FetchError(f"HTTP 403 while fetching {url}")
                return [
                    {
                        "companyName": "Alpha & Omega Semiconductor Ltd",
                        "sector": "Technology",
                        "industry": "Semiconductors",
                        "marketCap": 123456789,
                        "description": "Supplies semiconductor power products.",
                        "website": "https://example.test",
                    }
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES (?, ?, 'SEC company_tickers', 'now')
                    """,
                    [
                        ("AOSL", "ALPHA & OMEGA SEMICONDUCTOR Ltd"),
                        ("FAIL", "FAIL SEMICONDUCTOR INC"),
                    ],
                )
                summary = enrich_fmp_profile_candidates(
                    conn,
                    FakeClient(),  # type: ignore[arg-type]
                    api_key="secret-key",
                    limit=10,
                )
                profile_row = conn.execute(
                    """
                    SELECT sector, industry, market_cap, description, source
                    FROM company_profile
                    WHERE ticker = 'AOSL'
                    """
                ).fetchone()
                valuation_row = conn.execute(
                    "SELECT market_cap, source FROM valuation_snapshots WHERE ticker = 'AOSL'"
                ).fetchone()
                failure_row = conn.execute(
                    """
                    SELECT reason, source_url
                    FROM source_failures
                    WHERE ticker = 'FAIL' AND endpoint = 'profile'
                    """
                ).fetchone()

        self.assertEqual(summary.candidates_considered, 2)
        self.assertEqual(summary.profiles_written, 1)
        self.assertEqual(summary.valuation_snapshots_written, 1)
        self.assertEqual(len(summary.failures), 1)
        self.assertEqual(profile_row["sector"], "Technology")
        self.assertEqual(profile_row["industry"], "Semiconductors")
        self.assertEqual(profile_row["market_cap"], 123456789)
        self.assertIn("apikey=***", profile_row["source"])
        self.assertEqual(valuation_row["market_cap"], 123456789)
        self.assertIn("apikey=***", valuation_row["source"])
        self.assertNotIn("secret-key", failure_row["reason"])
        self.assertIn("apikey=***", failure_row["source_url"])

    def test_explicit_profile_enrichment_tickers_do_not_require_name_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, description, source, updated_at
                    )
                    VALUES (
                        'NVDA', 'NVIDIA Corp', 'Existing profile text.',
                        'manual', 'now'
                    )
                    """
                )
                candidates = select_ai_profile_enrichment_candidates(
                    conn,
                    tickers=["NVDA", "NEWC"],
                    limit=10,
                )

        self.assertEqual([candidate.ticker for candidate in candidates], ["NEWC", "NVDA"])
        sources = {candidate.ticker: candidate.source for candidate in candidates}
        self.assertEqual(sources["NEWC"], "explicit_ticker")


if __name__ == "__main__":
    unittest.main()
