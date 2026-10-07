from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.universe_filters import (
    apply_universe_filters,
    list_universe_exclusions,
)
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.watchlist import build_watchlist


class UniverseFilterTests(unittest.TestCase):
    def test_spac_rule_is_traceable_and_excluded_from_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, security_type,
                        is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES (?, ?, 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, 'now')
                    """,
                    [
                        ("GOOD", "Good Compute Inc.",),
                        ("SPAC", "Example Acquisition Corp Class A",),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (?, '2026-05-31', 'SEC 10-K', 'https://sec.example',
                            'artificial intelligence', 'AI evidence.', 2, 0.6, 'now')
                    """,
                    [("GOOD",), ("SPAC",)],
                )

                result = apply_universe_filters(conn, min_market_cap=None)
                exclusions = list_universe_exclusions(conn, ticker="SPAC")
                rows = build_watchlist(conn, limit=10)

        self.assertEqual(result.recorded, 1)
        self.assertEqual(result.active_rule_exclusions, 1)
        self.assertEqual(exclusions[0]["reason_type"], "blank_check_or_spac")
        self.assertIn("Nasdaq Trader", exclusions[0]["source_name"])
        self.assertEqual([row.ticker for row in rows], ["GOOD"])

    def test_profile_name_excludes_profile_only_etf_and_spac_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, source, updated_at
                    )
                    VALUES (?, ?, 'Technology', 'Software', 'SEC company_tickers', 'now')
                    """,
                    [
                        ("GOOD", "Good Compute Inc."),
                        ("AAAU", "Goldman Sachs Physical Gold ETF"),
                        ("AACB", "Artius II Acquisition Inc."),
                    ],
                )

                result = apply_universe_filters(conn, min_market_cap=None)
                rows = build_watchlist(conn, limit=10)
                etf_exclusions = list_universe_exclusions(conn, ticker="AAAU")
                spac_exclusions = list_universe_exclusions(conn, ticker="AACB")

        self.assertEqual(result.recorded, 2)
        self.assertEqual(result.active_rule_exclusions, 2)
        self.assertEqual([row.ticker for row in rows], ["GOOD"])
        self.assertEqual(etf_exclusions[0]["reason_type"], "exchange_traded_fund")
        self.assertEqual(etf_exclusions[0]["source_name"], "SEC company_tickers")
        self.assertEqual(spac_exclusions[0]["reason_type"], "blank_check_or_spac")
        self.assertIn("acquisition inc", spac_exclusions[0]["reason"])

    def test_low_market_cap_uses_latest_valuation_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO valuation_snapshots (
                        ticker, date, price, market_cap, source, updated_at
                    )
                    VALUES (?, ?, 1.0, ?, ?, ?)
                    """,
                    [
                        ("MICR", "2026-05-01", 25_000_000, "manual_csv:valuation.csv", "1"),
                        ("BIG", "2026-05-01", 100_000_000, "manual_csv:valuation.csv", "1"),
                    ],
                )

                result = apply_universe_filters(conn, min_market_cap=50_000_000)
                micro_exclusions = list_universe_exclusions(conn, ticker="MICR")
                big_exclusions = list_universe_exclusions(conn, ticker="BIG")

        self.assertEqual(result.recorded, 1)
        self.assertEqual(micro_exclusions[0]["reason_type"], "microcap")
        self.assertIn("market_cap=2.5e+07", micro_exclusions[0]["reason"])
        self.assertEqual(big_exclusions, [])

    def test_stale_market_cap_exclusion_becomes_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO valuation_snapshots (
                        ticker, date, price, market_cap, source, updated_at
                    )
                    VALUES ('MICR', '2026-05-01', 1.0, 25000000, 'manual_csv:old.csv', '1')
                    """
                )
                apply_universe_filters(conn, min_market_cap=50_000_000)
                self.assertEqual(len(list_universe_exclusions(conn, ticker="MICR")), 1)

                conn.execute(
                    """
                    INSERT INTO valuation_snapshots (
                        ticker, date, price, market_cap, source, updated_at
                    )
                    VALUES ('MICR', '2026-05-02', 2.0, 100000000, 'manual_csv:new.csv', '2')
                    """
                )
                apply_universe_filters(conn, min_market_cap=50_000_000)
                active = list_universe_exclusions(conn, ticker="MICR")
                all_rows = list_universe_exclusions(conn, ticker="MICR", active_only=False)

        self.assertEqual(active, [])
        self.assertEqual(len(all_rows), 1)
        self.assertEqual(int(all_rows[0]["is_active"]), 0)


if __name__ == "__main__":
    unittest.main()
