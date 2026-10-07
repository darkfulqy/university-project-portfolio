from pathlib import Path
import csv
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.source_failures import (
    build_source_failure_report,
    record_source_failure,
    write_source_failure_csv,
)


class SourceFailureTests(unittest.TestCase):
    def test_fmp_402_failures_include_replacement_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                record_source_failure(
                    conn,
                    source_name="Financial Modeling Prep historical price API",
                    ticker="TEST",
                    endpoint="historical-price-eod",
                    reason="TEST: HTTP 402 while fetching masked FMP URL",
                )
                record_source_failure(
                    conn,
                    source_name="Financial Modeling Prep financial statements API",
                    ticker="TEST",
                    endpoint="income-statement",
                    reason="TEST: HTTP 402 while fetching masked FMP URL",
                )
                record_source_failure(
                    conn,
                    source_name="Financial Modeling Prep quote API",
                    ticker="TEST",
                    endpoint="quote",
                    reason="TEST: HTTP 402 while fetching masked FMP URL",
                )
                rows = build_source_failure_report(conn)

        by_endpoint = {row.endpoint: row for row in rows}
        self.assertEqual(
            by_endpoint["historical-price-eod"].replacement_action,
            "fetch_market_bars_from_fallback_or_local_source",
        )
        self.assertIn("fetch-yahoo-market-bars --ticker TEST", by_endpoint["historical-price-eod"].replacement_command)
        self.assertEqual(by_endpoint["income-statement"].replacement_action, "backfill_sec_companyfacts")
        self.assertIn("backfill-sec-companyfacts --ticker TEST", by_endpoint["income-statement"].replacement_command)
        self.assertIn("SEC_USER_AGENT", by_endpoint["income-statement"].replacement_notes)
        self.assertEqual(by_endpoint["quote"].replacement_action, "fetch_quote_fallback_or_import_manual_snapshot")
        self.assertIn("fetch-eastmoney-quote --ticker TEST", by_endpoint["quote"].replacement_command)

    def test_source_failure_csv_writes_replacement_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            output_path = Path(temp_dir) / "source_failures.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                record_source_failure(
                    conn,
                    source_name="SEC companyfacts",
                    ticker="TEST",
                    endpoint="companyfacts",
                    reason="SEC_USER_AGENT is not configured; skipped.",
                    failure_type="missing_configuration",
                )
                rows = build_source_failure_report(conn)
            write_source_failure_csv(rows, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertIn("replacement_action", csv_rows[0])
        self.assertEqual(csv_rows[0]["replacement_action"], "configure_sec_user_agent_then_backfill")
        self.assertIn("backfill-sec-companyfacts --ticker TEST", csv_rows[0]["replacement_command"])


if __name__ == "__main__":
    unittest.main()
