from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.portfolio import (
    build_portfolio_risk_report,
    import_portfolio_positions_csv,
    write_portfolio_risk_csv,
)


class PortfolioTests(unittest.TestCase):
    def test_import_positions_and_build_risk_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "positions.csv"
            output = base / "portfolio_risk.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "portfolio_name,ticker,position_date,market_value,source_name,source_path,notes",
                        "core,TEST,2026-06-01,70000,unit positions,positions.csv,AI watch position",
                        "core,SMOL,2026-06-01,30000,unit positions,positions.csv,smaller position",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, sector, industry, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'Technology', 'Semiconductors', 'unit', 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO risk_flags (
                        ticker, risk_date, source_type, source_name, source_url,
                        risk_type, severity, description, confidence, status,
                        content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-06-01', 'SEC 10-K', 'unit filing',
                        'https://example.test/risk', 'customer_concentration', 'high',
                        'Source-backed risk flag.', 0.8, 'active', 'risk-test', 'now'
                    )
                    """
                )
                imported = import_portfolio_positions_csv(conn, csv_path)
                rows = build_portfolio_risk_report(conn, portfolio_name="core")
            write_portfolio_risk_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(imported, 2)
        by_ticker = {row.ticker: row for row in rows}
        self.assertAlmostEqual(by_ticker["TEST"].weight_pct or 0, 70.0)
        self.assertEqual(by_ticker["TEST"].sector, "Technology")
        self.assertEqual(by_ticker["TEST"].active_risk_flag_count, 1)
        self.assertEqual(by_ticker["TEST"].concentration_note, "high_concentration_review_required")
        self.assertIn("evidence_gap", by_ticker["TEST"].evidence_note)
        self.assertIn("risk_flag", by_ticker["TEST"].risk_note)
        self.assertIn("not allocation advice", by_ticker["TEST"].report_notes)
        self.assertIn("portfolio_name,ticker", text)
        self.assertIn("TEST", text)

    def test_weight_column_accepts_fractional_percent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "positions.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,weight,source_path",
                        "TEST,0.125,positions.csv",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_portfolio_positions_csv(conn, csv_path, default_portfolio="watch")
                rows = build_portfolio_risk_report(conn, portfolio_name="watch")

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].weight_pct or 0, 12.5)
        self.assertEqual(rows[0].concentration_note, "elevated_position_weight_review")


if __name__ == "__main__":
    unittest.main()
