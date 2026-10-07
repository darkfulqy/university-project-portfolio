from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.keywords import KeywordSignal, store_keyword_signals
from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db


class ScoringTests(unittest.TestCase):
    def test_score_without_evidence_stays_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                result = score_ticker(conn, "TEST")
        self.assertEqual(result.score_total, 0)
        self.assertEqual(result.status, "排除或暂不跟踪")

    def test_score_uses_only_stored_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            signal = KeywordSignal(
                ticker="TEST",
                signal_date="2026-05-31",
                source_type="SEC 10-K",
                source_url="https://www.sec.gov/example",
                keyword="artificial intelligence",
                context_snippet="Artificial intelligence revenue remains early and unquantified.",
                ai_relevance_level=2,
                confidence=0.6,
            )
            with open_db(db_path) as conn:
                store_keyword_signals(conn, [signal])
                result = score_ticker(conn, "TEST")
        self.assertGreater(result.ai_relevance_score, 0)
        self.assertEqual(result.expectation_gap_score, 0)

    def test_fundamental_score_uses_comparable_periods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO financial_facts (
                        ticker, period, fiscal_year, fiscal_quarter, revenue,
                        gross_profit, operating_income, free_cash_flow, source, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("TEST", "2025-06-30", 2025, "Q2", 100, 40, 10, 5, "unit", "now"),
                        ("TEST", "2026-03-31", 2026, "Q1", 80, 30, 8, 2, "unit", "now"),
                        ("TEST", "2026-05-31", 2026, "Q2", 90, 35, 9, 3, "unit", "now"),
                        ("TEST", "2026-06-30", 2026, "Q2", 120, 50, 12, 6, "unit", "now"),
                    ],
                )
                result = score_ticker(conn, "TEST")
        self.assertGreater(result.fundamental_score, 0)
        self.assertIn("Q2 2025 to Q2 2026", result.notes[1])

    def test_risk_penalty_uses_sec_financial_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO financial_facts (
                        ticker, period, fiscal_year, fiscal_quarter, free_cash_flow,
                        cash, debt, source, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("TEST", "2026-06-30", 2026, "Q2", -10, 100, 250, "unit", "now"),
                )
                result = score_ticker(conn, "TEST")
        self.assertLess(result.risk_penalty, 0)
        self.assertIn("negative free cash flow", result.risk_summary)
        self.assertIn("debt is more than 2x cash", result.risk_summary)


if __name__ == "__main__":
    unittest.main()
