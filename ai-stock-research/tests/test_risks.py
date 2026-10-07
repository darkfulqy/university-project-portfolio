from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.risks import import_risk_flags_csv


class RiskFlagTests(unittest.TestCase):
    def test_import_risk_flags_csv_feeds_penalty_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "risk_flags.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,risk_date,source_type,source_name,source_url,risk_type,severity,description,confidence,status",
                        "TEST,2026-05-31,SEC 10-K,unit filing,https://example.test/10-k,customer_concentration,high,Large customer concentration disclosed in risk factors,0.8,active",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_risk_flags_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                flag = conn.execute(
                    """
                    SELECT risk_type, severity
                    FROM risk_flags
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'risk_flag'
                    """
                ).fetchone()
        self.assertEqual(count, 1)
        self.assertLess(score.risk_penalty, 0)
        self.assertIn("customer_concentration", score.risk_summary)
        self.assertEqual(flag["severity"], "high")
        self.assertEqual(evidence["url"], "https://example.test/10-k")

    def test_resolved_risk_flag_does_not_penalize_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "risk_flags.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,source_url,risk_type,severity,description,confidence,status",
                        "TEST,https://example.test/8-k,litigation,critical,Resolved litigation disclosed by company,0.9,resolved",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_risk_flags_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
        self.assertEqual(score.risk_penalty, 0)


if __name__ == "__main__":
    unittest.main()
