from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit
from ai_stock_discovery.sources.guidance import import_guidance_events_csv
from ai_stock_discovery.watchlist import build_watchlist


class GuidanceEventTests(unittest.TestCase):
    def test_import_guidance_events_csv_feeds_fundamental_score_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "guidance.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,guidance_date,fiscal_period,metric,previous_value,current_value,unit,direction,source_url,description,confidence",
                        "TEST,2026-05-31,FY2026,revenue,100,120,USDm,raised,https://example.test/guidance,Management raised revenue guidance from a source-backed release,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_guidance_events_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                row = conn.execute(
                    "SELECT metric, direction, current_value FROM guidance_events WHERE ticker = 'TEST'"
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'guidance_event'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertEqual(row["metric"], "revenue")
        self.assertEqual(row["direction"], "raised")
        self.assertEqual(row["current_value"], 120)
        self.assertGreater(score.fundamental_score, 0)
        self.assertIn("Guidance context uses 1 source-backed event", score.notes[1])
        self.assertIn("not treated as realized revenue", score.notes[1])
        self.assertEqual(evidence["url"], "https://example.test/guidance")

    def test_lowered_guidance_does_not_add_fundamental_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "guidance.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,metric,direction,source_url,description,confidence",
                        "TEST,revenue,lowered,https://example.test/guidance,Management lowered revenue guidance,0.9",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_guidance_events_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")

        self.assertEqual(score.fundamental_score, 0)
        self.assertIn("1 negative", score.notes[1])

    def test_guidance_feeds_card_audit_and_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "guidance.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,guidance_date,fiscal_period,metric,direction,source_url,description,confidence",
                        "TEST,2026-05-31,FY2026,backlog,raised,https://example.test/guidance,Management raised backlog outlook,0.7",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_guidance_events_csv(conn, csv_path)
                card = generate_card(conn, "TEST", output_dir=None)
                audit_rows = build_evidence_audit(conn, tickers=["TEST"])
                watchlist_rows = build_watchlist(conn, limit=10)

        self.assertIn("## 管理层指引事件", card)
        self.assertIn("metric=backlog", card)
        self.assertEqual(audit_rows[0].guidance_event_count, 1)
        self.assertEqual(audit_rows[0].evidence_coverage_score, 8)
        self.assertIn("sec_financial_facts", audit_rows[0].missing_core_evidence)
        self.assertEqual([row.ticker for row in watchlist_rows], ["TEST"])


if __name__ == "__main__":
    unittest.main()
