from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit
from ai_stock_discovery.sources.institutions import import_institutional_holding_events_csv
from ai_stock_discovery.watchlist import build_watchlist


class InstitutionalHoldingEventTests(unittest.TestCase):
    def test_import_institutional_holding_event_feeds_market_context_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "institutional.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,event_date,report_period,filing_type,institution_name,manager_cik,event_type,shares_held,shares_change,market_value,source_url,description,confidence",
                        "TEST,2026-05-31,2026Q1,13F-HR,Example Capital,0001234567,new_position,100000,100000,2500000,https://sec.example/13f,Source-backed 13F shows a new reported position with disclosure lag,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_institutional_holding_events_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                row = conn.execute(
                    "SELECT event_type, direction, shares_held FROM institutional_holding_events WHERE ticker = 'TEST'"
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'institutional_holding_event'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertEqual(row["event_type"], "new_position")
        self.assertEqual(row["direction"], "bullish")
        self.assertEqual(row["shares_held"], 100000)
        self.assertGreater(score.market_confirmation_score, 0)
        self.assertIn("Institutional holding context uses 1 source-backed event", score.notes[5])
        self.assertIn("not proof of informed buying or selling", score.notes[5])
        self.assertEqual(evidence["url"], "https://sec.example/13f")

    def test_bearish_institutional_event_does_not_add_market_confirmation_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "institutional.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,event_date,filing_type,institution_name,event_type,source_url,description,confidence",
                        "TEST,2026-05-31,13F-HR,Example Capital,exited_position,https://sec.example/13f,Source-backed 13F shows the manager no longer reports a position,0.9",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_institutional_holding_events_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")

        self.assertEqual(score.market_confirmation_score, 0)
        self.assertIn("1 bearish", score.notes[5])

    def test_institutional_event_feeds_card_audit_and_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "institutional.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,event_date,report_period,filing_type,institution_name,event_type,source_url,description,confidence",
                        "TEST,2026-05-31,2026Q1,13D,Example Activist,activist_stake,https://sec.example/13d,Source-backed Schedule 13D indicates a reported activist stake,0.7",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_institutional_holding_events_csv(conn, csv_path)
                card = generate_card(conn, "TEST", output_dir=None)
                audit_rows = build_evidence_audit(conn, tickers=["TEST"])
                watchlist_rows = build_watchlist(conn, limit=10)

        self.assertIn("## 机构持仓变化事件", card)
        self.assertIn("Example Activist", card)
        self.assertEqual(audit_rows[0].institutional_holding_event_count, 1)
        self.assertEqual(audit_rows[0].evidence_coverage_score, 5)
        self.assertIn("ai_relevance_or_industry_tag", audit_rows[0].missing_core_evidence)
        self.assertEqual([row.ticker for row in watchlist_rows], ["TEST"])


if __name__ == "__main__":
    unittest.main()
