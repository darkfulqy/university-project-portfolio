from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit
from ai_stock_discovery.sources.peer_valuation import import_peer_valuation_csv
from ai_stock_discovery.watchlist import build_watchlist


class PeerValuationTests(unittest.TestCase):
    def test_import_peer_valuation_csv_feeds_score_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "peer_valuation.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,comparison_date,peer_group,peer_tickers,metric,target_value,peer_median,source_url,description,confidence",
                        "TEST,2026-05-31,AI infrastructure,PEER1;PEER2,pe,12,20,https://example.test/peer,Source-backed peer median comparison,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_peer_valuation_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                row = conn.execute(
                    """
                    SELECT metric, direction, ROUND(discount_premium_pct, 2) AS discount
                    FROM peer_valuation_comparisons
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'peer_valuation'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertEqual(row["metric"], "pe")
        self.assertEqual(row["direction"], "supports_discount")
        self.assertEqual(row["discount"], 0.4)
        self.assertGreater(score.valuation_score, 0)
        self.assertIn("Peer valuation context uses 1 source-backed comparison", score.notes[2])
        self.assertIn("not proof of undervaluation", score.notes[2])
        self.assertEqual(evidence["url"], "https://example.test/peer")

    def test_contradicting_peer_valuation_does_not_add_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "peer_valuation.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,peer_tickers,metric,target_value,peer_median,source_url,description,confidence",
                        "TEST,PEER1;PEER2,pe,30,20,https://example.test/peer,Target trades above peer median,0.9",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_peer_valuation_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                row = conn.execute(
                    "SELECT direction FROM peer_valuation_comparisons WHERE ticker = 'TEST'"
                ).fetchone()

        self.assertEqual(row["direction"], "contradicts_discount")
        self.assertEqual(score.valuation_score, 0)

    def test_peer_valuation_feeds_card_audit_and_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "peer_valuation.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,comparison_date,peer_group,peer_tickers,metric,target_value,peer_median,source_url,description,confidence",
                        "TEST,2026-05-31,AI infrastructure,PEER1;PEER2,ev_ebitda,8,12,https://example.test/peer,Source-backed peer discount comparison,0.7",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_peer_valuation_csv(conn, csv_path)
                card = generate_card(conn, "TEST", output_dir=None)
                audit_rows = build_evidence_audit(conn, tickers=["TEST"])
                watchlist_rows = build_watchlist(conn, limit=10)

        self.assertIn("## 同行估值比较", card)
        self.assertIn("peer_median=12", card)
        self.assertEqual(audit_rows[0].peer_valuation_count, 1)
        self.assertEqual(audit_rows[0].evidence_coverage_score, 15)
        self.assertEqual([row.ticker for row in watchlist_rows], ["TEST"])


if __name__ == "__main__":
    unittest.main()
