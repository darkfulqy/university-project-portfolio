from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.catalysts import extract_catalysts_from_evidence, store_catalysts
from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit
from ai_stock_discovery.sources.transcripts import import_transcript_snippets_csv
from ai_stock_discovery.watchlist import build_watchlist


class TranscriptSnippetTests(unittest.TestCase):
    def test_import_transcript_snippet_feeds_ai_score_and_card(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "transcripts.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,transcript_date,fiscal_period,event_type,speaker,title,source_url,transcript_excerpt,confidence",
                        "TEST,2026-05-31,FY2026,earnings_call,CFO,Q1 call,https://example.test/transcript,Management said artificial intelligence data center demand is contributing to revenue and backlog but exact revenue contribution remains under review.,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                result = import_transcript_snippets_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                card = generate_card(conn, "TEST", output_dir=None)
                snippet_row = conn.execute(
                    "SELECT event_type, speaker FROM transcript_snippets WHERE ticker = 'TEST'"
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'transcript_snippet'
                    """
                ).fetchone()

        self.assertEqual(result.snippets_imported, 1)
        self.assertGreater(result.ai_signals_stored, 0)
        self.assertGreater(score.ai_relevance_score, 0)
        self.assertEqual(snippet_row["event_type"], "earnings_call")
        self.assertEqual(snippet_row["speaker"], "CFO")
        self.assertEqual(evidence["url"], "https://example.test/transcript")
        self.assertIn("## 电话会/纪要片段", card)
        self.assertIn("artificial intelligence data center demand", card)

    def test_transcript_evidence_can_feed_risk_and_catalyst_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "transcripts.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,transcript_date,source_url,transcript_excerpt,confidence",
                        "TEST,2026-05-31,https://example.test/transcript,Management said guidance was raised for FY2026 and AI order backlog improved but gross margin declined due to pricing pressure.,0.75",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                result = import_transcript_snippets_csv(conn, csv_path)
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")
                inserted = store_catalysts(conn, candidates)
                risk_rows = conn.execute(
                    "SELECT risk_type, status FROM risk_flags WHERE ticker = 'TEST'"
                ).fetchall()
                catalyst_types = {
                    row["catalyst_type"]
                    for row in conn.execute(
                        "SELECT catalyst_type FROM catalysts WHERE ticker = 'TEST'"
                    ).fetchall()
                }

        self.assertGreater(result.risk_flags_stored, 0)
        self.assertIn("margin_deterioration", {row["risk_type"] for row in risk_rows})
        self.assertTrue(all(row["status"] == "watch" for row in risk_rows))
        self.assertGreater(inserted, 0)
        self.assertTrue({"guidance_change", "order_contract"}.intersection(catalyst_types))

    def test_transcript_only_ticker_enters_audit_and_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "transcripts.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,transcript_date,source_url,transcript_excerpt,confidence",
                        "TEST,2026-05-31,https://example.test/transcript,Management discussed quarterly operations without quantifying AI exposure.,0.6",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_transcript_snippets_csv(conn, csv_path, scan_ai=False, scan_risks=False)
                audit_rows = build_evidence_audit(conn, tickers=["TEST"])
                watchlist_rows = build_watchlist(conn, limit=10)

        self.assertEqual(audit_rows[0].transcript_snippet_count, 1)
        self.assertIn("ai_relevance_or_industry_tag", audit_rows[0].missing_core_evidence)
        self.assertEqual([row.ticker for row in watchlist_rows], ["TEST"])


if __name__ == "__main__":
    unittest.main()
