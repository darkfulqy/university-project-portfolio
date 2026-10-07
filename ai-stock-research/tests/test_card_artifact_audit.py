from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.card_artifact_audit import (
    archive_card_artifacts,
    build_card_artifact_audit,
    write_card_artifact_archive_csv,
    write_card_artifact_audit_csv,
)
from ai_stock_discovery.database import init_db, open_db


class CardArtifactAuditTests(unittest.TestCase):
    def test_audit_flags_stale_excluded_and_missing_card_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            cards_dir = base / "cards"
            index_path = base / "card_index.csv"
            output_path = base / "card_artifact_audit.csv"
            cards_dir.mkdir()
            (cards_dir / "TEST.md").write_text("# TEST\n", encoding="utf-8")
            (cards_dir / "OLD.md").write_text("# OLD\n", encoding="utf-8")
            (cards_dir / "AAAU.md").write_text("# stale excluded card\n", encoding="utf-8")
            index_path.write_text(
                "ticker,card_path\n"
                f"TEST,{cards_dir / 'TEST.md'}\n"
                f"MISS,{cards_dir / 'MISS.md'}\n",
                encoding="utf-8",
            )

            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 0, 0, 0, 1, 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO universe_exclusions (
                        ticker, reason_type, reason, source_name, source_url,
                        severity, method, is_active, detected_at, content_hash
                    )
                    VALUES (
                        'AAAU', 'exchange_traded_fund', 'ETF profile name',
                        'SEC company_tickers', 'SEC company_tickers',
                        'exclude', 'rule', 1, 'now', 'aaau-exclusion'
                    )
                    """
                )
                rows = build_card_artifact_audit(
                    conn,
                    cards_dir=cards_dir,
                    card_index_path=index_path,
                    limit=10,
                )
            write_card_artifact_audit_csv(rows, output_path)
            text = output_path.read_text(encoding="utf-8")

        by_ticker = {row.ticker: row for row in rows}
        self.assertEqual(by_ticker["TEST"].artifact_status, "current_indexed_card")
        self.assertTrue(by_ticker["TEST"].in_current_review_queue)
        self.assertEqual(by_ticker["MISS"].artifact_status, "indexed_missing_file")
        self.assertFalse(by_ticker["MISS"].file_exists)
        self.assertEqual(by_ticker["OLD"].artifact_status, "stale_unindexed_card")
        self.assertEqual(by_ticker["AAAU"].artifact_status, "stale_excluded_card")
        self.assertTrue(by_ticker["AAAU"].active_exclusion)
        self.assertEqual(by_ticker["AAAU"].exclusion_reason_types, "exchange_traded_fund")
        self.assertIn("no file was deleted", by_ticker["AAAU"].notes)
        self.assertIn("ticker,card_path,file_exists,indexed", text)
        self.assertIn("stale_excluded_card", text)

    def test_archive_card_artifacts_requires_apply_and_moves_only_generated_cards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            cards_dir = base / "cards"
            archive_dir = cards_dir / "_archive"
            index_path = base / "card_index.csv"
            plan_path = base / "card_artifact_archive_plan.csv"
            cards_dir.mkdir()
            current_card = cards_dir / "TEST.md"
            stale_card = cards_dir / "AAAU.md"
            current_card.write_text("# TEST\n", encoding="utf-8")
            stale_card.write_text("# stale excluded card\n", encoding="utf-8")
            index_path.write_text(
                "ticker,card_path\n"
                f"TEST,{current_card}\n",
                encoding="utf-8",
            )

            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 0, 0, 0, 1, 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO universe_exclusions (
                        ticker, reason_type, reason, source_name, source_url,
                        severity, method, is_active, detected_at, content_hash
                    )
                    VALUES (
                        'AAAU', 'exchange_traded_fund', 'ETF profile name',
                        'SEC company_tickers', 'SEC company_tickers',
                        'exclude', 'rule', 1, 'now', 'aaau-exclusion'
                    )
                    """
                )
                dry_run_rows = archive_card_artifacts(
                    conn,
                    cards_dir=cards_dir,
                    card_index_path=index_path,
                    archive_dir=archive_dir,
                    apply=False,
                )
                self.assertTrue(stale_card.exists())
                apply_rows = archive_card_artifacts(
                    conn,
                    cards_dir=cards_dir,
                    card_index_path=index_path,
                    archive_dir=archive_dir,
                    apply=True,
                )
                post_rows = build_card_artifact_audit(
                    conn,
                    cards_dir=cards_dir,
                    card_index_path=index_path,
                )
            write_card_artifact_archive_csv(apply_rows, plan_path)
            plan_text = plan_path.read_text(encoding="utf-8")
            stale_exists_after_apply = stale_card.exists()
            archive_exists_after_apply = Path(apply_rows[0].archive_path).exists()
            current_exists_after_apply = current_card.exists()

        self.assertEqual(len(dry_run_rows), 1)
        self.assertEqual(dry_run_rows[0].action_status, "planned_dry_run")
        self.assertEqual(len(apply_rows), 1)
        self.assertEqual(apply_rows[0].action_status, "archived")
        self.assertFalse(stale_exists_after_apply)
        self.assertTrue(archive_exists_after_apply)
        self.assertTrue(current_exists_after_apply)
        self.assertEqual([row.ticker for row in post_rows], ["TEST"])
        self.assertEqual(post_rows[0].artifact_status, "current_indexed_card")
        self.assertIn("ticker,artifact_status,source_path", plan_text)
        self.assertIn("archived", plan_text)


if __name__ == "__main__":
    unittest.main()
