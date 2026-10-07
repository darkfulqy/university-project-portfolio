from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.card_batch import build_research_cards, write_card_index_csv
from ai_stock_discovery.database import init_db, open_db


class CardBatchTests(unittest.TestCase):
    def test_build_research_cards_generates_markdown_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            cards_dir = base / "cards"
            index_path = base / "card_index.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, sector, industry, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'Technology', 'Software', 'unit', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example/test',
                        'artificial intelligence',
                        'Artificial intelligence revenue contribution was discussed.',
                        2, 0.6, '2026-05-31'
                    )
                    """
                )
                rows = build_research_cards(conn, tickers=["TEST"], output_dir=cards_dir)
            write_card_index_csv(rows, index_path)
            card_text = (cards_dir / "TEST.md").read_text(encoding="utf-8")
            index_text = index_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ticker, "TEST")
        self.assertEqual(rows[0].build_status, "generated")
        self.assertIn("not investment advice", rows[0].notes)
        self.assertIn("研究辅助，不是投资建议", card_text)
        self.assertIn("研究池与来源链状态", card_text)
        self.assertIn("Thesis gate: evidence_gap_no_company_thesis", card_text)
        self.assertIn("研究池状态: source_blocked", card_text)
        self.assertIn("下一步来源动作:", card_text)
        self.assertIn("AI 相关证据", card_text)
        self.assertIn("https://sec.example/test", card_text)
        self.assertIn("ticker,company_name", index_text)
        self.assertIn("TEST,Test Corp", index_text)


if __name__ == "__main__":
    unittest.main()
