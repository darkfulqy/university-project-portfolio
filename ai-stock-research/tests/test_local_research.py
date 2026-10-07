from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.local_research import (
    LOCAL_RESEARCH_RELATED_MODULE,
    import_local_research_markdown,
    parse_local_research_markdown,
)


LOCAL_RESEARCH_MD = """# X stock blogger research - 2026-06-02

This is an idea-discovery brief, not investment advice. Treat X posts as leads for further verification.

### @unitresearch

Representative links:

- MRVL/NVDA catalyst post: https://x.com/unitresearch/status/123
- DRAM pinned thesis: https://x.com/unitresearch/status/456

Watchlist extracted:

- Primary: DRAM/MU theme, HPE, SMCI, MRVL, NVDA, AI infrastructure
- Adjacent: IBM, GFS, QBTS, RGTI

Suggested monitoring queries

- from:unitresearch ($MRVL OR $HPE OR $SMCI OR DRAM OR HBM OR NVDA)
"""


class LocalResearchTests(unittest.TestCase):
    def test_parse_local_research_markdown_extracts_known_ticker_leads_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "x_stock_blogger_research_2026-06-02.md"
            path.write_text(LOCAL_RESEARCH_MD, encoding="utf-8")
            leads = parse_local_research_markdown(
                LOCAL_RESEARCH_MD,
                path=path,
                known_tickers={"MRVL", "NVDA", "MU", "HPE", "SMCI", "IBM", "GFS", "QBTS", "RGTI", "AI"},
            )

        tickers = {lead.related_ticker for lead in leads}
        self.assertIn("MRVL", tickers)
        self.assertIn("NVDA", tickers)
        self.assertIn("MU", tickers)
        self.assertIn("HPE", tickers)
        self.assertNotIn("AI", tickers)
        self.assertFalse(any("Suggested monitoring queries" in lead.summary for lead in leads))
        self.assertTrue(all("external research lead only" in lead.evidence_snippet.lower() for lead in leads))

    def test_import_local_research_markdown_stores_evidence_without_core_scoring_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            md_path = base / "x_stock_blogger_research_2026-06-02.md"
            md_path.write_text(LOCAL_RESEARCH_MD, encoding="utf-8")
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES (?, ?, 'SEC company_tickers', 'now')
                    """,
                    [
                        ("MRVL", "Marvell Technology Inc"),
                        ("NVDA", "NVIDIA Corp"),
                        ("MU", "Micron Technology Inc"),
                        ("HPE", "Hewlett Packard Enterprise Company"),
                        ("SMCI", "Super Micro Computer Inc"),
                        ("AI", "C3.ai Inc"),
                    ],
                )
                result = import_local_research_markdown(conn, md_path)
                evidence_rows = conn.execute(
                    """
                    SELECT related_ticker, related_module, confidence, evidence_snippet
                    FROM evidence_items
                    WHERE related_module = ?
                    ORDER BY related_ticker
                    """,
                    (LOCAL_RESEARCH_RELATED_MODULE,),
                ).fetchall()
                ai_signal_count = conn.execute(
                    "SELECT COUNT(*) AS count FROM ai_relevance_signals"
                ).fetchone()["count"]

        self.assertGreaterEqual(result.leads_seen, 5)
        self.assertGreaterEqual(result.distinct_tickers, 5)
        self.assertEqual(ai_signal_count, 0)
        self.assertIn("MRVL", {row["related_ticker"] for row in evidence_rows})
        self.assertTrue(all(row["confidence"] <= 0.35 for row in evidence_rows))
        self.assertTrue(all("verify with SEC filings" in row["evidence_snippet"] for row in evidence_rows))


if __name__ == "__main__":
    unittest.main()
