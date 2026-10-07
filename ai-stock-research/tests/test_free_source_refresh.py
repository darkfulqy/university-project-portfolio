from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.free_source_refresh import (
    refresh_free_source_context,
    select_free_source_refresh_tickers,
)
from ai_stock_discovery.sources.news_rss import NewsRssSource


RSS_SAMPLE = """
<rss><channel>
  <item>
    <title>Test Corp (Nasdaq: TEST) wins AI data center contract</title>
    <link>https://example.test/news/contract</link>
    <description>Test Corp announced a data center customer contract for AI workloads.</description>
    <pubDate>Mon, 01 Jun 2026 20:00 GMT</pubDate>
  </item>
  <item>
    <title>Test Corp (Nasdaq: TEST) discloses customer concentration risk</title>
    <link>https://example.test/news/risk</link>
    <description>Test Corp said a limited number of customers account for a substantial portion of revenue.</description>
    <pubDate>Mon, 01 Jun 2026 21:00 GMT</pubDate>
  </item>
</channel></rss>
"""


class FakeFreeSourceClient:
    def __init__(self) -> None:
        self.json_urls: list[str] = []
        self.text_urls: list[str] = []

    def get_text(self, url: str, *, accept: str = "*/*") -> str:
        self.text_urls.append(url)
        if "regsho/daily" in url:
            return "\n".join(
                [
                    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                    "20260601|TEST|450|0|1500|Q",
                ]
            )
        if "rss.example" in url:
            return RSS_SAMPLE
        raise AssertionError(f"Unexpected text URL: {url}")

    def get_json(self, url: str) -> object:
        self.json_urls.append(url)
        raise AssertionError("GDELT should not be called unless include_gdelt=True")


class FreeSourceRefreshTests(unittest.TestCase):
    def test_select_free_source_refresh_tickers_uses_refresh_plan_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_company_with_ai_evidence(conn)
                tickers = select_free_source_refresh_tickers(conn, limit=5)

        self.assertEqual(tickers, ["TEST"])

    def test_refresh_free_source_context_writes_finra_news_catalyst_and_risk_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            client = FakeFreeSourceClient()
            with open_db(db_path) as conn:
                _insert_company_with_ai_evidence(conn)
                with patch(
                    "ai_stock_discovery.free_source_refresh.news_rss.DEFAULT_NEWS_RSS_SOURCES",
                    (NewsRssSource("Unit RSS", "https://rss.example/feed"),),
                ):
                    result = refresh_free_source_context(
                        conn,
                        client,  # type: ignore[arg-type]
                        tickers=["TEST"],
                        include_gdelt=False,
                    )
                counts = {
                    table: conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
                    for table in ("short_sale_volume", "news_events", "catalysts", "risk_flags")
                }
                statuses = {
                    row["source_name"]: row["status"]
                    for row in conn.execute("SELECT source_name, status FROM data_source_status").fetchall()
                }
                evidence_source_types = [
                    row["source_type"]
                    for row in conn.execute(
                        "SELECT source_type FROM evidence_items WHERE related_ticker = 'TEST' ORDER BY source_type"
                    ).fetchall()
                ]

        self.assertEqual(result.tickers_considered, 1)
        self.assertEqual(result.finra_records, 1)
        self.assertEqual(result.news_events_seen, 2)
        self.assertEqual(result.news_events_mapped, 2)
        self.assertGreaterEqual(result.catalysts_inserted, 1)
        self.assertGreaterEqual(result.risk_flags_inserted, 1)
        self.assertEqual(result.failures, ())
        self.assertEqual(counts["short_sale_volume"], 1)
        self.assertEqual(counts["news_events"], 2)
        self.assertGreaterEqual(counts["catalysts"], 1)
        self.assertGreaterEqual(counts["risk_flags"], 1)
        self.assertEqual(statuses["FINRA Daily Short Sale Volume"], "ok")
        self.assertEqual(statuses["Unit RSS"], "ok")
        self.assertEqual(statuses["Automated catalyst extraction"], "ok")
        self.assertEqual(statuses["Automated risk text extraction"], "ok")
        self.assertIn("News RSS", evidence_source_types)
        self.assertEqual(client.json_urls, [])


def _insert_company_with_ai_evidence(conn) -> None:
    conn.execute(
        """
        INSERT INTO universe (
            ticker, company_name, exchange, security_type,
            is_etf, is_preferred, is_unit, is_active, last_checked_at
        )
        VALUES ('TEST', 'Test Corp', 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, '2026-06-01')
        """
    )
    conn.execute(
        """
        INSERT INTO company_profile (ticker, company_name, source, updated_at)
        VALUES ('TEST', 'Test Corp', 'unit', '2026-06-01')
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
            'artificial intelligence', 'AI context needs business verification.', 3, 0.8, '2026-05-31'
        )
        """
    )


if __name__ == "__main__":
    unittest.main()
