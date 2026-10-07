from pathlib import Path
from argparse import Namespace
import tempfile
import unittest
from unittest.mock import patch

from ai_stock_discovery.analysis.catalysts import (
    extract_catalysts_from_evidence,
    store_catalysts,
)
from ai_stock_discovery.cli import cmd_fetch_news_rss
from ai_stock_discovery.config import Settings
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.news_rss import (
    NewsEvent,
    NewsRssSource,
    parse_news_rss,
    store_news_events,
    ticker_news_rss_sources,
)


RSS_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Test Corp Announces New AI Data Center Contract</title>
      <link>https://example.test/news/1</link>
      <pubDate>Sun, 31 May 2026 12:00:00 GMT</pubDate>
      <description><![CDATA[Test Corp (NASDAQ: TEST) announced a new contract for AI data center capacity.]]></description>
    </item>
  </channel>
</rss>
"""


class NewsRssTests(unittest.TestCase):
    def test_parse_news_rss(self) -> None:
        events = parse_news_rss(RSS_SAMPLE, source_name="unit")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].title, "Test Corp Announces New AI Data Center Contract")
        self.assertIn("NASDAQ: TEST", events[0].summary)

    def test_store_news_events_maps_ticker_and_feeds_catalysts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )
                events = parse_news_rss(RSS_SAMPLE, source_name="unit")
                result = store_news_events(conn, events)
                candidates = extract_catalysts_from_evidence(conn, ticker="TEST")
                inserted = store_catalysts(conn, candidates)
                news_row = conn.execute(
                    "SELECT related_ticker FROM news_events WHERE url = 'https://example.test/news/1'"
                ).fetchone()
                evidence_row = conn.execute(
                    "SELECT related_ticker, related_module FROM evidence_items WHERE url = 'https://example.test/news/1'"
                ).fetchone()
        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.mapped, 1)
        self.assertEqual(news_row["related_ticker"], "TEST")
        self.assertEqual(evidence_row["related_module"], "news_monitor")
        self.assertGreaterEqual(inserted, 1)

    def test_exchange_ticker_must_exist_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                events = parse_news_rss(RSS_SAMPLE, source_name="unit")
                result = store_news_events(conn, events)
                news_row = conn.execute(
                    "SELECT related_ticker FROM news_events WHERE url = 'https://example.test/news/1'"
                ).fetchone()
        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.mapped, 0)
        self.assertIsNone(news_row["related_ticker"])

    def test_recomputed_mapping_clears_stale_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )
                events = parse_news_rss(RSS_SAMPLE, source_name="unit")
                first_result = store_news_events(conn, events)
                conn.execute("DELETE FROM universe WHERE ticker = 'TEST'")
                second_result = store_news_events(conn, events)
                news_row = conn.execute(
                    "SELECT related_ticker FROM news_events WHERE url = 'https://example.test/news/1'"
                ).fetchone()
                evidence_row = conn.execute(
                    "SELECT related_ticker FROM evidence_items WHERE url = 'https://example.test/news/1'"
                ).fetchone()
        self.assertEqual(first_result.mapped, 1)
        self.assertEqual(second_result.mapped, 0)
        self.assertIsNone(news_row["related_ticker"])
        self.assertIsNone(evidence_row["related_ticker"])

    def test_company_name_mapping_uses_local_universe(self) -> None:
        xml = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Test Systems announces AI data center expansion</title>
              <link>https://example.test/news/name-match</link>
              <description>Test Systems signed a new data center capacity agreement.</description>
            </item>
          </channel>
        </rss>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('TSYS', 'Test Systems Inc', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )
                events = parse_news_rss(xml, source_name="unit")
                result = store_news_events(conn, events)
        self.assertEqual(result.mapped, 1)

    def test_sec_company_tickers_profile_is_not_enough_when_universe_exists(self) -> None:
        xml = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Sample Trust announces update</title>
              <link>https://example.test/news/sec-profile-only</link>
              <description>Sample Trust (NASDAQ: AAAU) announced an update.</description>
            </item>
          </channel>
        </rss>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Systems Inc', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('AAAU', '0000000000', 'Sample Trust', 'SEC company_tickers', '2026-05-31')
                    """
                )
                events = parse_news_rss(xml, source_name="unit")
                result = store_news_events(conn, events)
        self.assertEqual(result.mapped, 0)

    def test_ambiguous_company_name_is_not_mapped(self) -> None:
        xml = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Acme Power announces AI data center expansion</title>
              <link>https://example.test/news/ambiguous</link>
              <description>Acme Power signed a capacity expansion agreement.</description>
            </item>
          </channel>
        </rss>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                for ticker in ("ACP", "ACPX"):
                    conn.execute(
                        """
                        INSERT INTO universe (
                            ticker, company_name, exchange, is_etf, is_preferred,
                            is_unit, is_active, last_checked_at
                        )
                        VALUES (?, 'Acme Power Inc', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                        """,
                        (ticker,),
                    )
                events = parse_news_rss(xml, source_name="unit")
                result = store_news_events(conn, events)
        self.assertEqual(result.mapped, 0)

    def test_company_name_mapping_requires_token_boundaries(self) -> None:
        xml = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Industry group announces new vice chair and board leadership</title>
              <link>https://example.test/news/chair-to</link>
              <description>The organization appointed a vice chair to its board.</description>
            </item>
          </channel>
        </rss>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('AIRT', 'Air T Inc', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )
                events = parse_news_rss(xml, source_name="unit")
                result = store_news_events(conn, events)
        self.assertEqual(result.mapped, 0)

    def test_ticker_scoped_rss_preserves_known_feed_ticker(self) -> None:
        xml = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Market update</title>
              <link>https://example.test/news/ticker-feed</link>
              <description>Headline returned from a ticker-scoped feed.</description>
            </item>
          </channel>
        </rss>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Systems Inc', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )
                events = parse_news_rss(xml, source_name="unit", related_ticker="TEST")
                result = store_news_events(conn, events)
                news_row = conn.execute(
                    "SELECT related_ticker FROM news_events WHERE url = 'https://example.test/news/ticker-feed'"
                ).fetchone()
        self.assertEqual(result.mapped, 1)
        self.assertEqual(news_row["related_ticker"], "TEST")

    def test_ticker_news_rss_sources_use_public_yahoo_and_nasdaq_feeds(self) -> None:
        sources = ticker_news_rss_sources("test")
        self.assertEqual([source.related_ticker for source in sources], ["TEST", "TEST"])
        self.assertIn("feeds.finance.yahoo.com", sources[0].url)
        self.assertIn("nasdaq.com/feed/rssoutbound", sources[1].url)

    def test_fetch_news_rss_all_defaults_records_per_source_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            settings = Settings(db_path=db_path, sec_user_agent="")
            args = Namespace(
                db=db_path,
                all_defaults=True,
                url="https://unused.example/rss",
                source_name="unused",
                limit=5,
            )
            sources = (
                NewsRssSource("Unit Good RSS", "https://example.test/good.rss"),
                NewsRssSource("Unit Broken RSS", "https://example.test/broken.rss"),
            )
            good_event = NewsEvent(
                source_name="Unit Good RSS",
                title="Test Corp Announces AI Infrastructure Update",
                url="https://example.test/news/good",
                published_at="2026-05-31",
                summary="Test Corp (NASDAQ: TEST) announced an update.",
                related_ticker=None,
                content_hash="unit-good-rss",
            )

            def fake_fetch(_client, *, url: str, source_name: str, limit: int, related_ticker=None):
                if source_name == "Unit Broken RSS":
                    raise RuntimeError("unit rss failure")
                return [good_event]

            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, is_etf, is_preferred,
                        is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 'NASDAQ', 0, 0, 0, 1, '2026-05-31')
                    """
                )

            with patch("ai_stock_discovery.cli.news_rss.DEFAULT_NEWS_RSS_SOURCES", sources), patch(
                "ai_stock_discovery.cli.news_rss.fetch_news_rss",
                side_effect=fake_fetch,
            ):
                cmd_fetch_news_rss(args, settings)

            with open_db(db_path) as conn:
                statuses = {
                    row["source_name"]: row
                    for row in conn.execute(
                        """
                        SELECT source_name, status, reason
                        FROM data_source_status
                        WHERE source_name IN ('Unit Good RSS', 'Unit Broken RSS')
                        """
                    ).fetchall()
                }
                news_count = conn.execute("SELECT COUNT(*) AS count FROM news_events").fetchone()["count"]
                mapped = conn.execute(
                    "SELECT related_ticker FROM news_events WHERE url = 'https://example.test/news/good'"
                ).fetchone()["related_ticker"]

        self.assertEqual(statuses["Unit Good RSS"]["status"], "ok")
        self.assertEqual(statuses["Unit Broken RSS"]["status"], "degraded")
        self.assertIn("unit rss failure", statuses["Unit Broken RSS"]["reason"])
        self.assertEqual(news_count, 1)
        self.assertEqual(mapped, "TEST")


if __name__ == "__main__":
    unittest.main()
