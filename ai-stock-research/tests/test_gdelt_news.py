from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_stock_discovery import pipeline
from ai_stock_discovery.cli import cmd_fetch_gdelt_doc_news
from ai_stock_discovery.config import Settings
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.pipeline import PipelineOptions
from ai_stock_discovery.sources import gdelt, news_rss


GDELT_PAYLOAD = {
    "articles": [
        {
            "title": "Test Corp (Nasdaq: TEST) expands AI data center monitoring platform",
            "url": "https://example.test/gdelt/test-ai",
            "seendate": "20260601T123000Z",
            "domain": "example.test",
            "language": "English",
            "sourcecountry": "United States",
        }
    ]
}


class GdeltNewsTests(unittest.TestCase):
    def test_parse_gdelt_doc_articles_and_store_news_search_evidence(self) -> None:
        events = gdelt.parse_gdelt_doc_articles(GDELT_PAYLOAD, limit=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_name, gdelt.GDELT_DOC_API_SOURCE_NAME)
        self.assertIn("review original article", events[0].summary)
        self.assertIn("domain=example.test", events[0].summary)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, security_type,
                        is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 'NASDAQ', 'common_stock', 0, 0, 0, 1, 'now')
                    """
                )
                result = news_rss.store_news_events(
                    conn,
                    events,
                    source_type="News Search",
                    content_hash_prefix="gdelt-doc-news",
                    mapped_confidence=0.6,
                    unmapped_confidence=0.35,
                )
                news_row = conn.execute(
                    "SELECT related_ticker FROM news_events WHERE url = 'https://example.test/gdelt/test-ai'"
                ).fetchone()
                evidence_row = conn.execute(
                    """
                    SELECT source_type, related_ticker, related_module, confidence, content_hash
                    FROM evidence_items
                    WHERE url = 'https://example.test/gdelt/test-ai'
                    """
                ).fetchone()

        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.mapped, 1)
        self.assertEqual(news_row["related_ticker"], "TEST")
        self.assertEqual(evidence_row["source_type"], "News Search")
        self.assertEqual(evidence_row["related_ticker"], "TEST")
        self.assertEqual(evidence_row["related_module"], "news_monitor")
        self.assertEqual(evidence_row["confidence"], 0.6)
        self.assertTrue(evidence_row["content_hash"].startswith("gdelt-doc-news:"))

    def test_fetch_gdelt_doc_news_builds_traceable_source_url(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                self.url = url
                return GDELT_PAYLOAD

        client = FakeClient()
        result = gdelt.fetch_gdelt_doc_news(
            client,  # type: ignore[arg-type]
            query='"Test Corp" AI',
            limit=5,
            timespan="3d",
        )

        self.assertEqual(len(result.events), 1)
        self.assertIn("query=%22Test+Corp%22+AI", result.source_url)
        self.assertIn("maxrecords=5", result.source_url)
        self.assertIn("timespan=3d", result.source_url)
        self.assertEqual(client.url, result.source_url)

    def test_cli_gdelt_failure_records_degraded_without_fake_news(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            settings = Settings(db_path=db_path, sec_user_agent="")
            args = Namespace(db=db_path, query="TEST AI", limit=5, timespan="7d")
            with patch("ai_stock_discovery.cli.gdelt.fetch_gdelt_doc_news", side_effect=RuntimeError("HTTP 429")):
                cmd_fetch_gdelt_doc_news(args, settings)
            with open_db(db_path) as conn:
                status = conn.execute(
                    """
                    SELECT status, reason
                    FROM data_source_status
                    WHERE source_name = ?
                    """,
                    (gdelt.GDELT_DOC_API_SOURCE_NAME,),
                ).fetchone()
                news_count = conn.execute("SELECT COUNT(*) AS count FROM news_events").fetchone()["count"]

        self.assertEqual(status["status"], "degraded")
        self.assertIn("HTTP 429", status["reason"])
        self.assertEqual(news_count, 0)

    def test_pipeline_fetch_gdelt_doc_news_uses_explicit_queries(self) -> None:
        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_json(self, url: str) -> object:
                return GDELT_PAYLOAD

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, security_type,
                        is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 'NASDAQ', 'common_stock', 0, 0, 0, 1, 'now')
                    """
                )
            original_client = pipeline.HttpClient
            pipeline.HttpClient = FakeClient  # type: ignore[assignment]
            try:
                count, message = pipeline._fetch_gdelt_doc_news(
                    db_path,
                    Settings(db_path=db_path, sec_user_agent=""),
                    PipelineOptions(gdelt_queries=('"Test Corp" AI',), gdelt_limit=5, gdelt_timespan="3d"),
                )
            finally:
                pipeline.HttpClient = original_client
            with open_db(db_path) as conn:
                status = conn.execute(
                    "SELECT status FROM data_source_status WHERE source_name = ?",
                    (gdelt.GDELT_DOC_API_SOURCE_NAME,),
                ).fetchone()["status"]
                event_count = conn.execute("SELECT COUNT(*) AS count FROM news_events").fetchone()["count"]

        self.assertEqual(count, 1)
        self.assertIn("1 GDELT DOC article metadata", message)
        self.assertEqual(status, "ok")
        self.assertEqual(event_count, 1)


if __name__ == "__main__":
    unittest.main()
