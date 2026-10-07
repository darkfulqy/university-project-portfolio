from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.catalyst_calendar import (
    FMP_EARNINGS_CALENDAR_SOURCE_NAME,
    fetch_fmp_earnings_calendar_entries,
    import_catalyst_calendar_csv,
    list_upcoming_catalysts,
    upsert_catalyst_calendar_entries,
)


class FakeFmpCalendarClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_json(self, url: str) -> object:
        self.urls.append(url)
        return [
            {
                "symbol": "TEST",
                "date": "2026-08-15",
                "epsActual": None,
                "epsEstimated": 1.2,
                "revenueActual": None,
                "revenueEstimated": 1000,
                "lastUpdated": "2026-06-01",
            },
            {
                "symbol": "OTHER",
                "date": "2026-08-20",
                "epsActual": None,
                "epsEstimated": 2.3,
                "revenueActual": None,
                "revenueEstimated": 2000,
            },
            {
                "symbol": "TEST",
                "date": "2026-11-15",
                "epsActual": None,
                "epsEstimated": 1.4,
                "revenueActual": None,
                "revenueEstimated": 1200,
            },
        ]


class CatalystCalendarTests(unittest.TestCase):
    def test_import_catalyst_calendar_feeds_score_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "catalysts.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,catalyst_type,catalyst_date,description,source_url,confidence,status",
                        "TEST,industry_conference,2026-06-15,Company scheduled to present AI data center product roadmap,https://example.test/event,0.8,scheduled",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                inserted = import_catalyst_calendar_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                upcoming = list_upcoming_catalysts(conn, ticker="TEST", from_date="2026-05-31", days=60)
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'catalyst_calendar'
                    """
                ).fetchone()
        self.assertEqual(inserted, 1)
        self.assertGreater(score.catalyst_score, 0)
        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming[0]["catalyst_type"], "industry_conference")
        self.assertEqual(evidence["url"], "https://example.test/event")

    def test_import_catalyst_calendar_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "catalysts.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,catalyst_type,catalyst_date,description,source_url,confidence",
                        "TEST,earnings_or_filing,2026-06-15,Expected earnings date from source,https://example.test/calendar,0.7",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                first = import_catalyst_calendar_csv(conn, csv_path)
                second = import_catalyst_calendar_csv(conn, csv_path)
                rows = conn.execute("SELECT COUNT(*) AS count FROM catalysts WHERE ticker = 'TEST'").fetchone()
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(rows["count"], 1)

    def test_fetch_fmp_earnings_calendar_filters_ticker_and_masks_key(self) -> None:
        client = FakeFmpCalendarClient()
        entries = fetch_fmp_earnings_calendar_entries(
            client,  # type: ignore[arg-type]
            ["TEST"],
            "secret-key",
            from_date="2026-06-01",
            to_date="2026-12-31",
            limit_per_ticker=1,
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].ticker, "TEST")
        self.assertEqual(entries[0].catalyst_date, "2026-08-15")
        self.assertEqual(entries[0].source_name, FMP_EARNINGS_CALENDAR_SOURCE_NAME)
        self.assertNotIn("secret-key", entries[0].source_url)
        self.assertIn("apikey=***", entries[0].source_url)
        self.assertIn("secret-key", client.urls[0])

    def test_fmp_earnings_calendar_entries_feed_catalysts_and_evidence(self) -> None:
        client = FakeFmpCalendarClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                entries = fetch_fmp_earnings_calendar_entries(
                    client,  # type: ignore[arg-type]
                    ["TEST"],
                    "secret-key",
                    from_date="2026-06-01",
                    to_date="2026-12-31",
                    limit_per_ticker=2,
                )
                inserted = upsert_catalyst_calendar_entries(conn, entries)
                evidence_sources = [
                    row["source_name"]
                    for row in conn.execute(
                        "SELECT source_name FROM evidence_items WHERE related_ticker = 'TEST'"
                    ).fetchall()
                ]

        self.assertEqual(inserted, 2)
        self.assertEqual(evidence_sources, [FMP_EARNINGS_CALENDAR_SOURCE_NAME, FMP_EARNINGS_CALENDAR_SOURCE_NAME])


if __name__ == "__main__":
    unittest.main()
