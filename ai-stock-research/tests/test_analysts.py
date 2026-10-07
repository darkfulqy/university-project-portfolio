from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.cli import cmd_fetch_fmp_analyst_events
from ai_stock_discovery.config import Settings
from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit
from ai_stock_discovery.sources.analysts import (
    FMP_ANALYST_ESTIMATES_SOURCE_NAME,
    fetch_fmp_analyst_estimate_events,
    import_analyst_estimate_events_csv,
    replace_fmp_analyst_estimate_events,
    upsert_analyst_estimate_events,
)
from ai_stock_discovery.watchlist import build_watchlist


class FakeJsonClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def get_json(self, url: str) -> object:
        self.urls.append(url)
        return self.payload


class AnalystEstimateEventTests(unittest.TestCase):
    def test_import_analyst_estimate_event_feeds_gap_score_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "analyst_events.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,event_date,fiscal_period,metric,event_type,direction,previous_value,current_value,unit,analyst_firm,source_url,description,confidence",
                        "TEST,2026-05-31,FY2026,revenue,estimate_lag,supports_gap,100,100,USDm,Example Research,https://example.test/estimate,Consensus revenue estimate has not changed after source-backed guidance improvement,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_analyst_estimate_events_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                row = conn.execute(
                    "SELECT event_type, direction, current_value FROM analyst_estimate_events WHERE ticker = 'TEST'"
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT related_module, url
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'analyst_estimate_event'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertEqual(row["event_type"], "estimate_lag")
        self.assertEqual(row["direction"], "supports_gap")
        self.assertEqual(row["current_value"], 100)
        self.assertGreater(score.expectation_gap_score, 0)
        self.assertIn("Analyst estimate context uses 1 source-backed event", score.notes[4])
        self.assertIn("analyst estimates are not inferred", score.notes[4])
        self.assertEqual(evidence["url"], "https://example.test/estimate")

    def test_contradicting_analyst_estimate_event_does_not_create_gap_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "analyst_events.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,event_date,metric,event_type,direction,source_url,description,confidence",
                        "TEST,2026-05-31,eps,estimate_revision,contradicts_gap,https://example.test/estimate,Source shows estimates were lowered after the filing,0.9",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_analyst_estimate_events_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")

        self.assertEqual(score.expectation_gap_score, 0)
        self.assertIn("1 contradicting", score.notes[4])

    def test_analyst_event_feeds_card_audit_and_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "analyst_events.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,event_date,metric,event_type,direction,source_url,description,confidence",
                        "TEST,2026-05-31,coverage,low_analyst_coverage,supports_gap,https://example.test/coverage,Only one source-backed analyst note found for the AI segment,0.7",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                import_analyst_estimate_events_csv(conn, csv_path)
                card = generate_card(conn, "TEST", output_dir=None)
                audit_rows = build_evidence_audit(conn, tickers=["TEST"])
                watchlist_rows = build_watchlist(conn, limit=10)

        self.assertIn("## 分析师预期/覆盖事件", card)
        self.assertIn("low_analyst_coverage", card)
        self.assertEqual(audit_rows[0].analyst_estimate_event_count, 1)
        self.assertEqual(audit_rows[0].evidence_coverage_score, 15)
        self.assertIn("ai_relevance_or_industry_tag", audit_rows[0].missing_core_evidence)
        self.assertEqual([row.ticker for row in watchlist_rows], ["TEST"])

    def test_fetch_fmp_analyst_estimates_redacts_key_and_creates_review_events(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "symbol": "TEST",
                    "date": "2026-12-31",
                    "revenueAvg": 123000000,
                    "epsAvg": 1.23,
                    "numAnalystsRevenue": 2,
                    "numAnalystsEps": 4,
                }
            ]
        )
        events = fetch_fmp_analyst_estimate_events(
            client,
            "test",
            "secret-key",
            period="annual",
            limit=3,
            low_coverage_threshold=3,
        )

        event_types = {event.event_type for event in events}
        self.assertIn("estimate_snapshot", event_types)
        self.assertIn("low_analyst_coverage", event_types)
        self.assertTrue(any(event.direction == "supports_gap" for event in events))
        self.assertTrue(all(event.source_name == FMP_ANALYST_ESTIMATES_SOURCE_NAME for event in events))
        self.assertTrue(all("apikey=%2A%2A%2A" in event.source_url for event in events))
        self.assertNotIn("secret-key", events[0].source_url)
        self.assertIn("secret-key", client.urls[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                upsert_analyst_estimate_events(conn, events)
                upsert_analyst_estimate_events(conn, events)
                row_count = conn.execute("SELECT COUNT(*) AS count FROM analyst_estimate_events").fetchone()["count"]
                score = score_ticker(conn, "TEST")

        self.assertEqual(row_count, len(events))
        self.assertGreater(score.expectation_gap_score, 0)

    def test_replace_fmp_analyst_estimates_removes_stale_fmp_rows_only(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "symbol": "TEST",
                    "date": "2026-12-31",
                    "revenueAvg": 2026,
                    "numAnalystsRevenue": 6,
                }
            ]
        )
        events = fetch_fmp_analyst_estimate_events(client, "TEST", "secret-key", limit=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO analyst_estimate_events (
                        ticker, event_date, fiscal_period, metric, event_type, direction,
                        previous_value, current_value, unit, analyst_firm, source_type,
                        source_name, source_url, description, confidence, content_hash,
                        updated_at
                    )
                    VALUES (
                        'TEST', NULL, '2030-12-31', 'revenue', 'estimate_snapshot',
                        'neutral', NULL, 2030, NULL, NULL,
                        'fmp_analyst_estimates_api',
                        'Financial Modeling Prep analyst estimates API',
                        'https://example.test/fmp',
                        'stale fmp row', 0.7, 'stale-fmp', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO analyst_estimate_events (
                        ticker, event_date, fiscal_period, metric, event_type, direction,
                        previous_value, current_value, unit, analyst_firm, source_type,
                        source_name, source_url, description, confidence, content_hash,
                        updated_at
                    )
                    VALUES (
                        'TEST', NULL, '2030-12-31', 'coverage', 'low_analyst_coverage',
                        'supports_gap', NULL, 1, 'analyst_count', NULL,
                        'analyst_estimate_csv', 'manual.csv', 'https://example.test/manual',
                        'manual row', 0.7, 'manual-row', 'now'
                    )
                    """
                )

                imported = replace_fmp_analyst_estimate_events(conn, "TEST", events)
                fiscal_periods = {
                    row["fiscal_period"]
                    for row in conn.execute(
                        """
                        SELECT fiscal_period
                        FROM analyst_estimate_events
                        WHERE ticker = 'TEST'
                        ORDER BY fiscal_period
                        """
                    ).fetchall()
                }
                manual_count = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM analyst_estimate_events
                    WHERE source_type = 'analyst_estimate_csv'
                    """
                ).fetchone()["count"]

        self.assertEqual(imported, len(events))
        self.assertEqual(fiscal_periods, {"2026-12-31", "2030-12-31"})
        self.assertEqual(manual_count, 1)

    def test_fetch_fmp_analyst_estimates_prefers_nearest_future_periods(self) -> None:
        client = FakeJsonClient(
            [
                {
                    "symbol": "TEST",
                    "date": "2030-12-31",
                    "revenueAvg": 2030,
                    "numAnalystsRevenue": 1,
                },
                {
                    "symbol": "TEST",
                    "date": "2026-12-31",
                    "revenueAvg": 2026,
                    "numAnalystsRevenue": 6,
                },
                {
                    "symbol": "TEST",
                    "date": "2025-12-31",
                    "revenueAvg": 2025,
                    "numAnalystsRevenue": 6,
                },
            ]
        )

        events = fetch_fmp_analyst_estimate_events(
            client,
            "TEST",
            "secret-key",
            limit=1,
            low_coverage_threshold=3,
        )

        self.assertEqual({event.fiscal_period for event in events}, {"2026-12-31"})
        self.assertFalse(any(event.event_type == "low_analyst_coverage" for event in events))

    def test_fetch_fmp_analyst_events_without_key_records_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            settings = Settings(db_path=db_path, sec_user_agent="", fmp_api_key=None)
            args = Namespace(
                db=db_path,
                ticker=["TEST"],
                period="annual",
                limit=5,
                low_coverage_threshold=3,
            )

            cmd_fetch_fmp_analyst_events(args, settings)

            with open_db(db_path) as conn:
                source_status = conn.execute(
                    """
                    SELECT status, reason
                    FROM data_source_status
                    WHERE source_name = ?
                    """,
                    (FMP_ANALYST_ESTIMATES_SOURCE_NAME,),
                ).fetchone()
                event_count = conn.execute("SELECT COUNT(*) AS count FROM analyst_estimate_events").fetchone()[
                    "count"
                ]

        self.assertEqual(source_status["status"], "unavailable")
        self.assertIn("FMP_API_KEY is not configured", source_status["reason"])
        self.assertEqual(event_count, 0)


if __name__ == "__main__":
    unittest.main()
