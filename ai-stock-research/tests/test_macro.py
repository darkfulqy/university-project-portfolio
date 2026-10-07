from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.config import Settings
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.pipeline import PipelineOptions, _fetch_fred_macro
from ai_stock_discovery.sources import macro


class MacroSourceTests(unittest.TestCase):
    def test_import_macro_csv_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "macro.csv"
            csv_path.write_text(
                "series_id,source_name,metric_name,category,geography,frequency,period,value,unit,source_url\n"
                "DGS10,FRED,10-Year Treasury,macro_rate,US,daily,2026-05-31,4.25,percent,https://fred.example/DGS10\n",
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = macro.import_macro_indicators_csv(conn, csv_path)
                row = macro.list_latest_macro_indicators(conn)[0]
                evidence = conn.execute(
                    """
                    SELECT related_module, related_ticker
                    FROM evidence_items
                    WHERE related_module = 'macro_context'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertEqual(row["series_id"], "DGS10")
        self.assertEqual(row["value"], 4.25)
        self.assertEqual(evidence["related_ticker"], None)

    def test_fetch_fred_observations_sanitizes_api_key(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                self.url = url
                return {
                    "observations": [
                        {"date": "2026-05-30", "value": "4.21"},
                        {"date": "2026-05-29", "value": "."},
                    ]
                }

        client = FakeClient()
        observations = macro.fetch_fred_observations(
            client,  # type: ignore[arg-type]
            api_key="secret",
            series_id="DGS10",
            metric_name="10-Year Treasury",
            limit=2,
        )

        self.assertEqual(len(observations), 1)
        self.assertIn("api_key=secret", client.url)
        self.assertNotIn("secret", observations[0].source_url)
        self.assertIn("api_key=%2A%2A%2A", observations[0].source_url)

    def test_fetch_fred_public_observations_parses_csv_and_is_idempotent(self) -> None:
        class FakeClient:
            def get_text(self, url: str, *, accept: str = "*/*") -> str:
                self.url = url
                self.accept = accept
                return (
                    "observation_date,DGS10\n"
                    "2026-05-27,4.20\n"
                    "2026-05-28,.\n"
                    "2026-05-29,4.25\n"
                )

        client = FakeClient()
        observations = macro.fetch_fred_public_observations(
            client,  # type: ignore[arg-type]
            series_id="DGS10",
            metric_name="10-Year Treasury",
            limit=2,
        )

        self.assertEqual([item.period for item in observations], ["2026-05-29", "2026-05-27"])
        self.assertEqual(observations[0].value, 4.25)
        self.assertEqual(observations[0].source_url, "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        self.assertIn("text/csv", client.accept)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                macro.upsert_macro_indicators(conn, observations)
                macro.upsert_macro_indicators(conn, observations)
                row_count = conn.execute("SELECT COUNT(*) AS count FROM macro_indicators").fetchone()["count"]
        self.assertEqual(row_count, 2)

    def test_pipeline_fred_macro_uses_public_csv_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            settings = Settings(db_path=db_path, sec_user_agent="", fred_api_key=None)
            options = PipelineOptions(macro_limit=2)

            def fake_public_observations(*_args, series_id: str, metric_name: str, category: str, **_kwargs):
                return [
                    macro.MacroIndicator(
                        series_id=series_id,
                        source_name="FRED",
                        metric_name=metric_name,
                        category=category,
                        geography="US",
                        frequency=None,
                        period="2026-05-29",
                        value=4.25,
                        unit=None,
                        source_url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                        content_hash=f"{series_id}-unit",
                    )
                ]

            with patch(
                "ai_stock_discovery.pipeline.macro.fetch_fred_public_observations",
                side_effect=fake_public_observations,
            ) as public_fetch:
                count, message = _fetch_fred_macro(db_path, settings, options)

            with open_db(db_path) as conn:
                source_status = conn.execute(
                    "SELECT status, reason FROM data_source_status WHERE source_name = 'FRED'"
                ).fetchone()
                row_count = conn.execute("SELECT COUNT(*) AS count FROM macro_indicators").fetchone()["count"]

        self.assertEqual(count, 2)
        self.assertEqual(row_count, 2)
        self.assertEqual(source_status["status"], "ok")
        self.assertIn("Public CSV", source_status["reason"])
        self.assertIn("Stored 2 FRED", message)
        self.assertEqual(public_fetch.call_count, 2)

    def test_fetch_eia_electricity_retail_sales_sanitizes_api_key(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                self.url = url
                return {
                    "response": {
                        "data": [
                            {
                                "period": "2026-03",
                                "sales": "12345",
                                "sales-units": "million kilowatthours",
                                "stateid": "US",
                                "sectorid": "ALL",
                            }
                        ]
                    }
                }

        client = FakeClient()
        observations = macro.fetch_eia_electricity_retail_sales(
            client,  # type: ignore[arg-type]
            api_key="secret",
            limit=1,
            stateid="US",
            sectorid="ALL",
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].category, "electricity")
        self.assertIn("api_key=secret", client.url)
        self.assertNotIn("secret", observations[0].source_url)

    def test_card_includes_macro_context_without_company_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                macro.upsert_macro_indicators(
                    conn,
                    [
                        macro.MacroIndicator(
                            series_id="DGS10",
                            source_name="FRED",
                            metric_name="10-Year Treasury",
                            category="macro_rate",
                            geography="US",
                            frequency="daily",
                            period="2026-05-31",
                            value=4.25,
                            unit="percent",
                            source_url="https://fred.example/DGS10",
                            content_hash="unit-hash",
                        )
                    ],
                )
                content = generate_card(conn, "TEST", output_dir=None)

        self.assertIn("宏观/电力背景", content)
        self.assertIn("DGS10", content)
        self.assertIn("不是公司级订单", content)


if __name__ == "__main__":
    unittest.main()
