from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.http import FetchError
from ai_stock_discovery.sources.valuation import (
    ValuationSnapshot,
    fetch_eastmoney_quote_snapshot,
    fetch_fmp_quote_snapshot,
    fetch_yahoo_snapshot,
    import_valuation_csv,
    upsert_valuation_snapshots,
)


class ValuationTests(unittest.TestCase):
    def test_import_valuation_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "valuation.csv"
            csv_path.write_text(
                "ticker,date,price,market_cap,pe,source\n"
                "TEST,2026-05-31,10.5,1000000,12.3,unit-source\n",
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_valuation_csv(conn, csv_path)
                row = conn.execute(
                    "SELECT price, market_cap, pe FROM valuation_snapshots WHERE ticker = 'TEST'"
                ).fetchone()
        self.assertEqual(count, 1)
        self.assertEqual(row["price"], 10.5)
        self.assertEqual(row["market_cap"], 1000000)
        self.assertEqual(row["pe"], 12.3)

    def test_valuation_score_uses_snapshot_fields_without_claiming_cheapness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                upsert_valuation_snapshots(
                    conn,
                    [
                        ValuationSnapshot(
                            ticker="TEST",
                            date="2026-05-31",
                            price=10,
                            market_cap=1000000,
                            pe=15,
                            source="unit",
                        )
                    ],
                )
                result = score_ticker(conn, "TEST")
        self.assertGreater(result.valuation_score, 0)
        self.assertIn("does not by itself prove undervaluation", result.notes[2])

    def test_valuation_score_uses_loaded_history_as_review_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                upsert_valuation_snapshots(
                    conn,
                    [
                        ValuationSnapshot(ticker="TEST", date="2026-01-31", pe=35, source="unit"),
                        ValuationSnapshot(ticker="TEST", date="2026-02-28", pe=30, source="unit"),
                        ValuationSnapshot(ticker="TEST", date="2026-03-31", pe=24, source="unit"),
                        ValuationSnapshot(ticker="TEST", date="2026-04-30", pe=18, source="unit"),
                        ValuationSnapshot(
                            ticker="TEST",
                            date="2026-05-31",
                            price=10,
                            market_cap=1000000,
                            pe=12,
                            sector_percentile=25,
                            source="unit",
                        ),
                    ],
                )
                result = score_ticker(conn, "TEST")

        self.assertGreaterEqual(result.valuation_score, 7)
        self.assertIn("loaded historical snapshot", result.notes[2])
        self.assertIn("sector_percentile", result.notes[2])
        self.assertIn("review input, not a valuation conclusion", result.notes[2])
        self.assertIn("does not by itself prove undervaluation", result.notes[2])

    def test_yahoo_snapshot_falls_back_to_chart_price(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def get_json(self, url: str) -> object:
                self.calls.append(url)
                if "/v7/finance/quote" in url:
                    raise FetchError("unauthorized")
                return {
                    "chart": {
                        "result": [
                            {
                                "meta": {
                                    "regularMarketPrice": 123.45,
                                }
                            }
                        ]
                    }
                }

        client = FakeClient()
        snapshot = fetch_yahoo_snapshot(client, "TEST")  # type: ignore[arg-type]
        self.assertEqual(snapshot.price, 123.45)
        self.assertIn("/v8/finance/chart/TEST", snapshot.source)
        self.assertEqual(len(client.calls), 2)

    def test_eastmoney_quote_snapshot_scales_public_quote_price(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                self.url = url
                return {
                    "data": {
                        "f43": 12345,
                        "f59": 2,
                        "f86": 0,
                    }
                }

        client = FakeClient()
        snapshot = fetch_eastmoney_quote_snapshot(client, "TEST")  # type: ignore[arg-type]
        self.assertEqual(snapshot.price, 123.45)
        self.assertEqual(snapshot.market_cap, None)
        self.assertIn("push2.eastmoney.com/api/qt/stock/get", snapshot.source)
        self.assertIn("secid=105.TEST", client.url)

    def test_fmp_snapshot_sanitizes_api_key_in_source(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                self.url = url
                return [{"price": 20, "marketCap": 2000000, "pe": 18}]

        client = FakeClient()
        snapshot = fetch_fmp_quote_snapshot(client, "TEST", "secret-key")  # type: ignore[arg-type]
        self.assertEqual(snapshot.price, 20)
        self.assertEqual(snapshot.market_cap, 2000000)
        self.assertNotIn("secret-key", snapshot.source)
        self.assertIn("apikey=***", snapshot.source)

    def test_fmp_snapshot_error_redacts_api_key(self) -> None:
        class FakeClient:
            def get_json(self, url: str) -> object:
                raise FetchError(f"HTTP 403 while fetching {url}")

        with self.assertRaises(FetchError) as raised:
            fetch_fmp_quote_snapshot(FakeClient(), "TEST", "secret-key")  # type: ignore[arg-type]

        self.assertNotIn("secret-key", str(raised.exception))
        self.assertIn("apikey=***", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
