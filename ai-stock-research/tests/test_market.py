from argparse import Namespace
from datetime import date
from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.analysis.scoring import score_ticker
from ai_stock_discovery.cli import cmd_fetch_fmp_market_bars
from ai_stock_discovery.config import Settings
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.market import (
    FMP_HISTORICAL_PRICE_SOURCE_NAME,
    YAHOO_CHART_SOURCE_NAME,
    detect_market_anomalies,
    detect_local_market_sources,
    fetch_fmp_market_price_bars,
    fetch_yahoo_market_price_bars,
    import_market_confirmation_csv,
    import_market_price_bars_csv,
    upsert_market_price_bars,
)


class FakeJsonClient:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = list(payloads)
        self.urls: list[str] = []

    def get_json(self, url: str) -> object:
        self.urls.append(url)
        if not self.payloads:
            raise AssertionError(f"Unexpected URL: {url}")
        return self.payloads.pop(0)


class MarketSourceTests(unittest.TestCase):
    def test_detect_local_market_sources_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            options_db = base_path / "options_flow.sqlite"
            options_db.write_text("placeholder", encoding="utf-8")
            sources = detect_local_market_sources(base_path)
        status_by_name = {source.source_name: source.exists for source in sources}
        self.assertTrue(status_by_name["Local options_flow.sqlite"])
        self.assertFalse(status_by_name["Local opening_confirmation_system.py"])

    def test_import_market_confirmation_csv_feeds_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "market.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,signal_date,source_type,source_name,source_path,signal_type,direction,magnitude,description,confidence",
                        "TEST,2026-05-31,options_flow,unit options,options_flow.sqlite,call_sweep,bullish,1.4,Unusual call sweep from local options flow database,0.8",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_market_confirmation_csv(conn, csv_path)
                score = score_ticker(conn, "TEST")
                evidence = conn.execute(
                    """
                    SELECT related_module, evidence_snippet
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'market_confirmation'
                    """
                ).fetchone()
        self.assertEqual(count, 1)
        self.assertGreater(score.market_confirmation_score, 0)
        self.assertEqual(evidence["related_module"], "market_confirmation")

    def test_detect_market_anomalies_from_local_price_bars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "price_bars.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "ticker,date,open,close,volume,benchmark_close,premarket_price,source_name,source_path",
                        "TEST,2026-05-28,10,10,100,100,,unit bars,price_bars.csv",
                        "TEST,2026-05-29,10,10,100,100,,unit bars,price_bars.csv",
                        "TEST,2026-05-30,10,10,100,100,,unit bars,price_bars.csv",
                        "TEST,2026-05-31,11,12,400,101,11.2,unit bars,price_bars.csv",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                imported = import_market_price_bars_csv(conn, csv_path)
                result = detect_market_anomalies(
                    conn,
                    lookback=3,
                    min_volume_multiple=2.0,
                    min_gap_pct=0.03,
                    min_relative_strength_pct=0.03,
                )
                signal_types = {
                    row["signal_type"]
                    for row in conn.execute(
                        """
                        SELECT signal_type
                        FROM market_confirmation_signals
                        WHERE ticker = 'TEST'
                        """
                    ).fetchall()
                }
                evidence = conn.execute(
                    """
                    SELECT evidence_snippet
                    FROM evidence_items
                    WHERE related_ticker = 'TEST'
                      AND related_module = 'market_confirmation'
                      AND raw_title = 'volume_spike'
                    """
                ).fetchone()
                score = score_ticker(conn, "TEST")

        self.assertEqual(imported, 4)
        self.assertEqual(result.tickers_checked, 1)
        self.assertGreaterEqual(result.signals_written, 3)
        self.assertIn("volume_spike", signal_types)
        self.assertIn("premarket_or_open_gap", signal_types)
        self.assertIn("relative_strength", signal_types)
        self.assertIn("moving_average_breakout", signal_types)
        self.assertIn("market-confirmation input only", evidence["evidence_snippet"])
        self.assertGreater(score.market_confirmation_score, 0)

    def test_fetch_fmp_market_price_bars_joins_benchmark_and_redacts_key(self) -> None:
        client = FakeJsonClient(
            [
                [
                    {"date": "2026-05-31", "open": 11, "high": 12.5, "low": 10.5, "close": 12, "volume": 400},
                    {"date": "2026-05-30", "open": 10, "high": 10.5, "low": 9.5, "close": 10, "volume": 100},
                ],
                [
                    {"date": "2026-05-31", "close": 101, "volume": 1000},
                    {"date": "2026-05-30", "close": 100, "volume": 900},
                ],
            ]
        )
        bars = fetch_fmp_market_price_bars(
            client,
            "test",
            "secret-key",
            benchmark_ticker="QQQ",
            lookback_days=5,
            end_date=date(2026, 6, 1),
        )

        self.assertEqual([bar.bar_date for bar in bars], ["2026-05-30", "2026-05-31"])
        self.assertEqual(bars[1].benchmark_close, 101)
        self.assertEqual(bars[1].source_name, FMP_HISTORICAL_PRICE_SOURCE_NAME)
        self.assertIn("apikey=***", bars[0].source_path)
        self.assertNotIn("secret-key", bars[0].source_path)
        self.assertIn("secret-key", client.urls[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                upsert_market_price_bars(conn, bars)
                upsert_market_price_bars(conn, bars)
                row_count = conn.execute("SELECT COUNT(*) AS count FROM market_price_bars").fetchone()["count"]
        self.assertEqual(row_count, 2)

    def test_fetch_fmp_market_price_bars_rejects_missing_required_fields(self) -> None:
        client = FakeJsonClient(
            [
                [{"date": "2026-05-31", "close": 12}],
                [{"date": "2026-05-31", "close": 101}],
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing volume"):
            fetch_fmp_market_price_bars(
                client,
                "TEST",
                "secret-key",
                end_date=date(2026, 6, 1),
            )

    def test_fetch_fmp_market_bars_without_key_records_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            settings = Settings(db_path=db_path, sec_user_agent="", fmp_api_key=None)
            args = Namespace(db=db_path, ticker=["TEST"], benchmark="QQQ", lookback_days=90)

            cmd_fetch_fmp_market_bars(args, settings)

            with open_db(db_path) as conn:
                source_status = conn.execute(
                    """
                    SELECT status, reason
                    FROM data_source_status
                    WHERE source_name = ?
                    """,
                    (FMP_HISTORICAL_PRICE_SOURCE_NAME,),
                ).fetchone()
                bar_count = conn.execute("SELECT COUNT(*) AS count FROM market_price_bars").fetchone()["count"]
                failure = conn.execute(
                    """
                    SELECT ticker, endpoint, failure_type, status, source_url
                    FROM source_failures
                    WHERE source_name = ?
                    """,
                    (FMP_HISTORICAL_PRICE_SOURCE_NAME,),
                ).fetchone()

        self.assertEqual(source_status["status"], "unavailable")
        self.assertIn("FMP_API_KEY is not configured", source_status["reason"])
        self.assertEqual(bar_count, 0)
        self.assertEqual(failure["ticker"], "TEST")
        self.assertEqual(failure["endpoint"], "historical-price-eod")
        self.assertEqual(failure["failure_type"], "missing_configuration")
        self.assertEqual(failure["status"], "open")
        self.assertIn("apikey=***", failure["source_url"])

    def test_fetch_yahoo_market_price_bars_joins_benchmark(self) -> None:
        client = FakeJsonClient(
            [
                _yahoo_chart_payload(
                    timestamps=[1780099200, 1780185600],
                    opens=[10.0, 11.0],
                    highs=[10.5, 12.5],
                    lows=[9.5, 10.5],
                    closes=[10.0, 12.0],
                    volumes=[100, 400],
                ),
                _yahoo_chart_payload(
                    timestamps=[1780099200, 1780185600],
                    opens=[99.0, 100.0],
                    highs=[100.0, 102.0],
                    lows=[98.0, 99.0],
                    closes=[100.0, 101.0],
                    volumes=[900, 1000],
                ),
            ]
        )
        bars = fetch_yahoo_market_price_bars(
            client,
            "test",
            benchmark_ticker="QQQ",
            lookback_days=5,
            end_date=date(2026, 6, 1),
        )

        self.assertEqual([bar.bar_date for bar in bars], ["2026-05-30", "2026-05-31"])
        self.assertEqual(bars[1].benchmark_close, 101.0)
        self.assertEqual(bars[1].source_name, YAHOO_CHART_SOURCE_NAME)
        self.assertIn("/v8/finance/chart/TEST", bars[0].source_path)
        self.assertIn("benchmark=QQQ:", bars[0].source_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                upsert_market_price_bars(conn, bars)
                upsert_market_price_bars(conn, bars)
                row_count = conn.execute("SELECT COUNT(*) AS count FROM market_price_bars").fetchone()["count"]
        self.assertEqual(row_count, 2)


def _yahoo_chart_payload(
    *,
    timestamps: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[int],
) -> dict[str, object]:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


if __name__ == "__main__":
    unittest.main()
