from datetime import date
from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.http import FetchError
from ai_stock_discovery.sources.finra import (
    fetch_finra_daily_short_sale_volume,
    fetch_latest_finra_daily_short_sale_volume,
    finra_daily_short_sale_url,
    import_short_sale_volume_file,
    parse_short_sale_volume_file,
)


class FakeTextClient:
    def __init__(self, payload_by_url: dict[str, str]) -> None:
        self.payload_by_url = payload_by_url
        self.urls: list[str] = []

    def get_text(self, url: str, *, accept: str = "*/*") -> str:
        self.urls.append(url)
        if url not in self.payload_by_url:
            raise FetchError(f"HTTP 404 while fetching {url}")
        return self.payload_by_url[url]


class FinraShortSaleTests(unittest.TestCase):
    def test_parse_pipe_delimited_short_sale_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "short_sale.txt"
            file_path.write_text(
                "\n".join(
                    [
                        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                        "20260531|TEST|400|10|1000|Q",
                    ]
                ),
                encoding="utf-8",
            )
            records = parse_short_sale_volume_file(
                file_path,
                source_name="unit",
                source_url="https://example.test/finra",
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].ticker, "TEST")
        self.assertEqual(records[0].trade_date, "2026-05-31")
        self.assertAlmostEqual(records[0].short_volume_ratio or 0, 0.4)

    def test_fetch_finra_daily_file_filters_tickers_and_tracks_source_url(self) -> None:
        source_url = finra_daily_short_sale_url(trade_date=date(2026, 5, 29))
        client = FakeTextClient(
            {
                source_url: "\n".join(
                    [
                        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                        "20260529|NOK|400|10|1000|Q",
                        "20260529|AAPL|100|0|1000|Q",
                    ]
                )
            }
        )

        result = fetch_finra_daily_short_sale_volume(
            client,
            trade_date=date(2026, 5, 29),
            tickers=["NOK"],
        )

        self.assertEqual(result.trade_date, "2026-05-29")
        self.assertEqual(result.source_url, source_url)
        self.assertEqual(client.urls, [source_url])
        self.assertEqual([record.ticker for record in result.records], ["NOK"])
        self.assertEqual(result.records[0].source_url, source_url)

    def test_fetch_latest_finra_daily_file_skips_missing_dates(self) -> None:
        source_url = finra_daily_short_sale_url(trade_date=date(2026, 5, 29))
        client = FakeTextClient(
            {
                source_url: "\n".join(
                    [
                        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                        "20260529|NOK|400|10|1000|Q",
                    ]
                )
            }
        )

        result = fetch_latest_finra_daily_short_sale_volume(
            client,
            end_date=date(2026, 6, 2),
            lookback_days=5,
            tickers=["NOK"],
        )

        self.assertEqual(result.trade_date, "2026-05-29")
        self.assertEqual(len(client.urls), 5)
        self.assertEqual(client.urls[-1], source_url)

    def test_import_short_sale_volume_writes_evidence_with_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "short_sale.csv"
            file_path.write_text(
                "\n".join(
                    [
                        "trade_date,ticker,short_volume,short_exempt_volume,total_volume,market",
                        "2026-05-31,TEST,300,0,1000,Q",
                    ]
                ),
                encoding="utf-8",
            )
            db_path = temp_path / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = import_short_sale_volume_file(
                    conn,
                    file_path,
                    source_name="unit FINRA",
                    source_url="https://example.test/finra",
                )
                row = conn.execute(
                    """
                    SELECT short_volume_ratio
                    FROM short_sale_volume
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()
                evidence = conn.execute(
                    """
                    SELECT evidence_snippet
                    FROM evidence_items
                    WHERE related_ticker = 'TEST' AND related_module = 'short_sale_volume'
                    """
                ).fetchone()
        self.assertEqual(count, 1)
        self.assertAlmostEqual(row["short_volume_ratio"], 0.3)
        self.assertIn("not short interest", evidence["evidence_snippet"])


if __name__ == "__main__":
    unittest.main()
