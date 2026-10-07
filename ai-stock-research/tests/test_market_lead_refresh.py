from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.market_lead_refresh import (
    refresh_yahoo_market_bars_from_leads,
    select_market_bar_fallback_tickers,
)
from ai_stock_discovery.source_failures import record_source_failure
from ai_stock_discovery.sources.market import (
    MarketPriceBar,
    YAHOO_CHART_SOURCE_NAME,
    YAHOO_CHART_SOURCE_TYPE,
)


class DummyClient:
    pass


class MarketLeadRefreshTests(unittest.TestCase):
    def test_select_market_bar_fallback_tickers_uses_data_mining_p0_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_candidate_with_market_failure(conn, ticker="HIST")
                _insert_candidate_without_market_failure(conn, ticker="OTHER")
                selected = select_market_bar_fallback_tickers(conn, limit=5)

        self.assertEqual(selected, ["HIST"])

    def test_refresh_yahoo_market_bars_from_leads_writes_bars_and_anomalies(self) -> None:
        def fake_fetch(client, ticker, *, benchmark_ticker="QQQ", lookback_days=90):
            return _bars_for(ticker)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_candidate_with_market_failure(conn, ticker="HIST")
                with patch(
                    "ai_stock_discovery.market_lead_refresh.market.fetch_yahoo_market_price_bars",
                    fake_fetch,
                ):
                    result = refresh_yahoo_market_bars_from_leads(
                        conn,
                        DummyClient(),  # type: ignore[arg-type]
                        tickers=["HIST"],
                        lookback_days=30,
                        anomaly_lookback=3,
                    )
                bar_count = conn.execute(
                    "SELECT COUNT(*) AS count FROM market_price_bars WHERE ticker = 'HIST'"
                ).fetchone()["count"]
                signal_count = conn.execute(
                    "SELECT COUNT(*) AS count FROM market_confirmation_signals WHERE ticker = 'HIST'"
                ).fetchone()["count"]
                yahoo_status = conn.execute(
                    """
                    SELECT status, reason
                    FROM data_source_status
                    WHERE source_name = ?
                    """,
                    (YAHOO_CHART_SOURCE_NAME,),
                ).fetchone()

        self.assertEqual(result.tickers_considered, 1)
        self.assertEqual(result.bars_written, 4)
        self.assertEqual(result.anomaly_tickers_checked, 1)
        self.assertGreaterEqual(result.anomaly_signals_written, 1)
        self.assertEqual(result.failures, ())
        self.assertEqual(bar_count, 4)
        self.assertGreaterEqual(signal_count, 1)
        self.assertEqual(yahoo_status["status"], "ok")
        self.assertIn("Prototype-only source used as FMP EOD fallback", yahoo_status["reason"])


def _insert_candidate_with_market_failure(conn, *, ticker: str) -> None:
    _insert_candidate_without_market_failure(conn, ticker=ticker)
    record_source_failure(
        conn,
        source_name="Financial Modeling Prep historical price API",
        ticker=ticker,
        endpoint="historical-price-eod",
        reason=f"{ticker}: HTTP 402 while fetching masked URL",
        source_url=f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&apikey=***",
    )


def _insert_candidate_without_market_failure(conn, *, ticker: str) -> None:
    conn.execute(
        """
        INSERT INTO universe (
            ticker, company_name, exchange, security_type,
            is_etf, is_preferred, is_unit, is_active, last_checked_at
        )
        VALUES (?, ?, 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, '2026-06-01')
        """,
        (ticker, f"{ticker} Corp"),
    )
    conn.execute(
        """
        INSERT INTO ai_relevance_signals (
            ticker, signal_date, source_type, source_url, keyword,
            context_snippet, ai_relevance_level, confidence, updated_at
        )
        VALUES (
            ?, '2026-05-31', 'SEC 10-K', 'https://sec.example/source',
            'artificial intelligence', 'Source-backed AI context for market lead triage.', 2, 0.6, '2026-05-31'
        )
        """,
        (ticker,),
    )


def _bars_for(ticker: str) -> list[MarketPriceBar]:
    rows = [
        ("2026-05-28", 10.0, 10.0, 100.0, 100.0),
        ("2026-05-29", 10.0, 10.0, 100.0, 100.0),
        ("2026-05-30", 10.0, 10.0, 100.0, 100.0),
        ("2026-05-31", 11.0, 12.0, 400.0, 101.0),
    ]
    return [
        MarketPriceBar(
            ticker=ticker,
            bar_date=bar_date,
            open=open_price,
            high=max(open_price, close),
            low=min(open_price, close),
            close=close,
            volume=volume,
            benchmark_close=benchmark_close,
            premarket_price=None,
            source_type=YAHOO_CHART_SOURCE_TYPE,
            source_name=YAHOO_CHART_SOURCE_NAME,
            source_path=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            content_hash=f"unit-yahoo-{ticker}-{bar_date}",
        )
        for bar_date, open_price, close, volume, benchmark_close in rows
    ]


if __name__ == "__main__":
    unittest.main()
