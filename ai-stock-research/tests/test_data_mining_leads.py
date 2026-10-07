from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.data_mining_leads import build_data_mining_leads, write_data_mining_leads_csv
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.source_failures import record_source_failure


class DataMiningLeadsTests(unittest.TestCase):
    def test_data_mining_leads_surface_source_backed_review_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "data_mining_leads.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_complete_source_chain(conn, ticker="TEST")
                rows = build_data_mining_leads(conn, tickers=["TEST"])
            write_data_mining_leads_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.ticker, "TEST")
        self.assertEqual(row.lead_type, "source_backed_review_lead")
        self.assertEqual(row.mining_priority, "P2")
        self.assertEqual(row.open_source_failures, 0)
        self.assertIn("ai_relevance=1", row.evidence_modules)
        self.assertIn("verify original sources", row.lead_notes)
        self.assertIn("ticker,company_name,score_total,lead_type", text)
        self.assertIn("TEST,Test Corp", text)

    def test_data_mining_leads_prioritize_p0_source_failure_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                _insert_sparse_candidate(conn, ticker="MISS")
                record_source_failure(
                    conn,
                    source_name="Financial Modeling Prep financial statements API",
                    ticker="MISS",
                    endpoint="financial-statements",
                    reason="MISS FMP financial statements: HTTP 402 while fetching masked URL",
                    source_url="https://financialmodelingprep.com/stable/income-statement?symbol=MISS&apikey=***",
                )
                rows = build_data_mining_leads(conn, tickers=["MISS"])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.ticker, "MISS")
        self.assertEqual(row.lead_type, "p0_source_failure_recovery")
        self.assertEqual(row.mining_priority, "P0")
        self.assertEqual(row.open_source_failures, 1)
        self.assertIn("P0:Financial Modeling Prep financial statements API:financial-statements:http_402", row.source_blockers)
        self.assertIn("backfill-sec-companyfacts --ticker MISS", row.recommended_commands)
        self.assertIn("no synthetic rows", row.lead_notes)


def _insert_sparse_candidate(conn, *, ticker: str) -> None:
    conn.execute(
        """
        INSERT INTO universe (
            ticker, company_name, exchange, security_type,
            is_etf, is_preferred, is_unit, is_active, last_checked_at
        )
        VALUES (?, 'Missing Sources Inc.', 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, '2026-06-01')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO ai_relevance_signals (
            ticker, signal_date, source_type, source_url, keyword,
            context_snippet, ai_relevance_level, confidence, updated_at
        )
        VALUES (
            ?, '2026-05-31', 'SEC 10-K', 'https://sec.example/miss',
            'artificial intelligence', 'AI context still needs primary-source validation.', 2, 0.6, '2026-05-31'
        )
        """,
        (ticker,),
    )


def _insert_complete_source_chain(conn, *, ticker: str) -> None:
    conn.execute(
        """
        INSERT INTO universe (
            ticker, company_name, exchange, security_type,
            is_etf, is_preferred, is_unit, is_active, last_checked_at
        )
        VALUES (?, 'Test Corp', 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, '2026-06-01')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO company_profile (
            ticker, company_name, sector, industry, source, updated_at
        )
        VALUES (?, 'Test Corp', 'Technology', 'AI Infrastructure', 'unit', '2026-06-01')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO ai_relevance_signals (
            ticker, signal_date, source_type, source_url, keyword,
            context_snippet, ai_relevance_level, confidence, updated_at
        )
        VALUES (
            ?, '2026-05-31', 'SEC 10-K', 'https://sec.example/test',
            'artificial intelligence', 'AI revenue and backlog context.', 4, 0.9, '2026-05-31'
        )
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO financial_facts (
            ticker, period, fiscal_year, fiscal_quarter, revenue,
            gross_profit, operating_income, free_cash_flow, source, updated_at
        )
        VALUES (?, '2026-03-31', 2026, 'Q1', 140, 60, 20, 12, 'https://sec.example/facts', '2026-05-31')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO valuation_snapshots (
            ticker, date, price, market_cap, enterprise_value, ev_sales, sector_percentile,
            source, updated_at
        )
        VALUES (?, '2026-05-31', 10, 1000000000, 900000000, 3.2, 25,
                'https://valuation.example/test', '2026-05-31')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO expectation_gap_signals (
            ticker, signal_date, source_type, source_name, source_url,
            signal_type, direction, description, confidence, content_hash, updated_at
        )
        VALUES (?, '2026-05-31', 'manual', 'unit', 'https://gap.example/test',
                'legacy_market_label', 'supports_gap', 'Legacy label evidence.', 0.9,
                'gap-hash', '2026-05-31')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO catalysts (
            ticker, catalyst_type, catalyst_date, description, source_url,
            confidence, status, updated_at
        )
        VALUES (?, 'earnings_or_filing', '2026-08-01', 'Source-backed earnings catalyst.',
                'https://catalyst.example/test', 0.9, 'scheduled', '2026-05-31')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO risk_flags (
            ticker, risk_date, source_type, source_name, source_url, risk_type,
            severity, description, confidence, status, content_hash, updated_at
        )
        VALUES (?, '2026-05-31', 'SEC 10-K', 'unit filing',
                'https://risk.example/test', 'customer_concentration', 'medium',
                'Source-backed risk flag.', 0.8, 'watch', 'risk-hash', '2026-05-31')
        """,
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at,
            raw_title, summary, evidence_snippet, confidence,
            related_ticker, related_module, content_hash
        )
        VALUES (
            'SEC 10-K', 'unit filing', 'https://sec.example/test', '2026-05-31', '2026-05-31',
            'Test filing', 'AI context', 'Source-backed AI evidence.', 0.9,
            ?, 'ai_relevance', 'evidence-hash'
        )
        """,
        (ticker,),
    )


if __name__ == "__main__":
    unittest.main()
