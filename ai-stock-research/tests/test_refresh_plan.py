from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.refresh_plan import build_refresh_plan, write_refresh_plan_csv
from ai_stock_discovery.source_failures import record_source_failure


class RefreshPlanTests(unittest.TestCase):
    def test_refresh_plan_prioritizes_missing_sources_and_ir_due(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "refresh_plan.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, security_type,
                        is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, cik, ir_url, source, updated_at)
                    VALUES ('TEST', 'Test Corp', '0000000001', 'https://example.test/ir', 'unit', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ir_pages (
                        ticker, url, page_type, source, status, last_checked_at
                    )
                    VALUES ('TEST', 'https://example.test/ir', 'ir_home', 'unit', 'active', '2026-05-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example/test',
                        'artificial intelligence', 'AI context needs business verification.', 3, 0.8, '2026-05-31'
                    )
                    """
                )
                rows = build_refresh_plan(conn, tickers=["TEST"], as_of="2026-06-01")
            write_refresh_plan_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        by_action = {row.action_type: row for row in rows}
        self.assertEqual(by_action["refresh_financial_facts"].due_status, "missing_source_evidence")
        self.assertIn("fetch-companyfacts --ticker TEST", by_action["refresh_financial_facts"].recommended_command)
        self.assertIn("refresh-ai-seed-data --ticker TEST", by_action["refresh_valuation_context"].recommended_command)
        self.assertIn("Financial Modeling Prep quote API", by_action["refresh_valuation_context"].source_name)
        self.assertEqual(by_action["refresh_market_confirmation"].due_status, "never_checked")
        self.assertIn("fetch-yahoo-market-bars --ticker TEST", by_action["refresh_market_confirmation"].recommended_command)
        self.assertEqual(by_action["check_ir_monitor"].due_status, "stale")
        self.assertEqual(by_action["check_ir_monitor"].days_since_observed, 31)
        self.assertIn("fetch-gdelt-doc-news", by_action["search_news_for_catalysts"].recommended_command)
        self.assertIn("fetch-news-rss --all-defaults", by_action["search_news_for_catalysts"].recommended_command)
        self.assertIn("Test Corp", by_action["search_news_for_catalysts"].recommended_command)
        self.assertIn("metadata is not a company-level conclusion", by_action["search_news_for_catalysts"].reason)
        self.assertIn(
            "fetch-finra-short-sale-volume --ticker TEST",
            by_action["refresh_short_sale_context"].recommended_command,
        )
        self.assertIn("not short interest", by_action["refresh_short_sale_context"].reason)
        self.assertIn("refresh planning only", by_action["check_ir_monitor"].plan_notes.lower())
        self.assertIn("action_type", text)
        self.assertIn("refresh_financial_facts", text)

    def test_refresh_plan_flags_stale_valuation_for_ready_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, cik, ir_url, source, updated_at)
                    VALUES ('TEST', 'Test Corp', '0000000001', 'https://example.test/ir', 'unit', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ir_pages (
                        ticker, url, page_type, source, status, last_checked_at
                    )
                    VALUES ('TEST', 'https://example.test/ir', 'ir_home', 'unit', 'active', '2026-05-30')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO data_source_status (source_name, status, reason, last_checked_at)
                    VALUES ('Financial Modeling Prep quote API', 'ok', 'unit status', '2026-06-01T00:00:00+00:00')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'https://sec.example/test',
                        'artificial intelligence', 'AI revenue and backlog context.', 4, 0.9, '2026-05-31'
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO financial_facts (
                        ticker, period, fiscal_year, fiscal_quarter, revenue,
                        gross_profit, operating_income, free_cash_flow, source, updated_at
                    )
                    VALUES ('TEST', ?, ?, 'Q1', ?, ?, ?, ?, 'https://sec.example/facts', '2026-05-31')
                    """,
                    [
                        ("2025-03-31", 2025, 100, 40, 10, 5),
                        ("2026-03-31", 2026, 140, 60, 20, 12),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO valuation_snapshots (
                        ticker, date, price, market_cap, enterprise_value, ev_sales, sector_percentile,
                        source, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-01', 10, 1000000000, 900000000, 3.2, 20,
                        'https://valuation.example/test', '2026-05-01'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO market_price_bars (
                        ticker, bar_date, close, volume, source_type, source_name,
                        source_path, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 10, 1000, 'unit',
                        'Yahoo Finance chart endpoint prototype',
                        'https://query1.finance.yahoo.com/chart/TEST',
                        'market-hash', '2026-05-31'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO expectation_gap_signals (
                        ticker, signal_date, source_type, source_name, source_url,
                        signal_type, direction, description, confidence, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'manual', 'unit', 'https://gap.example/test',
                        'legacy_market_label', 'supports_gap', 'Legacy label evidence.', 0.9, 'gap-hash', '2026-05-31'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO catalysts (
                        ticker, catalyst_type, catalyst_date, description, source_url,
                        confidence, status, updated_at
                    )
                    VALUES (
                        'TEST', 'earnings_or_filing', '2026-08-01', 'Source-backed earnings catalyst.',
                        'https://catalyst.example/test', 0.9, 'scheduled', '2026-05-31'
                    )
                    """
                )
                rows = build_refresh_plan(conn, tickers=["TEST"], as_of="2026-06-01")

        by_action = {row.action_type: row for row in rows}
        self.assertEqual(by_action["refresh_valuation_context"].due_status, "stale")
        self.assertEqual(by_action["refresh_valuation_context"].days_since_observed, 31)
        self.assertIn("ok", by_action["refresh_valuation_context"].source_status)
        self.assertNotIn("refresh_market_confirmation", by_action)
        self.assertNotIn("check_ir_monitor", by_action)

    def test_refresh_plan_recommends_ir_discovery_when_profile_website_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, website, source, updated_at
                    )
                    VALUES (
                        'TEST', 'Test Corp', 'https://www.test.test',
                        'https://financialmodelingprep.com/stable/profile?symbol=TEST&apikey=***',
                        '2026-06-01'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'FMP profile', 'https://profile.example/test',
                        'artificial intelligence', 'AI context needs business verification.', 3, 0.8, '2026-05-31'
                    )
                    """
                )
                rows = build_refresh_plan(conn, tickers=["TEST"], as_of="2026-06-02")

        by_action = {row.action_type: row for row in rows}
        self.assertIn("configure_ir_monitor", by_action)
        self.assertIn("discover-ir-pages --ticker TEST", by_action["configure_ir_monitor"].recommended_command)
        self.assertIn("source-backed company website", by_action["configure_ir_monitor"].reason)

    def test_refresh_plan_uses_source_failure_replacement_for_financial_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'FMP profile', 'https://profile.example/test',
                        'artificial intelligence', 'AI context needs business verification.', 3, 0.8, '2026-05-31'
                    )
                    """
                )
                record_source_failure(
                    conn,
                    source_name="Financial Modeling Prep financial statements API",
                    ticker="TEST",
                    endpoint="income-statement",
                    reason="TEST FMP financial statements: HTTP 402 while fetching masked URL",
                )
                rows = build_refresh_plan(conn, tickers=["TEST"], as_of="2026-06-02")

        by_action = {row.action_type: row for row in rows}
        self.assertIn("backfill-sec-companyfacts --ticker TEST", by_action["refresh_financial_facts"].recommended_command)
        self.assertIn("SEC_USER_AGENT", by_action["refresh_financial_facts"].reason)

    def test_refresh_plan_uses_source_failure_replacement_for_quote(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'FMP profile', 'https://profile.example/test',
                        'artificial intelligence', 'AI context needs business verification.', 3, 0.8, '2026-05-31'
                    )
                    """
                )
                record_source_failure(
                    conn,
                    source_name="Financial Modeling Prep quote API",
                    ticker="TEST",
                    endpoint="quote",
                    reason="TEST quote: HTTP 402 while fetching masked URL",
                )
                rows = build_refresh_plan(conn, tickers=["TEST"], as_of="2026-06-02")

        by_action = {row.action_type: row for row in rows}
        self.assertIn("fetch-eastmoney-quote --ticker TEST", by_action["refresh_valuation_context"].recommended_command)
        self.assertIn("fallback valuation price context", by_action["refresh_valuation_context"].reason)

    def test_refresh_plan_requires_primary_source_verification_for_external_research_lead(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('LEAD', 'Lead Systems Inc', 'SEC company_tickers', '2026-06-01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        source_type, source_name, url, published_at, fetched_at, raw_title, summary,
                        evidence_snippet, confidence, related_ticker, related_module, content_hash
                    )
                    VALUES (
                        'local_external_research_markdown', 'Local external research markdown',
                        'F:/stock/local.md#line=1', '2026-06-02', '2026-06-02T00:00:00+00:00',
                        'External lead', 'Lead Systems was mentioned in a local research brief.',
                        'External research lead only; verify primary sources.', 0.25,
                        'LEAD', 'external_research_monitor', 'local-research-lead'
                    )
                    """
                )
                rows = build_refresh_plan(conn, tickers=["LEAD"], as_of="2026-06-02")

        by_action = {row.action_type: row for row in rows}
        self.assertEqual(
            by_action["verify_external_research_lead"].due_status,
            "missing_primary_source_verification",
        )
        self.assertIn("enrich-fmp-profiles --ticker LEAD", by_action["verify_external_research_lead"].recommended_command)
        self.assertIn("Verify it with company profile", by_action["verify_external_research_lead"].reason)

    def test_refresh_plan_does_not_repeat_external_research_profile_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES (
                        'LEAD', 'Lead Systems Inc',
                        'https://financialmodelingprep.com/stable/profile?symbol=LEAD&apikey=***',
                        '2026-06-01'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        source_type, source_name, url, published_at, fetched_at, raw_title, summary,
                        evidence_snippet, confidence, related_ticker, related_module, content_hash
                    )
                    VALUES (
                        'local_external_research_markdown', 'Local external research markdown',
                        'F:/stock/local.md#line=1', '2026-06-02', '2026-06-02T00:00:00+00:00',
                        'External lead', 'Lead Systems was mentioned in a local research brief.',
                        'External research lead only; verify primary sources.', 0.25,
                        'LEAD', 'external_research_monitor', 'local-research-lead'
                    )
                    """
                )
                rows = build_refresh_plan(conn, tickers=["LEAD"], as_of="2026-06-02")

        self.assertNotIn("verify_external_research_lead", {row.action_type for row in rows})


if __name__ == "__main__":
    unittest.main()
