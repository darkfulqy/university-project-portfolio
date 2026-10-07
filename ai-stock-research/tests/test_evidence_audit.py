from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit, write_evidence_audit_csv


class EvidenceAuditTests(unittest.TestCase):
    def test_audit_flags_missing_core_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 0, 0, 0, 1, 'now')
                    """
                )
                rows = build_evidence_audit(conn, tickers=["TEST"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].review_priority, "needs_ai_evidence")
        self.assertIn("ai_relevance_or_industry_tag", rows[0].missing_core_evidence)
        self.assertEqual(rows[0].evidence_coverage_score, 0)

    def test_audit_counts_source_backed_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', 'now')
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
                        'artificial intelligence', 'AI revenue context.', 3, 0.7, 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO financial_facts (
                        ticker, period, fiscal_year, fiscal_quarter, revenue, source, updated_at
                    )
                    VALUES ('TEST', '2026-03-31', 2026, 'Q1', 100, 'https://sec.example/facts', 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO valuation_snapshots (
                        ticker, date, price, source, updated_at
                    )
                    VALUES ('TEST', '2026-05-31', 10, 'https://valuation.example/test', 'now')
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
                        'legacy_market_label', 'supports_gap', 'Legacy label evidence.', 0.8, 'gap-hash', 'now'
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
                        'TEST', 'product_launch', '2026-08-01', 'Source-backed launch.',
                        'https://catalyst.example/test', 0.8, 'candidate', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO risk_flags (
                        ticker, risk_date, source_type, source_name, source_url, risk_type,
                        severity, description, confidence, status, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'SEC 10-K', 'SEC', 'https://risk.example/test',
                        'customer_concentration', 'medium', 'Customer concentration.', 0.8,
                        'watch', 'risk-hash', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO insider_transactions (
                        ticker, owner_name, relationship, transaction_date, transaction_code,
                        acquired_disposed_code, transaction_shares, source_type, source_name,
                        source_url, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', 'Jane Doe', 'Director', '2026-05-31', 'S', 'D', 1000,
                        'SEC Form 4', 'SEC EDGAR', 'https://insider.example/test',
                        'insider-hash', 'now'
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO market_confirmation_signals (
                        ticker, signal_date, source_type, source_name, source_path,
                        signal_type, direction, description, confidence, content_hash, updated_at
                    )
                    VALUES (
                        'TEST', '2026-05-31', 'local', 'unit', 'local.csv',
                        'relative_strength', 'bullish', 'Source-backed confirmation.', 0.6,
                        'market-hash', 'now'
                    )
                    """
                )
                rows = build_evidence_audit(conn, tickers=["TEST"])

        self.assertEqual(rows[0].review_priority, "ready_for_human_review")
        self.assertEqual(rows[0].ai_signal_count, 1)
        self.assertEqual(rows[0].financial_period_count, 1)
        self.assertEqual(rows[0].valuation_snapshot_count, 1)
        self.assertEqual(rows[0].expectation_gap_count, 1)
        self.assertEqual(rows[0].catalyst_count, 1)
        self.assertEqual(rows[0].risk_flag_count, 1)
        self.assertEqual(rows[0].insider_transaction_count, 1)
        self.assertEqual(rows[0].market_confirmation_count, 1)
        self.assertEqual(rows[0].evidence_coverage_score, 100)
        self.assertEqual(rows[0].missing_core_evidence, "")
        self.assertGreaterEqual(rows[0].source_link_count, 7)

    def test_write_evidence_audit_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "audit.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('TEST', 'Test Corp', 'unit', 'now')
                    """
                )
                rows = build_evidence_audit(conn, tickers=["TEST"])
            write_evidence_audit_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("ticker,company_name", text)
        self.assertIn("TEST,Test Corp", text)

    def test_candidate_pool_does_not_expand_to_all_profiles_when_universe_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('AAA', 'AAA Corp', 0, 0, 0, 1, 'now')
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES (?, ?, 'SEC company_tickers', 'now')
                    """,
                    [
                        ("AAA", "AAA Corp"),
                        ("NOEVID", "No Evidence Corp"),
                    ],
                )
                rows = build_evidence_audit(conn, limit=20)

        self.assertEqual([row.ticker for row in rows], ["AAA"])

    def test_external_research_leads_enter_candidate_pool_without_core_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('AAA', 'AAA Corp', 0, 0, 0, 1, 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, company_name, source, updated_at)
                    VALUES ('LEAD', 'Lead Systems Inc', 'SEC company_tickers', 'now')
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
                        'F:/stock/local.md#line=1', '2026-06-02', 'now',
                        'External lead', 'Lead Systems was mentioned in a local research brief.',
                        'External research lead only; verify primary sources.', 0.25,
                        'LEAD', 'external_research_monitor', 'local-research-lead'
                    )
                    """
                )
                rows = build_evidence_audit(conn, limit=20)

        by_ticker = {row.ticker: row for row in rows}
        self.assertIn("LEAD", by_ticker)
        self.assertEqual(by_ticker["LEAD"].review_priority, "needs_ai_evidence")
        self.assertEqual(by_ticker["LEAD"].evidence_item_count, 1)
        self.assertEqual(by_ticker["LEAD"].evidence_coverage_score, 0)


if __name__ == "__main__":
    unittest.main()
