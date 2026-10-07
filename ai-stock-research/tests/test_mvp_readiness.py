from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.mvp_readiness import build_mvp_readiness, write_mvp_readiness_csv
from ai_stock_discovery.sources import sec


class MvpReadinessTests(unittest.TestCase):
    def test_build_mvp_readiness_summarizes_sources_tables_pipeline_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reports_dir = base / "reports"
            reports_dir.mkdir()
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            (reports_dir / "watchlist.csv").write_text("ticker,score\nTEST,72\n", encoding="utf-8")
            (reports_dir / "ops_report.md").write_text("# ops\n", encoding="utf-8")
            cards_dir = reports_dir / "cards"
            cards_dir.mkdir()
            (cards_dir / "TEST.md").write_text("# TEST card\n", encoding="utf-8")
            (reports_dir / "card_index.csv").write_text(
                "ticker,card_path\nTEST,reports/cards/TEST.md\n",
                encoding="utf-8",
            )

            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                sec.mark_source_status(
                    conn,
                    source_name="Nasdaq Trader Symbol Directory",
                    status="ok",
                )
                sec.mark_source_status(
                    conn,
                    source_name="FRED",
                    status="unavailable",
                    reason="unit test missing optional key",
                )
                sec.mark_source_status(
                    conn,
                    source_name="Financial Modeling Prep historical price API",
                    status="ok",
                )
                cursor = conn.execute(
                    """
                    INSERT INTO pipeline_runs (
                        run_type, status, started_at, completed_at, options_json, summary
                    )
                    VALUES ('mvp_pipeline', 'success', '2026-06-01T00:00:00Z',
                            '2026-06-01T00:01:00Z', '{}', 'unit run')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO pipeline_steps (
                        run_id, step_name, status, started_at, completed_at,
                        records_changed, message, error
                    )
                    VALUES (?, 'build_watchlist', 'success', '2026-06-01T00:00:00Z',
                            '2026-06-01T00:01:00Z', 1, 'wrote watchlist', NULL)
                    """,
                    (cursor.lastrowid,),
                )
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 0, 0, 0, 1, 'now')
                    """
                )
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=reports_dir,
                    design_doc=design_doc,
                )

            by_area = {row.area: row for row in rows}
            self.assertEqual(by_area["research_boundary"].status, "ready")
            self.assertEqual(by_area["design_document"].status, "ready")
            self.assertEqual(by_area["schema"].status, "ready")
            self.assertEqual(by_area["latest_pipeline_run"].status, "ready")
            self.assertEqual(by_area["source_stock_universe"].status, "ready")
            self.assertEqual(by_area["source_market_confirmation"].status, "ready")
            self.assertEqual(by_area["source_macro"].status, "unavailable")
            self.assertEqual(by_area["table_universe"].status, "ready")
            self.assertEqual(by_area["table_financial_facts"].status, "needs_data")
            self.assertEqual(by_area["report_watchlist"].status, "ready")
            self.assertEqual(by_area["report_research_cards"].status, "ready")
            self.assertEqual(by_area["report_evidence_audit"].status, "needs_data")
            self.assertEqual(by_area["report_source_input_audit"].status, "needs_data")
            self.assertEqual(by_area["report_source_input_import"].status, "needs_data")
            self.assertEqual(by_area["report_source_input_plan"].status, "needs_data")

    def test_expectation_gap_table_readiness_accepts_analyst_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            db_path = base / "test.sqlite"
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
                        'TEST', NULL, 'FY2026', 'coverage', 'low_analyst_coverage',
                        'supports_gap', NULL, 2, 'analyst_count', NULL,
                        'unit', 'Unit analyst source', 'https://example.test/source',
                        'Source-backed analyst context only.', 0.7, 'unit-hash', 'now'
                    )
                    """
                )
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=base / "reports",
                    design_doc=design_doc,
                )

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["table_expectation_gap"].status, "ready")
        self.assertIn("analyst_estimate_events=1", by_area["table_expectation_gap"].evidence)

    def test_sec_mapping_readiness_uses_local_sec_company_tickers_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                sec.mark_source_status(
                    conn,
                    source_name="SEC company_tickers",
                    status="unavailable",
                    reason="SEC_USER_AGENT is not configured; SEC network step skipped.",
                )
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, cik, company_name, source, updated_at
                    )
                    VALUES ('TEST', '0000000001', 'Test Corp', 'SEC company_tickers', 'now')
                    """
                )
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=base / "reports",
                    design_doc=design_doc,
                )

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["source_sec_mapping"].status, "ready")
        self.assertIn("SEC company_tickers=ok", by_area["source_sec_mapping"].evidence)
        self.assertIn("Local source-backed SEC company_tickers rows=1", by_area["source_sec_mapping"].notes)

    def test_financial_fact_source_readiness_uses_local_source_backed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                sec.mark_source_status(
                    conn,
                    source_name="SEC companyfacts",
                    status="unavailable",
                    reason="SEC_USER_AGENT is not configured.",
                )
                sec.mark_source_status(
                    conn,
                    source_name="Financial Modeling Prep financial statements API",
                    status="degraded",
                    reason="HTTP 402 for some tickers.",
                )
                conn.execute(
                    """
                    INSERT INTO financial_facts (
                        ticker, period, fiscal_year, fiscal_quarter, revenue, source, updated_at
                    )
                    VALUES (
                        'TEST', '2026-03-31', 2026, 'Q1', 100,
                        'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json', 'now'
                    )
                    """
                )
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=base / "reports",
                    design_doc=design_doc,
                )

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["source_financial_facts"].status, "ready")
        self.assertIn("SEC companyfacts=ok", by_area["source_financial_facts"].evidence)
        self.assertIn("Local source-backed SEC companyfacts financial_facts rows=1", by_area["source_financial_facts"].notes)

    def test_sec_filing_trigger_readiness_uses_local_rss_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                sec.mark_source_status(
                    conn,
                    source_name="SEC latest filings Atom feed",
                    status="unavailable",
                    reason="SEC_USER_AGENT is not configured; SEC network step skipped.",
                )
                conn.execute(
                    """
                    INSERT INTO rss_filing_events (
                        title, form, filing_url, fetched_at, content_hash
                    )
                    VALUES (
                        '10-Q - Test Corp', '10-Q',
                        'https://www.sec.gov/Archives/test-index.html', 'now', 'rss-hash'
                    )
                    """
                )
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=base / "reports",
                    design_doc=design_doc,
                )

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["source_sec_filing_triggers"].status, "ready")
        self.assertIn("SEC latest filings Atom feed=ok", by_area["source_sec_filing_triggers"].evidence)
        self.assertIn("Local source-backed SEC latest filings Atom feed rows=1", by_area["source_sec_filing_triggers"].notes)

    def test_research_cards_report_degrades_when_directory_has_stale_cards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reports_dir = base / "reports"
            cards_dir = reports_dir / "cards"
            cards_dir.mkdir(parents=True)
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            (reports_dir / "card_index.csv").write_text(
                "ticker,card_path\nTEST,reports/cards/TEST.md\n",
                encoding="utf-8",
            )
            (cards_dir / "TEST.md").write_text("# TEST card\n", encoding="utf-8")
            (cards_dir / "OLD.md").write_text("# stale generated card\n", encoding="utf-8")

            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=reports_dir,
                    design_doc=design_doc,
                )

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["report_research_cards"].status, "degraded")
        self.assertIn("markdown_files=2", by_area["report_research_cards"].evidence)
        self.assertIn("indexed_cards=1", by_area["report_research_cards"].evidence)
        self.assertIn("stale_card_files=1", by_area["report_research_cards"].evidence)
        self.assertIn("OLD.MD", by_area["report_research_cards"].notes)
        self.assertIn("No files were deleted", by_area["report_research_cards"].notes)

    def test_write_mvp_readiness_csv_and_missing_design_doc_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "mvp_readiness.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=base / "reports",
                    design_doc=base / "missing.md",
                )
            write_mvp_readiness_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["design_document"].status, "blocked")
        self.assertIn("area,requirement,status,evidence,next_action,notes", text)
        self.assertIn("design_document", text)

    def test_empty_source_input_reports_are_ready_when_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            reports_dir = base / "reports"
            reports_dir.mkdir()
            design_doc = base / "ai_potential_stock_discovery_system.md"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            for filename in ("source_input_audit.csv", "source_input_import.csv", "source_input_plan.csv"):
                (reports_dir / filename).write_text("template_name,status\n", encoding="utf-8")
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                rows = build_mvp_readiness(
                    conn,
                    reports_dir=reports_dir,
                    design_doc=design_doc,
                )

        by_area = {row.area: row for row in rows}
        self.assertEqual(by_area["report_source_input_audit"].status, "ready")
        self.assertEqual(by_area["report_source_input_import"].status, "ready")
        self.assertEqual(by_area["report_source_input_plan"].status, "ready")
        self.assertIn("rows=0", by_area["report_source_input_plan"].evidence)


if __name__ == "__main__":
    unittest.main()
