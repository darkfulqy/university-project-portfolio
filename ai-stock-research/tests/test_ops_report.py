from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.ops_report import build_ops_report, write_ops_report
from ai_stock_discovery.sources import sec


class OpsReportTests(unittest.TestCase):
    def test_build_ops_report_summarizes_pipeline_sources_evidence_and_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            output = Path(temp_dir) / "ops_report.md"
            source_input_import = Path(temp_dir) / "source_input_import.csv"
            source_input_plan = Path(temp_dir) / "source_input_plan.csv"
            source_input_import.write_text(
                "template_name,audit_status,import_status,data_rows,rows_imported,recommended_action\n"
                "ai_industry_tags,ready_to_import,dry_run_ready,1,0,Rerun with --apply\n"
                "market_price_bars,ready_to_fill,not_ready,0,0,Fill rows\n",
                encoding="utf-8",
            )
            source_input_plan.write_text(
                "template_name,readiness_statuses,audit_status,import_status,"
                "recommended_next_action,audit_command\n"
                "ai_industry_tags,table_ai_chain_tags=needs_data,ready_to_import,"
                "dry_run_ready,Review the dry-run report,python -m ai_stock_discovery.cli build-source-input-audit --template ai_industry_tags\n"
                "market_price_bars,source_market_confirmation=unavailable;table_market_confirmation=needs_data,"
                "ready_to_fill,not_ready,Fill rows,python -m ai_stock_discovery.cli build-source-input-audit --template market_price_bars\n",
                encoding="utf-8",
            )
            init_db(db_path)
            with open_db(db_path) as conn:
                sec.mark_source_status(
                    conn,
                    source_name="SEC latest filings Atom feed",
                    status="degraded",
                    reason="unit test degraded source",
                )
                sec.mark_source_status(
                    conn,
                    source_name="Nasdaq Trader Symbol Directory",
                    status="ok",
                )
                cursor = conn.execute(
                    """
                    INSERT INTO pipeline_runs (
                        run_type, status, started_at, completed_at, options_json, summary
                    )
                    VALUES ('mvp_refresh', 'success', '2026-06-01T00:00:00Z',
                            '2026-06-01T00:01:00Z', '{}', 'unit run')
                    """
                )
                run_id = cursor.lastrowid
                conn.executemany(
                    """
                    INSERT INTO pipeline_steps (
                        run_id, step_name, status, started_at, completed_at,
                        records_changed, message, error
                    )
                    VALUES (?, ?, ?, '2026-06-01T00:00:00Z',
                            '2026-06-01T00:01:00Z', ?, ?, ?)
                    """,
                    [
                        (run_id, "fetch_sec_rss", "skipped", None, "skip network", None),
                        (run_id, "build_review_queue", "success", 1, "wrote queue", None),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO universe (
                        ticker, company_name, is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES ('TEST', 'Test Corp', 0, 0, 0, 1, 'now')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO evidence_items (
                        source_type, source_name, url, fetched_at, raw_title, summary,
                        evidence_snippet, confidence, related_ticker, related_module, content_hash
                    )
                    VALUES (
                        'rss', 'unit source', 'https://example.test/news',
                        '2026-06-01T00:00:00Z', 'Unit title', 'Unit summary',
                        'Source-backed unit evidence.', 0.8, 'TEST', 'news_monitor',
                        'ops-report-evidence'
                    )
                    """
                )
                report = build_ops_report(
                    conn,
                    review_limit=5,
                    source_input_import_path=source_input_import,
                    source_input_plan_path=source_input_plan,
                )
            write_ops_report(report, output)
            written = output.read_text(encoding="utf-8")

        self.assertIn("# AI Stock Discovery Operations Report", written)
        self.assertIn("Boundary: this report is for research operations only and is not investment advice.", written)
        self.assertIn("Run id: 1", written)
        self.assertIn("build_review_queue", written)
        self.assertIn("SEC latest filings Atom feed", written)
        self.assertIn("degraded", written)
        self.assertIn("news_monitor", written)
        self.assertIn("Source Input Import Plan", written)
        self.assertIn("dry_run_ready=1", written)
        self.assertIn("Rows ready for explicit --apply: 1", written)
        self.assertIn("Source Input Action Plan", written)
        self.assertIn("Audit status counts: ready_to_fill=1, ready_to_import=1", written)
        self.assertIn("Rows ready for explicit --apply after review: 1", written)
        self.assertIn("table_ai_chain_tags=needs_data", written)
        self.assertIn("--template ai_industry_tags", written)
        self.assertIn("ai_industry_tags", written)
        self.assertIn("TEST", written)
        self.assertIn("evidence_gap_review", written)
        self.assertIn("Verify original source links", written)

    def test_build_ops_report_handles_empty_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            missing_source_input_import = Path(temp_dir) / "missing_source_input_import.csv"
            missing_source_input_plan = Path(temp_dir) / "missing_source_input_plan.csv"
            init_db(db_path)
            with open_db(db_path) as conn:
                report = build_ops_report(
                    conn,
                    source_input_import_path=missing_source_input_import,
                    source_input_plan_path=missing_source_input_plan,
                )

        self.assertIn("No pipeline run has been recorded yet.", report)
        self.assertIn("No data source status has been recorded yet.", report)
        self.assertIn("No evidence_items rows have been recorded yet.", report)
        self.assertIn("No review queue rows are available", report)
        self.assertIn("No source input import dry-run report found", report)
        self.assertIn("No source input action plan found", report)


if __name__ == "__main__":
    unittest.main()
