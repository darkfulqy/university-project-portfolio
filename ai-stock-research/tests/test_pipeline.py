from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.config import Settings
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.pipeline import (
    PipelineOptions,
    _fetch_sec_submissions,
    _process_filing_queue,
    list_pipeline_runs,
    list_pipeline_steps,
    run_pipeline,
)
from ai_stock_discovery import pipeline
from ai_stock_discovery.sources import sec


class PipelineTests(unittest.TestCase):
    def test_skip_network_pipeline_records_steps_and_builds_watchlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            output = base / "watchlist.csv"
            audit_output = base / "evidence_audit.csv"
            thesis_output = base / "thesis_checklist.csv"
            review_output = base / "review_queue.csv"
            research_pool_output = base / "research_pool.csv"
            ai_chain_coverage_output = base / "ai_chain_coverage.csv"
            score_provenance_output = base / "score_provenance.csv"
            card_artifact_audit_output = base / "card_artifact_audit.csv"
            refresh_output = base / "refresh_plan.csv"
            source_failure_output = base / "source_failures.csv"
            data_mining_output = base / "data_mining_leads.csv"
            source_input_dir = base / "input_templates"
            source_input_audit_output = base / "source_input_audit.csv"
            source_input_import_output = base / "source_input_import.csv"
            source_input_plan_output = base / "source_input_plan.csv"
            change_output = base / "snapshot_changes.csv"
            ops_report_output = base / "ops_report.md"
            mvp_readiness_output = base / "mvp_readiness.csv"
            design_doc = base / "design.md"
            cards_dir = base / "cards"
            card_index_output = base / "card_index.csv"
            market_source_dir = base / "market_sources"
            design_doc.write_text("# AI \u6f5c\u529b\u80a1 MVP\n", encoding="utf-8")
            market_source_dir.mkdir()
            (market_source_dir / "options_flow.sqlite").write_text("placeholder", encoding="utf-8")
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.executemany(
                    """
                    INSERT INTO universe (
                        ticker, company_name, exchange, security_type,
                        is_etf, is_preferred, is_unit, is_active, last_checked_at
                    )
                    VALUES (?, ?, 'NASDAQ', 'common_stock_candidate', 0, 0, 0, 1, 'now')
                    """,
                    [
                        ("GOOD", "Good Compute Inc."),
                        ("SPAC", "Example Acquisition Corp Class A"),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO ai_relevance_signals (
                        ticker, signal_date, source_type, source_url, keyword,
                        context_snippet, ai_relevance_level, confidence, updated_at
                    )
                    VALUES (?, '2026-05-31', 'SEC 10-K', 'https://sec.example',
                            ?, ?, 2, 0.6, 'now')
                    """,
                    [
                        ("GOOD", "liquid cooling", "Liquid cooling for AI data center workloads."),
                        ("SPAC", "artificial intelligence", "AI evidence."),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, company_name, sector, industry, description, source, updated_at
                    )
                    VALUES (
                        'GOOD', 'Good Compute Inc.', 'Industrials', 'Thermal Management',
                        'Liquid cooling systems for AI data center workloads.',
                        'unit-profile', 'now'
                    )
                    """
                )
                for source_name in (
                    "SEC company_tickers",
                    "SEC latest filings Atom feed",
                    "SEC submissions",
                    "SEC filing documents",
                ):
                    sec.mark_source_status(
                        conn,
                        source_name=source_name,
                        status="ok",
                        reason=f"{source_name} was available before offline validation.",
                    )

            settings = Settings(db_path=db_path, sec_user_agent="")
            result = run_pipeline(
                db_path,
                settings,
                options=PipelineOptions(
                    skip_network=True,
                    watchlist_output=output,
                    evidence_audit_output=audit_output,
                    thesis_checklist_output=thesis_output,
                    review_queue_output=review_output,
                    research_pool_output=research_pool_output,
                    ai_chain_coverage_output=ai_chain_coverage_output,
                    score_provenance_output=score_provenance_output,
                    card_artifact_audit_output=card_artifact_audit_output,
                    refresh_plan_output=refresh_output,
                    source_failure_output=source_failure_output,
                    data_mining_leads_output=data_mining_output,
                    source_input_dir=source_input_dir,
                    source_input_audit_output=source_input_audit_output,
                    source_input_import_output=source_input_import_output,
                    source_input_plan_output=source_input_plan_output,
                    change_report_output=change_output,
                    ops_report_output=ops_report_output,
                    mvp_readiness_output=mvp_readiness_output,
                    mvp_readiness_design_doc=design_doc,
                    market_source_dir=market_source_dir,
                    cards_dir=cards_dir,
                    card_index_output=card_index_output,
                    watchlist_limit=10,
                    min_market_cap=None,
                ),
            )
            with open_db(db_path) as conn:
                runs = list_pipeline_runs(conn, limit=5)
                steps = list_pipeline_steps(conn, run_id=result.run_id)
                exclusions = conn.execute(
                    """
                    SELECT reason_type
                    FROM universe_exclusions
                    WHERE ticker = 'SPAC' AND is_active = 1
                    """
                ).fetchall()
                snapshot_rows = conn.execute(
                    "SELECT ticker, run_id FROM score_snapshots ORDER BY ticker"
                ).fetchall()
                tag_rows = conn.execute(
                    "SELECT ticker, tag FROM ai_industry_tags ORDER BY ticker, tag"
                ).fetchall()
                source_status_rows = conn.execute(
                    """
                    SELECT source_name, status
                    FROM data_source_status
                    WHERE source_name LIKE 'SEC%'
                    ORDER BY source_name
                    """
                ).fetchall()
                market_source_status = conn.execute(
                    """
                    SELECT status, reason
                    FROM data_source_status
                    WHERE source_name = 'Local options_flow.sqlite'
                    """
                ).fetchone()

            csv_text = output.read_text(encoding="utf-8")
            audit_text = audit_output.read_text(encoding="utf-8")
            thesis_text = thesis_output.read_text(encoding="utf-8")
            review_text = review_output.read_text(encoding="utf-8")
            research_pool_text = research_pool_output.read_text(encoding="utf-8")
            ai_chain_coverage_text = ai_chain_coverage_output.read_text(encoding="utf-8")
            score_provenance_text = score_provenance_output.read_text(encoding="utf-8")
            card_artifact_audit_text = card_artifact_audit_output.read_text(encoding="utf-8")
            refresh_text = refresh_output.read_text(encoding="utf-8")
            source_failure_text = source_failure_output.read_text(encoding="utf-8")
            data_mining_text = data_mining_output.read_text(encoding="utf-8")
            source_input_audit_text = source_input_audit_output.read_text(encoding="utf-8")
            source_input_import_text = source_input_import_output.read_text(encoding="utf-8")
            source_input_plan_text = source_input_plan_output.read_text(encoding="utf-8")
            change_text = change_output.read_text(encoding="utf-8")
            ops_report_text = ops_report_output.read_text(encoding="utf-8")
            mvp_readiness_text = mvp_readiness_output.read_text(encoding="utf-8")
            card_index_text = card_index_output.read_text(encoding="utf-8")
            card_text = (cards_dir / "GOOD.md").read_text(encoding="utf-8")

        self.assertEqual(result.status, "success")
        self.assertEqual(runs[0]["id"], result.run_id)
        self.assertGreaterEqual(len(steps), 10)
        self.assertEqual(steps[0]["step_name"], "fetch_universe")
        self.assertEqual(steps[0]["status"], "skipped")
        step_names = [step["step_name"] for step in steps]
        for expected_step in (
            "check_ir_pages",
            "fetch_sec_submissions",
            "backfill_sec_companyfacts_failures",
            "fetch_fred_macro",
            "fetch_eia_electricity",
            "detect_market_anomalies",
            "fetch_finra_short_sale_volume",
            "enrich_fmp_profiles",
            "extract_risk_flags",
            "infer_ai_chain_tags",
            "scan_local_ai_context",
            "infer_expectation_gaps",
            "build_watchlist",
            "build_evidence_audit",
            "build_thesis_checklist",
            "build_review_queue",
            "build_research_pool",
            "build_ai_chain_coverage",
            "build_score_provenance",
            "build_research_cards",
            "build_card_artifact_audit",
            "build_refresh_plan",
            "build_source_failure_report",
            "build_data_mining_leads",
            "build_source_input_audit",
            "build_source_input_import_plan",
            "build_source_input_action_plan",
            "snapshot_scores",
            "build_snapshot_change_report",
            "build_ops_report",
            "build_mvp_readiness",
        ):
            self.assertIn(expected_step, step_names)
        expected_local_tail = [
            "build_research_pool",
            "build_ai_chain_coverage",
            "build_score_provenance",
            "build_research_cards",
            "build_card_artifact_audit",
            "build_refresh_plan",
            "build_source_failure_report",
            "build_data_mining_leads",
            "build_source_input_audit",
            "build_source_input_import_plan",
            "build_source_input_action_plan",
            "snapshot_scores",
            "build_snapshot_change_report",
            "build_ops_report",
            "build_mvp_readiness",
        ]
        self.assertEqual(
            [name for name in step_names if name in expected_local_tail],
            expected_local_tail,
        )
        self.assertEqual(steps[-1]["status"], "success")
        self.assertEqual(exclusions[0]["reason_type"], "blank_check_or_spac")
        self.assertIn("GOOD", csv_text)
        self.assertNotIn("SPAC", csv_text)
        self.assertIn("GOOD", audit_text)
        self.assertNotIn("SPAC", audit_text)
        self.assertIn("GOOD", thesis_text)
        self.assertNotIn("SPAC", thesis_text)
        self.assertIn("GOOD", review_text)
        self.assertNotIn("SPAC", review_text)
        self.assertIn("GOOD", research_pool_text)
        self.assertNotIn("SPAC", research_pool_text)
        self.assertIn("cooling", ai_chain_coverage_text)
        self.assertIn("GOOD", ai_chain_coverage_text)
        self.assertNotIn("SPAC", ai_chain_coverage_text)
        self.assertIn("GOOD", score_provenance_text)
        self.assertNotIn("SPAC", score_provenance_text)
        self.assertIn("GOOD", card_artifact_audit_text)
        self.assertNotIn("SPAC", card_artifact_audit_text)
        self.assertIn("current_indexed_card", card_artifact_audit_text)
        self.assertIn("GOOD", refresh_text)
        self.assertNotIn("SPAC", refresh_text)
        self.assertIn("source_name,ticker,endpoint,failure_type", source_failure_text)
        self.assertIn("GOOD", data_mining_text)
        self.assertNotIn("SPAC", data_mining_text)
        self.assertIn("lead_type", data_mining_text)
        self.assertIn("template_name", source_input_audit_text)
        self.assertIn("missing_template", source_input_audit_text)
        self.assertIn("import_status", source_input_import_text)
        self.assertIn("not_ready", source_input_import_text)
        self.assertIn("recommended_next_action", source_input_plan_text)
        self.assertIn("missing_template", source_input_plan_text)
        self.assertIn("GOOD", change_text)
        self.assertNotIn("SPAC", change_text)
        self.assertIn("# AI Stock Discovery Operations Report", ops_report_text)
        self.assertIn("build_snapshot_change_report", ops_report_text)
        self.assertIn("Source Input Import Plan", ops_report_text)
        self.assertIn("Source Input Action Plan", ops_report_text)
        self.assertIn("Open Source Failures", ops_report_text)
        self.assertIn("not_ready", ops_report_text)
        self.assertIn("latest_pipeline_run", mvp_readiness_text)
        self.assertIn("source_ai_chain_tags", mvp_readiness_text)
        self.assertIn("report_ops_report", mvp_readiness_text)
        self.assertIn("path=" + str(ops_report_output).replace("\\", "/"), mvp_readiness_text)
        self.assertIn("report_watchlist", mvp_readiness_text)
        self.assertIn("report_research_cards", mvp_readiness_text)
        self.assertIn("report_card_artifact_audit", mvp_readiness_text)
        self.assertIn("report_source_input_audit", mvp_readiness_text)
        self.assertIn("report_source_input_import", mvp_readiness_text)
        self.assertIn("report_source_input_plan", mvp_readiness_text)
        self.assertIn("report_source_failures", mvp_readiness_text)
        self.assertIn("report_data_mining_leads", mvp_readiness_text)
        self.assertIn("GOOD", card_index_text)
        self.assertNotIn("SPAC", card_index_text)
        self.assertIn("GOOD 研究卡片", card_text)
        self.assertIn(("GOOD", "cooling"), [(row["ticker"], row["tag"]) for row in tag_rows])
        self.assertNotIn("SPAC", {row["ticker"] for row in tag_rows})
        self.assertEqual([(row["ticker"], row["run_id"]) for row in snapshot_rows], [("GOOD", result.run_id)])
        source_statuses = {row["source_name"]: row["status"] for row in source_status_rows}
        self.assertEqual(source_statuses["SEC company_tickers"], "ok")
        self.assertEqual(source_statuses["SEC latest filings Atom feed"], "ok")
        self.assertEqual(source_statuses["SEC submissions"], "ok")
        self.assertEqual(source_statuses["SEC filing documents"], "ok")
        self.assertEqual(market_source_status["status"], "ok")
        self.assertIn(str(market_source_dir), market_source_status["reason"])

    def test_missing_sec_user_agent_preserves_local_sec_mapping_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (
                        ticker, cik, company_name, source, updated_at
                    )
                    VALUES ('TEST', '0000000001', 'Test Corp', 'SEC company_tickers', 'now')
                    """
                )

            skipped_status = pipeline._skipped_source_status(
                db_path,
                pipeline._Step(
                    name="sync_sec_tickers",
                    network=True,
                    requires_sec_user_agent=True,
                    source_name="SEC company_tickers",
                    action=lambda: (0, "unused"),
                ),
                "SEC_USER_AGENT is not configured; SEC network step skipped.",
                PipelineOptions(),
            )

        self.assertIsNotNone(skipped_status)
        status, reason = skipped_status
        self.assertEqual(status, "ok")
        self.assertIn("Local source-backed rows remain available", reason)
        self.assertIn("rows=1", reason)

    def test_missing_sec_user_agent_preserves_local_sec_rss_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
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

            skipped_status = pipeline._skipped_source_status(
                db_path,
                pipeline._Step(
                    name="fetch_sec_rss",
                    network=True,
                    requires_sec_user_agent=True,
                    source_name="SEC latest filings Atom feed",
                    action=lambda: (0, "unused"),
                ),
                "SEC_USER_AGENT is not configured; SEC network step skipped.",
                PipelineOptions(),
            )

        self.assertIsNotNone(skipped_status)
        status, reason = skipped_status
        self.assertEqual(status, "ok")
        self.assertIn("Local source-backed rows remain available", reason)
        self.assertIn("rows=1", reason)

    def test_process_filing_queue_parses_sec_ownership_xml(self) -> None:
        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_text(self, url: str, *, accept: str = "*/*") -> str:
                return """
                <ownershipDocument>
                  <issuer><issuerTradingSymbol>TEST</issuerTradingSymbol></issuer>
                  <reportingOwner>
                    <reportingOwnerId><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId>
                    <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship>
                  </reportingOwner>
                  <nonDerivativeTable>
                    <nonDerivativeTransaction>
                      <transactionDate><value>2026-05-31</value></transactionDate>
                      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
                      <transactionAmounts>
                        <transactionShares><value>1000</value></transactionShares>
                        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
                      </transactionAmounts>
                    </nonDerivativeTransaction>
                  </nonDerivativeTable>
                </ownershipDocument>
                """

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO filing_queue (
                        ticker, cik, form, accession_number, filing_url, document_url,
                        source, status, queued_at
                    )
                    VALUES (
                        'TEST', '0000000001', '4', '0000000001-26-000001',
                        'https://www.sec.gov/Archives/test-index.html',
                        'https://www.sec.gov/Archives/test-form4.xml',
                        'unit', 'queued', 'now'
                    )
                    """
                )
            original_client = pipeline.HttpClient
            pipeline.HttpClient = FakeClient  # type: ignore[assignment]
            try:
                count, message = _process_filing_queue(
                    db_path,
                    Settings(db_path=db_path, sec_user_agent="unit@example.com"),
                    PipelineOptions(process_filing_limit=1, forms="4"),
                )
            finally:
                pipeline.HttpClient = original_client
            with open_db(db_path) as conn:
                tx = conn.execute(
                    "SELECT owner_name, transaction_code FROM insider_transactions WHERE ticker = 'TEST'"
                ).fetchone()
                queue = conn.execute("SELECT status FROM filing_queue").fetchone()
                risk_count = conn.execute("SELECT COUNT(*) AS count FROM risk_flags").fetchone()["count"]

        self.assertEqual(count, 1)
        self.assertIn("1 insider transaction", message)
        self.assertEqual(tx["owner_name"], "Jane Doe")
        self.assertEqual(tx["transaction_code"], "S")
        self.assertEqual(queue["status"], "processed")
        self.assertEqual(risk_count, 0)

    def test_fetch_sec_submissions_enqueues_recent_filings(self) -> None:
        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_json(self, url: str) -> dict:
                return {
                    "filings": {
                        "recent": {
                            "form": ["4", "10-K"],
                            "accessionNumber": ["0000000001-26-000001", "0000000001-26-000002"],
                            "filingDate": ["2026-05-31", "2026-05-30"],
                            "reportDate": ["", "2026-03-31"],
                            "primaryDocument": ["ownership.xml", "test-10k.htm"],
                        }
                    }
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('TEST', '0000000001', 'Test Corp', 'unit', 'now')
                    """
                )
            original_client = pipeline.HttpClient
            pipeline.HttpClient = FakeClient  # type: ignore[assignment]
            try:
                count, message = _fetch_sec_submissions(
                    db_path,
                    Settings(db_path=db_path, sec_user_agent="unit@example.com"),
                    PipelineOptions(
                        sec_submissions_tickers=("TEST",),
                        sec_submissions_limit=1,
                        forms="4",
                    ),
                )
            finally:
                pipeline.HttpClient = original_client
            with open_db(db_path) as conn:
                filing_count = conn.execute("SELECT COUNT(*) AS count FROM filings").fetchone()["count"]
                queue_row = conn.execute(
                    """
                    SELECT ticker, form, source, status, document_url
                    FROM filing_queue
                    WHERE ticker = 'TEST'
                    """
                ).fetchone()

        self.assertEqual(count, 1)
        self.assertIn("queued/refreshed 1 filing", message)
        self.assertEqual(filing_count, 1)
        self.assertEqual(queue_row["form"], "4")
        self.assertEqual(queue_row["source"], "SEC submissions")
        self.assertEqual(queue_row["status"], "queued")
        self.assertTrue(queue_row["document_url"].endswith("/ownership.xml"))


if __name__ == "__main__":
    unittest.main()
