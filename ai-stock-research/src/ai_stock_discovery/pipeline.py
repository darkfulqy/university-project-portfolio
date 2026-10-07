from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Callable

from ai_stock_discovery.analysis.catalysts import extract_catalysts_from_evidence, store_catalysts
from ai_stock_discovery.analysis.industry_tags import infer_ai_chain_tags
from ai_stock_discovery.analysis.keywords import (
    LOCAL_AI_CONTEXT_SOURCE_NAME,
    scan_local_ai_context,
    scan_text,
    store_keyword_signals,
)
from ai_stock_discovery.analysis.risk_extraction import (
    extract_risk_flags_from_evidence,
    scan_risk_text,
    store_extracted_risk_flags,
)
from ai_stock_discovery.analysis import universe_filters
from ai_stock_discovery.ai_chain_coverage import build_ai_chain_coverage, write_ai_chain_coverage_csv
from ai_stock_discovery.config import Settings
from ai_stock_discovery.data_mining_leads import build_data_mining_leads, write_data_mining_leads_csv
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit, write_evidence_audit_csv
from ai_stock_discovery.change_report import build_snapshot_change_report, write_snapshot_change_csv
from ai_stock_discovery.card_artifact_audit import build_card_artifact_audit, write_card_artifact_audit_csv
from ai_stock_discovery.card_batch import build_research_cards, write_card_index_csv
from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.mvp_readiness import build_mvp_readiness, write_mvp_readiness_csv
from ai_stock_discovery.ops_report import build_ops_report, write_ops_report
from ai_stock_discovery.review_queue import build_review_queue, write_review_queue_csv
from ai_stock_discovery.refresh_plan import build_refresh_plan, write_refresh_plan_csv
from ai_stock_discovery.research_pool import build_research_pool, write_research_pool_csv
from ai_stock_discovery.sec_companyfacts_backfill import (
    backfill_sec_companyfacts,
    record_sec_companyfacts_backfill_unavailable,
    select_sec_companyfacts_backfill_tickers,
)
from ai_stock_discovery.score_provenance import build_score_provenance, write_score_provenance_csv
from ai_stock_discovery.source_failures import build_source_failure_report, write_source_failure_csv
from ai_stock_discovery.source_input_import import (
    import_source_input_pack,
    write_source_input_import_csv,
)
from ai_stock_discovery.source_input_plan import (
    build_source_input_plan,
    write_source_input_plan_csv,
)
from ai_stock_discovery.source_input_templates import (
    build_source_input_audit,
    select_source_input_templates,
    write_source_input_audit_csv,
)
from ai_stock_discovery.sources import (
    expectations,
    finra,
    gdelt,
    insiders,
    ir,
    macro,
    market,
    nasdaq,
    news_rss,
    profile,
    sec,
    sec_rss,
)
from ai_stock_discovery.thesis_checklist import build_thesis_checklist, write_thesis_checklist_csv
from ai_stock_discovery.timeutils import utc_now_iso
from ai_stock_discovery.tracking import snapshot_scores
from ai_stock_discovery.watchlist import build_watchlist, write_watchlist_csv


DEFAULT_FORMS = "10-K,10-Q,8-K"
OWNERSHIP_FORMS = {"3", "3/A", "4", "4/A", "5", "5/A"}
SKIP_NETWORK_REASON = "Skipped because --skip-network was set."


@dataclass(frozen=True)
class PipelineOptions:
    skip_network: bool = False
    universe_limit: int | None = 500
    sec_rss_limit: int = 40
    sec_submissions_tickers: tuple[str, ...] = ()
    sec_submissions_limit: int = 10
    sec_companyfacts_failure_limit: int = 0
    process_filing_limit: int = 0
    ir_limit: int = 20
    news_limit: int = 25
    gdelt_queries: tuple[str, ...] = ()
    gdelt_limit: int = 25
    gdelt_timespan: str = "7d"
    macro_limit: int = 12
    finra_short_sale_tickers: tuple[str, ...] = ()
    finra_short_sale_lookback_days: int = 10
    finra_short_sale_file_code: str = finra.DEFAULT_DAILY_FILE_CODE
    market_source_dir: Path = Path(".")
    market_anomaly_limit: int = 200
    market_anomaly_lookback: int = 20
    catalyst_limit: int = 200
    risk_limit: int = 200
    fmp_profile_enrichment_limit: int = 0
    fmp_profile_enrichment_include_existing: bool = False
    profile_valuation_snapshot: bool = True
    ai_tag_limit: int = 500
    ai_context_limit: int = 500
    expectation_gap_limit: int = 500
    watchlist_limit: int = 200
    watchlist_min_score: float | None = None
    watchlist_output: Path = Path("reports/watchlist.csv")
    evidence_audit_output: Path = Path("reports/evidence_audit.csv")
    thesis_checklist_output: Path = Path("reports/thesis_checklist.csv")
    review_queue_output: Path = Path("reports/review_queue.csv")
    research_pool_output: Path = Path("reports/research_pool.csv")
    ai_chain_coverage_output: Path = Path("reports/ai_chain_coverage.csv")
    score_provenance_output: Path = Path("reports/score_provenance.csv")
    data_mining_leads_output: Path = Path("reports/data_mining_leads.csv")
    refresh_plan_output: Path = Path("reports/refresh_plan.csv")
    source_failure_output: Path = Path("reports/source_failures.csv")
    source_input_dir: Path = Path("data/input_templates")
    source_input_audit_output: Path = Path("reports/source_input_audit.csv")
    source_input_import_output: Path = Path("reports/source_input_import.csv")
    source_input_plan_output: Path = Path("reports/source_input_plan.csv")
    change_report_output: Path = Path("reports/snapshot_changes.csv")
    ops_report_output: Path = Path("reports/ops_report.md")
    mvp_readiness_output: Path = Path("reports/mvp_readiness.csv")
    mvp_readiness_design_doc: Path = Path("ai_potential_stock_discovery_system.md")
    cards_dir: Path = Path("reports/cards")
    card_index_output: Path = Path("reports/card_index.csv")
    card_artifact_audit_output: Path = Path("reports/card_artifact_audit.csv")
    forms: str = DEFAULT_FORMS
    min_market_cap: float | None = universe_filters.DEFAULT_MIN_MARKET_CAP
    continue_on_error: bool = True


@dataclass(frozen=True)
class PipelineStepResult:
    step_name: str
    status: str
    records_changed: int | None
    message: str
    error: str | None = None


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: int
    status: str
    steps: list[PipelineStepResult]


@dataclass(frozen=True)
class _Step:
    name: str
    action: Callable[[], tuple[int | None, str]]
    network: bool = False
    requires_sec_user_agent: bool = False
    source_name: str | None = None
    skip_when: Callable[[], str | None] | None = None


def run_pipeline(
    db_path: Path,
    settings: Settings,
    *,
    options: PipelineOptions | None = None,
) -> PipelineRunResult:
    options = options or PipelineOptions()
    init_db(db_path)
    run_id = _start_run(db_path, options)
    step_results: list[PipelineStepResult] = []

    for step in _build_steps(db_path, settings, options, run_id):
        skip_reason = _skip_reason(step, settings, options)
        if skip_reason:
            skipped_status = _skipped_source_status(db_path, step, skip_reason, options)
            if skipped_status:
                status, reason = skipped_status
                _mark_source_status(db_path, step.source_name, status, reason)
            result = _record_step(
                db_path,
                run_id=run_id,
                step_name=step.name,
                status="skipped",
                records_changed=None,
                message=skip_reason,
            )
            step_results.append(result)
            continue
        try:
            records_changed, message = step.action()
            result = _record_step(
                db_path,
                run_id=run_id,
                step_name=step.name,
                status="success",
                records_changed=records_changed,
                message=message,
            )
            step_results.append(result)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if step.source_name:
                _mark_source_status(db_path, step.source_name, "degraded", message)
            result = _record_step(
                db_path,
                run_id=run_id,
                step_name=step.name,
                status="error",
                records_changed=None,
                message="Step failed; see error.",
                error=message,
            )
            step_results.append(result)
            if not options.continue_on_error:
                _finish_run(db_path, run_id, "failed", step_results)
                raise

    for post_step_name, post_action in (
        ("build_ops_report", lambda: _build_ops_report(db_path, options)),
        ("build_mvp_readiness", lambda: _build_mvp_readiness(db_path, options)),
    ):
        status = "degraded" if any(step.status == "error" for step in step_results) else "success"
        _finish_run(db_path, run_id, status, step_results)
        try:
            records_changed, message = post_action()
            step_results.append(
                _record_step(
                    db_path,
                    run_id=run_id,
                    step_name=post_step_name,
                    status="success",
                    records_changed=records_changed,
                    message=message,
                )
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            step_results.append(
                _record_step(
                    db_path,
                    run_id=run_id,
                    step_name=post_step_name,
                    status="error",
                    records_changed=None,
                    message="Step failed; see error.",
                    error=message,
                )
            )
            if not options.continue_on_error:
                _finish_run(db_path, run_id, "failed", step_results)
                raise

    status = "degraded" if any(step.status == "error" for step in step_results) else "success"
    _finish_run(db_path, run_id, status, step_results)
    return PipelineRunResult(run_id=run_id, status=status, steps=step_results)


def list_pipeline_runs(conn: sqlite3.Connection, *, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, run_type, status, started_at, completed_at, summary
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def list_pipeline_steps(conn: sqlite3.Connection, *, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT step_name, status, records_changed, message, error, started_at, completed_at
        FROM pipeline_steps
        WHERE run_id = ?
        ORDER BY id
        """,
        (run_id,),
    ).fetchall()


def _build_steps(db_path: Path, settings: Settings, options: PipelineOptions, run_id: int) -> list[_Step]:
    return [
        _Step(
            name="fetch_universe",
            network=True,
            source_name="Nasdaq Trader Symbol Directory",
            action=lambda: _fetch_universe(db_path, settings, options),
        ),
        _Step(
            name="sync_sec_tickers",
            network=True,
            requires_sec_user_agent=True,
            source_name="SEC company_tickers",
            action=lambda: _sync_sec_tickers(db_path, settings),
        ),
        _Step(
            name="fetch_sec_submissions",
            network=True,
            requires_sec_user_agent=True,
            source_name="SEC submissions",
            skip_when=lambda: (
                "No SEC submissions backfill tickers configured; submissions step skipped."
                if not options.sec_submissions_tickers
                else None
            ),
            action=lambda: _fetch_sec_submissions(db_path, settings, options),
        ),
        _Step(
            name="backfill_sec_companyfacts_failures",
            network=True,
            source_name="SEC companyfacts",
            skip_when=lambda: (
                "sec_companyfacts_failure_limit is 0; source-failure backfill step skipped."
                if options.sec_companyfacts_failure_limit <= 0
                else None
            ),
            action=lambda: _backfill_sec_companyfacts_failures(db_path, settings, options),
        ),
        _Step(
            name="fetch_sec_rss",
            network=True,
            requires_sec_user_agent=True,
            source_name="SEC latest filings Atom feed",
            action=lambda: _fetch_sec_rss(db_path, settings, options),
        ),
        _Step(
            name="process_filing_queue",
            network=True,
            requires_sec_user_agent=True,
            source_name="SEC filing documents",
            skip_when=lambda: (
                "process_filing_limit is 0; queued filing documents were not fetched."
                if options.process_filing_limit <= 0
                else None
            ),
            action=lambda: _process_filing_queue(db_path, settings, options),
        ),
        _Step(
            name="check_ir_pages",
            network=True,
            source_name="Company IR page",
            action=lambda: _check_ir_pages(db_path, settings, options),
        ),
        _Step(
            name="fetch_news_rss",
            network=True,
            source_name="GlobeNewswire Press Releases",
            action=lambda: _fetch_news_rss(db_path, settings, options),
        ),
        _Step(
            name="fetch_gdelt_doc_news",
            network=True,
            source_name=gdelt.GDELT_DOC_API_SOURCE_NAME,
            skip_when=lambda: (
                "No GDELT DOC API queries configured; news search step skipped."
                if not options.gdelt_queries
                else None
            ),
            action=lambda: _fetch_gdelt_doc_news(db_path, settings, options),
        ),
        _Step(
            name="detect_market_sources",
            action=lambda: _detect_market_sources(db_path, options),
        ),
        _Step(
            name="detect_market_anomalies",
            action=lambda: _detect_market_anomalies(db_path, options),
        ),
        _Step(
            name="fetch_finra_short_sale_volume",
            network=True,
            source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
            skip_when=lambda: (
                "No FINRA short-sale ticker filters configured; step skipped to avoid full-market import."
                if not options.finra_short_sale_tickers
                else None
            ),
            action=lambda: _fetch_finra_short_sale_volume(db_path, settings, options),
        ),
        _Step(
            name="fetch_fred_macro",
            network=True,
            source_name="FRED",
            action=lambda: _fetch_fred_macro(db_path, settings, options),
        ),
        _Step(
            name="fetch_eia_electricity",
            network=True,
            source_name="EIA Open Data",
            skip_when=lambda: (
                "EIA_API_KEY is not configured; EIA electricity step skipped."
                if not settings.eia_api_key
                else None
            ),
            action=lambda: _fetch_eia_electricity(db_path, settings, options),
        ),
        _Step(
            name="enrich_fmp_profiles",
            network=True,
            source_name=profile.FMP_PROFILE_SOURCE_NAME,
            skip_when=lambda: _skip_fmp_profile_enrichment(settings, options),
            action=lambda: _enrich_fmp_profiles(db_path, settings, options),
        ),
        _Step(
            name="apply_universe_filters",
            action=lambda: _apply_universe_filters(db_path, options),
        ),
        _Step(
            name="extract_catalysts",
            action=lambda: _extract_catalysts(db_path, options),
        ),
        _Step(
            name="extract_risk_flags",
            action=lambda: _extract_risk_flags(db_path, options),
        ),
        _Step(
            name="infer_ai_chain_tags",
            action=lambda: _infer_ai_chain_tags(db_path, options),
        ),
        _Step(
            name="scan_local_ai_context",
            action=lambda: _scan_local_ai_context(db_path, options),
        ),
        _Step(
            name="infer_expectation_gaps",
            action=lambda: _infer_expectation_gaps(db_path, options),
        ),
        _Step(
            name="build_watchlist",
            action=lambda: _build_watchlist(db_path, options),
        ),
        _Step(
            name="build_evidence_audit",
            action=lambda: _build_evidence_audit(db_path, options),
        ),
        _Step(
            name="build_thesis_checklist",
            action=lambda: _build_thesis_checklist(db_path, options),
        ),
        _Step(
            name="build_review_queue",
            action=lambda: _build_review_queue(db_path, options),
        ),
        _Step(
            name="build_research_pool",
            action=lambda: _build_research_pool(db_path, options),
        ),
        _Step(
            name="build_ai_chain_coverage",
            action=lambda: _build_ai_chain_coverage(db_path, options),
        ),
        _Step(
            name="build_score_provenance",
            action=lambda: _build_score_provenance(db_path, options),
        ),
        _Step(
            name="build_research_cards",
            action=lambda: _build_research_cards(db_path, options),
        ),
        _Step(
            name="build_card_artifact_audit",
            action=lambda: _build_card_artifact_audit(db_path, options),
        ),
        _Step(
            name="build_refresh_plan",
            action=lambda: _build_refresh_plan(db_path, options),
        ),
        _Step(
            name="build_source_failure_report",
            action=lambda: _build_source_failure_report(db_path, options),
        ),
        _Step(
            name="build_data_mining_leads",
            action=lambda: _build_data_mining_leads(db_path, options),
        ),
        _Step(
            name="build_source_input_audit",
            action=lambda: _build_source_input_audit(db_path, options),
        ),
        _Step(
            name="build_source_input_import_plan",
            action=lambda: _build_source_input_import_plan(db_path, options),
        ),
        _Step(
            name="build_source_input_action_plan",
            action=lambda: _build_source_input_action_plan(db_path, options),
        ),
        _Step(
            name="snapshot_scores",
            action=lambda: _snapshot_scores(db_path, options, run_id),
        ),
        _Step(
            name="build_snapshot_change_report",
            action=lambda: _build_snapshot_change_report(db_path, options),
        ),
    ]


def _skip_reason(step: _Step, settings: Settings, options: PipelineOptions) -> str | None:
    if step.network and options.skip_network:
        return SKIP_NETWORK_REASON
    if step.skip_when:
        reason = step.skip_when()
        if reason:
            return reason
    if step.requires_sec_user_agent and not settings.sec_user_agent:
        return "SEC_USER_AGENT is not configured; SEC network step skipped."
    return None


def _skipped_source_status(
    db_path: Path,
    step: _Step,
    skip_reason: str,
    options: PipelineOptions,
) -> tuple[str, str] | None:
    if not step.source_name:
        return None
    if step.network and options.skip_network and skip_reason == SKIP_NETWORK_REASON:
        return None
    if "not configured" not in skip_reason.lower():
        return None
    local_rows = _local_source_backed_row_count(db_path, step.source_name)
    if local_rows:
        return (
            "ok",
            (
                f"Local source-backed rows remain available for this source "
                f"(rows={local_rows}); network refresh skipped: {skip_reason}"
            ),
        )
    return "unavailable", skip_reason


def _local_source_backed_row_count(db_path: Path, source_name: str) -> int:
    queries = {
        "SEC company_tickers": """
            SELECT COUNT(*) AS count
            FROM company_profile
            WHERE source = 'SEC company_tickers'
              AND COALESCE(ticker, '') != ''
              AND COALESCE(cik, '') != ''
        """,
        "SEC latest filings Atom feed": """
            SELECT COUNT(*) AS count
            FROM rss_filing_events
            WHERE COALESCE(filing_url, '') != ''
        """,
        profile.FMP_PROFILE_SOURCE_NAME: """
            SELECT COUNT(*) AS count
            FROM company_profile
            WHERE source LIKE '%financialmodelingprep.com/stable/profile%'
              AND COALESCE(description, '') != ''
        """,
    }
    query = queries.get(source_name)
    if not query:
        return 0
    with open_db(db_path) as conn:
        row = conn.execute(query).fetchone()
    return int(row["count"] or 0)


def _fetch_universe(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings))
    with open_db(db_path) as conn:
        records = nasdaq.fetch_universe(client, limit=options.universe_limit)
        count = nasdaq.upsert_universe(conn, records)
        sec.mark_source_status(conn, source_name="Nasdaq Trader Symbol Directory", status="ok")
    return count, f"Upserted {count} Nasdaq Trader universe record(s)."


def _sync_sec_tickers(db_path: Path, settings: Settings) -> tuple[int, str]:
    client = HttpClient(user_agent=settings.sec_user_agent)
    with open_db(db_path) as conn:
        count = sec.sync_company_tickers(conn, client)
        sec.mark_source_status(conn, source_name="SEC company_tickers", status="ok")
    return count, f"Upserted {count} SEC ticker/CIK mapping record(s)."


def _fetch_sec_rss(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    forms = _parse_forms(options.forms)
    client = HttpClient(user_agent=settings.sec_user_agent)
    with open_db(db_path) as conn:
        events = sec_rss.fetch_current_filings(client, count=options.sec_rss_limit)
        result = sec_rss.store_rss_events(conn, events, forms=forms)
        sec.mark_source_status(conn, source_name="SEC latest filings Atom feed", status="ok")
    return result.events_seen, (
        f"Stored/updated {result.events_seen} SEC RSS filing event(s); "
        f"queued {result.queued} mapped filing(s)."
    )


def _fetch_sec_submissions(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    forms = _parse_forms(options.forms)
    client = HttpClient(user_agent=settings.sec_user_agent, min_interval_seconds=0.2)
    metadata_count = 0
    queue_count = 0
    skipped: list[str] = []
    with open_db(db_path) as conn:
        for ticker in options.sec_submissions_tickers:
            normalized = ticker.upper()
            cik = sec.get_cik(conn, normalized)
            if not cik:
                skipped.append(normalized)
                continue
            submissions = sec.fetch_submissions(client, cik)
            filings = sec.list_recent_filings(
                ticker=normalized,
                cik=cik,
                submissions=submissions,
                forms=forms,
            )
            if options.sec_submissions_limit:
                filings = filings[: options.sec_submissions_limit]
            metadata_count += sec.upsert_filings(conn, filings)
            queue_count += sec.enqueue_recent_filings(conn, filings, source="SEC submissions")
        reason = (
            f"Backfilled {metadata_count} SEC submissions filing metadata record(s) "
            f"and queued/refreshed {queue_count} filing(s)."
        )
        if skipped:
            reason += " Skipped ticker(s) without local CIK mapping: " + ", ".join(skipped) + "."
        sec.mark_source_status(conn, source_name="SEC submissions", status="ok", reason=reason)
    return queue_count, reason


def _backfill_sec_companyfacts_failures(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        tickers = select_sec_companyfacts_backfill_tickers(
            conn,
            from_source_failures=True,
            limit=options.sec_companyfacts_failure_limit,
        )
        if not tickers:
            return 0, "No open FMP financial statements failures selected for SEC companyfacts backfill."
        if not settings.sec_user_agent:
            recorded = record_sec_companyfacts_backfill_unavailable(conn, tickers=tickers)
            return 0, (
                "SEC_USER_AGENT is not configured; recorded "
                f"{recorded} ticker-level SEC companyfacts failure(s) without fetching."
            )
        client = HttpClient(user_agent=settings.sec_user_agent, min_interval_seconds=0.2)
        result = backfill_sec_companyfacts(conn, client, tickers=tickers)
    return result.financial_fact_periods, (
        f"Backfilled SEC companyfacts for {result.tickers_considered} ticker(s); "
        f"financial_fact_periods={result.financial_fact_periods}; "
        f"successes={len(result.successes)}; failures={len(result.failures)}."
    )


def _process_filing_queue(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    forms = _parse_forms(options.forms)
    client = HttpClient(user_agent=settings.sec_user_agent, min_interval_seconds=0.2)
    processed = 0
    errors = 0
    signals_stored = 0
    risk_flags_stored = 0
    insider_transactions_stored = 0
    with open_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, ticker, cik, form, accession_number, filing_url, document_url
            FROM filing_queue
            WHERE status = 'queued'
              AND (? = 1 OR form IN (%s))
            ORDER BY id
            LIMIT ?
            """
            % _sql_placeholders(forms),
            (1 if not forms else 0, *sorted(forms), options.process_filing_limit),
        ).fetchall()
        for row in rows:
            try:
                document_url = row["document_url"] or sec.resolve_primary_document_url(
                    client,
                    filing_url=row["filing_url"],
                    accession_number=row["accession_number"],
                )
                if str(row["form"]).upper() in OWNERSHIP_FORMS:
                    raw_xml = client.get_text(document_url, accept="application/xml,text/xml,text/plain,*/*")
                    transactions = insiders.parse_ownership_xml(
                        ticker=row["ticker"],
                        xml_text=raw_xml,
                        source_url=document_url,
                        source_type=f"SEC {row['form']}",
                        source_name="SEC ownership filing",
                    )
                    insider_transactions_stored += insiders.upsert_insider_transactions(conn, transactions)
                else:
                    doc = sec.fetch_filing_document(
                        client,
                        ticker=row["ticker"],
                        cik=row["cik"],
                        form=row["form"],
                        filed_at="",
                        accession_number=row["accession_number"] or row["filing_url"],
                        document_url=document_url,
                    )
                    signals = scan_text(
                        ticker=row["ticker"],
                        text=doc.text,
                        source_type=f"SEC {doc.form}",
                        source_url=doc.document_url,
                        signal_date=doc.filed_at,
                    )
                    signals_stored += store_keyword_signals(conn, signals)
                    risk_flags = scan_risk_text(
                        ticker=row["ticker"],
                        text=doc.text,
                        source_type=f"SEC {doc.form}",
                        source_name="SEC filing documents",
                        source_url=doc.document_url,
                        risk_date=doc.filed_at,
                    )
                    risk_flags_stored += store_extracted_risk_flags(conn, risk_flags)
                _upsert_queue_filing(conn, row, document_url)
                conn.execute(
                    """
                    UPDATE filing_queue
                    SET status = 'processed', processed_at = datetime('now'), document_url = ?, last_error = NULL
                    WHERE id = ?
                    """,
                    (document_url, row["id"]),
                )
                processed += 1
            except Exception as exc:
                conn.execute(
                    """
                    UPDATE filing_queue
                    SET status = 'error', processed_at = datetime('now'), last_error = ?
                    WHERE id = ?
                    """,
                    (str(exc), row["id"]),
                )
                errors += 1
        sec.mark_source_status(
            conn,
            source_name="SEC filing documents",
            status="ok" if errors == 0 else "degraded",
            reason=None if errors == 0 else f"{errors} queued filing(s) failed during processing.",
        )
    return processed, (
        f"Processed {processed} queued filing(s), errors={errors}, "
        f"stored {signals_stored} AI signal candidate(s) and "
        f"{risk_flags_stored} risk flag candidate(s), plus "
        f"{insider_transactions_stored} insider transaction event(s)."
    )


def _fetch_news_rss(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    total_seen = 0
    total_mapped = 0
    failures = 0
    with open_db(db_path) as conn:
        for source in news_rss.DEFAULT_NEWS_RSS_SOURCES:
            try:
                events = news_rss.fetch_news_rss(
                    client,
                    url=source.url,
                    source_name=source.source_name,
                    limit=options.news_limit,
                )
                result = news_rss.store_news_events(conn, events)
            except Exception as exc:
                failures += 1
                sec.mark_source_status(
                    conn,
                    source_name=source.source_name,
                    status="degraded",
                    reason=_short_message(str(exc), 500),
                )
                continue
            total_seen += result.events_seen
            total_mapped += result.mapped
            sec.mark_source_status(
                conn,
                source_name=source.source_name,
                status="ok",
                reason="RSS announcements stored as title/summary/url evidence only.",
            )
    return total_seen, (
        f"Stored/updated {total_seen} news RSS event(s) from "
        f"{len(news_rss.DEFAULT_NEWS_RSS_SOURCES)} source(s); "
        f"mapped {total_mapped} to local tickers; failures={failures}."
    )


def _fetch_gdelt_doc_news(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=1.0)
    total_seen = 0
    total_mapped = 0
    failures = 0
    with open_db(db_path) as conn:
        for query in options.gdelt_queries:
            try:
                result = gdelt.fetch_gdelt_doc_news(
                    client,
                    query=query,
                    limit=options.gdelt_limit,
                    timespan=options.gdelt_timespan,
                )
                stored = news_rss.store_news_events(
                    conn,
                    result.events,
                    source_type="News Search",
                    content_hash_prefix="gdelt-doc-news",
                    mapped_confidence=0.6,
                    unmapped_confidence=0.35,
                )
            except Exception as exc:
                failures += 1
                sec.mark_source_status(
                    conn,
                    source_name=gdelt.GDELT_DOC_API_SOURCE_NAME,
                    status="degraded",
                    reason=_short_message(str(exc), 500),
                )
                continue
            total_seen += stored.events_seen
            total_mapped += stored.mapped
        status = "ok" if failures == 0 else "degraded"
        sec.mark_source_status(
            conn,
            source_name=gdelt.GDELT_DOC_API_SOURCE_NAME,
            status=status,
            reason=(
                "GDELT DOC API article metadata stored as title/url evidence only; "
                f"queries={len(options.gdelt_queries)}; failures={failures}."
            ),
        )
    return total_seen, (
        f"Stored/updated {total_seen} GDELT DOC article metadata event(s) from "
        f"{len(options.gdelt_queries)} query/queries; mapped {total_mapped} to local tickers; "
        f"failures={failures}."
    )


def _check_ir_pages(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.5)
    with open_db(db_path) as conn:
        result = ir.check_ir_pages(conn, client, limit=options.ir_limit)
        status = "degraded" if result.errors else "ok"
        sec.mark_source_status(
            conn,
            source_name="Company IR page",
            status=status,
            reason=(
                f"Checked {result.checked} active IR page(s); "
                f"changed={result.changed}; errors={len(result.errors)}."
            ),
        )
    return result.checked, (
        f"Checked {result.checked} active IR page(s); "
        f"changed={result.changed}; errors={len(result.errors)}."
    )


def _detect_market_sources(db_path: Path, options: PipelineOptions) -> tuple[int, str]:
    sources = market.detect_local_market_sources(options.market_source_dir)
    with open_db(db_path) as conn:
        for source in sources:
            status = "ok" if source.exists else "unavailable"
            reason = (
                f"Detected local source at {source.path}; no read/write performed."
                if source.exists
                else f"Local source not found at {source.path}."
            )
            sec.mark_source_status(conn, source_name=source.source_name, status=status, reason=reason)
    found = sum(1 for source in sources if source.exists)
    return found, (
        f"Detected {found}/{len(sources)} local market confirmation source(s); "
        f"market_source_dir={options.market_source_dir}."
    )


def _detect_market_anomalies(db_path: Path, options: PipelineOptions) -> tuple[int, str]:
    with open_db(db_path) as conn:
        result = market.detect_market_anomalies(
            conn,
            limit=options.market_anomaly_limit,
            lookback=options.market_anomaly_lookback,
        )
        status = "ok" if result.tickers_checked else "unavailable"
        reason = (
            f"Checked {result.tickers_checked} ticker(s); wrote {result.signals_written} signal(s)."
            if result.tickers_checked
            else "No local market price bars are available for anomaly detection."
        )
        sec.mark_source_status(conn, source_name="Local market anomaly detector", status=status, reason=reason)
    return result.signals_written, reason


def _fetch_finra_short_sale_volume(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        result = finra.fetch_latest_finra_daily_short_sale_volume(
            client,
            lookback_days=options.finra_short_sale_lookback_days,
            tickers=list(options.finra_short_sale_tickers),
            file_code=options.finra_short_sale_file_code,
        )
        count = finra.upsert_short_sale_volume(conn, result.records)
        status = "ok" if count else "degraded"
        sec.mark_source_status(
            conn,
            source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
            status=status,
            reason=(
                f"Fetched FINRA daily short sale volume for {result.trade_date}; "
                f"file_code={options.finra_short_sale_file_code.strip().upper()}; "
                f"ticker_filter={','.join(options.finra_short_sale_tickers)}; "
                f"imported {count} row(s). This is not short interest."
            ),
        )
    return count, (
        f"Fetched FINRA daily short sale volume for {result.trade_date}; "
        f"imported {count} row(s)."
    )


def _fetch_fred_macro(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings))
    default_series = [
        ("DGS10", "10-Year Treasury Constant Maturity Rate", "macro_rate"),
        ("FEDFUNDS", "Federal Funds Effective Rate", "macro_rate"),
    ]
    count = 0
    with open_db(db_path) as conn:
        for series_id, metric_name, category in default_series:
            if settings.fred_api_key:
                observations = macro.fetch_fred_observations(
                    client,
                    api_key=settings.fred_api_key,
                    series_id=series_id,
                    metric_name=metric_name,
                    category=category,
                    limit=options.macro_limit,
                )
            else:
                observations = macro.fetch_fred_public_observations(
                    client,
                    series_id=series_id,
                    metric_name=metric_name,
                    category=category,
                    limit=options.macro_limit,
                )
            count += macro.upsert_macro_indicators(conn, observations)
        sec.mark_source_status(
            conn,
            source_name="FRED",
            status="ok",
            reason=(
                "Macro rate observations stored as non-company background evidence."
                if settings.fred_api_key
                else "Public CSV macro observations stored as non-company background evidence."
            ),
        )
    return count, f"Stored {count} FRED macro observation(s)."


def _fetch_eia_electricity(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings))
    with open_db(db_path) as conn:
        observations = macro.fetch_eia_electricity_retail_sales(
            client,
            api_key=settings.eia_api_key or "",
            limit=options.macro_limit,
            frequency="monthly",
        )
        count = macro.upsert_macro_indicators(conn, observations)
        sec.mark_source_status(
            conn,
            source_name="EIA Open Data",
            status="ok",
            reason="Electricity observations stored as non-company background evidence.",
        )
    return count, f"Stored {count} EIA electricity observation(s)."


def _apply_universe_filters(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        result = universe_filters.apply_universe_filters(conn, min_market_cap=options.min_market_cap)
        sec.mark_source_status(
            conn,
            source_name="Universe exclusion rules",
            status="ok",
            reason=(
                f"Recorded {result.recorded} rule hit(s); "
                f"{result.active_rule_exclusions} active exclusion(s)."
            ),
        )
    return result.recorded, (
        f"Recorded {result.recorded} universe exclusion rule hit(s); "
        f"{result.active_rule_exclusions} active rule exclusion(s)."
    )


def _skip_fmp_profile_enrichment(settings: Settings, options: PipelineOptions) -> str | None:
    if options.fmp_profile_enrichment_limit <= 0:
        return "fmp_profile_enrichment_limit is 0; FMP profile enrichment step skipped."
    if not settings.fmp_api_key:
        return "FMP_API_KEY is not configured; FMP profile enrichment step skipped."
    return None


def _enrich_fmp_profiles(
    db_path: Path,
    settings: Settings,
    options: PipelineOptions,
) -> tuple[int, str]:
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        result = profile.enrich_fmp_profile_candidates(
            conn,
            client,
            api_key=settings.fmp_api_key or "",
            limit=options.fmp_profile_enrichment_limit,
            include_existing=options.fmp_profile_enrichment_include_existing,
            write_valuation_snapshot=options.profile_valuation_snapshot,
        )
        status = "ok" if result.profiles_written and not result.failures else (
            "degraded" if result.profiles_written or result.failures else "unavailable"
        )
        sec.mark_source_status(
            conn,
            source_name=profile.FMP_PROFILE_SOURCE_NAME,
            status=status,
            reason=(
                f"FMP profile enrichment selected {result.candidates_considered} candidate(s); "
                f"profiles={result.profiles_written}; "
                f"profile_market_cap_valuation_snapshots={result.valuation_snapshots_written}; "
                f"failures={len(result.failures)}."
            ),
        )
    return result.profiles_written + result.valuation_snapshots_written, (
        f"Selected {result.candidates_considered} FMP profile enrichment candidate(s); "
        f"profiles={result.profiles_written}; "
        f"profile_market_cap_valuation_snapshots={result.valuation_snapshots_written}; "
        f"failures={len(result.failures)}."
    )


def _extract_catalysts(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        candidates = extract_catalysts_from_evidence(conn, limit=options.catalyst_limit)
        inserted = store_catalysts(conn, candidates)
    return inserted, f"Extracted {len(candidates)} catalyst candidate(s); inserted {inserted}."


def _extract_risk_flags(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        flags = extract_risk_flags_from_evidence(conn, limit=options.risk_limit)
        inserted = store_extracted_risk_flags(conn, flags)
        sec.mark_source_status(
            conn,
            source_name="Automated risk text extraction",
            status="ok",
            reason=f"Extracted {inserted} watch-status risk flag candidate(s) from stored evidence.",
        )
    return inserted, f"Extracted {inserted} risk flag candidate(s) from stored evidence."


def _infer_ai_chain_tags(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        result = infer_ai_chain_tags(conn, limit=options.ai_tag_limit)
        sec.mark_source_status(
            conn,
            source_name="Automated AI industry tag inference",
            status="ok",
            reason=(
                f"Considered {result.tickers_considered} local candidate ticker(s); "
                f"tagged {result.tickers_tagged}; wrote {result.tags_written} inferred tag(s)."
            ),
        )
    return result.tags_written, (
        f"Considered {result.tickers_considered} local candidate ticker(s); "
        f"tagged {result.tickers_tagged}; wrote {result.tags_written} inferred AI industry-chain tag(s)."
    )


def _scan_local_ai_context(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        result = scan_local_ai_context(conn, limit=options.ai_context_limit)
        sec.mark_source_status(
            conn,
            source_name=LOCAL_AI_CONTEXT_SOURCE_NAME,
            status="ok" if result.signals_written else "degraded",
            reason=(
                f"Considered {result.tickers_considered} local ticker(s); "
                f"scanned {result.profiles_scanned} source-backed profile(s) and "
                f"{result.tag_rows_scanned} AI tag row(s); "
                f"wrote {result.signals_written} conservative AI context signal(s). "
                "Signals are research candidates only and do not prove revenue, orders, customers, or investment merit."
            ),
        )
    return result.signals_written, (
        f"Scanned {result.profiles_scanned} profile(s) and {result.tag_rows_scanned} AI tag row(s); "
        f"stored {result.signals_written} local AI context signal(s)."
    )


def _infer_expectation_gaps(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        signals = expectations.infer_profile_ai_tag_expectation_gaps(
            conn,
            limit=options.expectation_gap_limit,
        )
        count = expectations.upsert_expectation_gap_signals(conn, signals)
        sec.mark_source_status(
            conn,
            source_name=expectations.LOCAL_EXPECTATION_GAP_SOURCE_NAME,
            status="ok" if signals else "degraded",
            reason=(
                f"Inferred {len(signals)} review-only legacy-label expectation-gap candidate(s) "
                "from local source-backed company_profile and ai_industry_tags rows."
            )
            if signals
            else "No local profile/AI-tag combinations met the conservative expectation-gap inference rules.",
        )
    return count, f"Inferred {len(signals)} local expectation-gap candidate(s); stored {count}."


def _build_watchlist(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_watchlist(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_watchlist_csv(rows, options.watchlist_output)
    return len(rows), f"Wrote {len(rows)} watchlist row(s) to {options.watchlist_output}."


def _build_evidence_audit(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_evidence_audit(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_evidence_audit_csv(rows, options.evidence_audit_output)
    return len(rows), f"Wrote {len(rows)} evidence audit row(s) to {options.evidence_audit_output}."


def _build_thesis_checklist(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_thesis_checklist(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_thesis_checklist_csv(rows, options.thesis_checklist_output)
    return len(rows), f"Wrote {len(rows)} thesis checklist row(s) to {options.thesis_checklist_output}."


def _build_review_queue(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_review_queue(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_review_queue_csv(rows, options.review_queue_output)
    return len(rows), f"Wrote {len(rows)} review queue row(s) to {options.review_queue_output}."


def _build_research_pool(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_research_pool(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
            cards_dir=options.cards_dir,
        )
        write_research_pool_csv(rows, options.research_pool_output)
    return len(rows), f"Wrote {len(rows)} research pool row(s) to {options.research_pool_output}."


def _build_ai_chain_coverage(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_ai_chain_coverage(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_ai_chain_coverage_csv(rows, options.ai_chain_coverage_output)
    return len(rows), f"Wrote {len(rows)} AI chain coverage row(s) to {options.ai_chain_coverage_output}."


def _build_score_provenance(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_score_provenance(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_score_provenance_csv(rows, options.score_provenance_output)
    return len(rows), f"Wrote {len(rows)} score provenance row(s) to {options.score_provenance_output}."


def _build_research_cards(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_research_cards(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
            output_dir=options.cards_dir,
        )
        write_card_index_csv(rows, options.card_index_output)
    return len(rows), f"Generated {len(rows)} research card(s); index written to {options.card_index_output}."


def _build_card_artifact_audit(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_card_artifact_audit(
            conn,
            cards_dir=options.cards_dir,
            card_index_path=options.card_index_output,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_card_artifact_audit_csv(rows, options.card_artifact_audit_output)
    issue_count = sum(1 for row in rows if row.artifact_status != "current_indexed_card")
    return (
        len(rows),
        f"Wrote {len(rows)} card artifact audit row(s) to {options.card_artifact_audit_output}; "
        f"issues={issue_count}.",
    )


def _build_refresh_plan(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_refresh_plan(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_refresh_plan_csv(rows, options.refresh_plan_output)
    return len(rows), f"Wrote {len(rows)} refresh plan row(s) to {options.refresh_plan_output}."


def _build_source_failure_report(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_source_failure_report(conn, status="open", limit=options.watchlist_limit)
        write_source_failure_csv(rows, options.source_failure_output)
    return len(rows), f"Wrote {len(rows)} open source failure row(s) to {options.source_failure_output}."


def _build_data_mining_leads(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_data_mining_leads(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
        )
        write_data_mining_leads_csv(rows, options.data_mining_leads_output)
    return len(rows), f"Wrote {len(rows)} data mining lead row(s) to {options.data_mining_leads_output}."


def _build_source_input_audit(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        readiness_statuses = {
            row.area: row.status
            for row in build_mvp_readiness(
                conn,
                reports_dir=options.mvp_readiness_output.parent,
                design_doc=options.mvp_readiness_design_doc,
                report_paths=_mvp_report_paths(options),
                cards_dir=options.cards_dir,
            )
        }
        templates = select_source_input_templates(
            readiness_statuses=readiness_statuses,
            only_gaps=True,
        )
        rows = build_source_input_audit(templates, input_dir=options.source_input_dir)
        write_source_input_audit_csv(rows, options.source_input_audit_output)
    return len(rows), f"Wrote {len(rows)} source input audit row(s) to {options.source_input_audit_output}."


def _build_source_input_import_plan(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        readiness_statuses = {
            row.area: row.status
            for row in build_mvp_readiness(
                conn,
                reports_dir=options.mvp_readiness_output.parent,
                design_doc=options.mvp_readiness_design_doc,
                report_paths=_mvp_report_paths(options),
                cards_dir=options.cards_dir,
            )
        }
        templates = select_source_input_templates(
            readiness_statuses=readiness_statuses,
            only_gaps=True,
        )
        rows = import_source_input_pack(
            conn,
            templates,
            input_dir=options.source_input_dir,
            apply=False,
        )
        write_source_input_import_csv(rows, options.source_input_import_output)
    ready = sum(1 for row in rows if row.import_status == "dry_run_ready")
    return len(rows), (
        f"Wrote {len(rows)} source input import dry-run row(s) to "
        f"{options.source_input_import_output}; ready_to_apply={ready}."
    )


def _build_source_input_action_plan(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        readiness_statuses = {
            row.area: row.status
            for row in build_mvp_readiness(
                conn,
                reports_dir=options.mvp_readiness_output.parent,
                design_doc=options.mvp_readiness_design_doc,
                report_paths=_mvp_report_paths(options),
                cards_dir=options.cards_dir,
            )
        }
        templates = select_source_input_templates(
            readiness_statuses=readiness_statuses,
            only_gaps=True,
        )
        rows = build_source_input_plan(
            templates,
            input_dir=options.source_input_dir,
            readiness_statuses=readiness_statuses,
            audit_output=options.source_input_audit_output,
            import_output=options.source_input_import_output,
        )
        write_source_input_plan_csv(rows, options.source_input_plan_output)
    ready = sum(1 for row in rows if row.import_status == "dry_run_ready")
    return len(rows), (
        f"Wrote {len(rows)} source input action plan row(s) to "
        f"{options.source_input_plan_output}; ready_to_apply={ready}."
    )


def _snapshot_scores(
    db_path: Path,
    options: PipelineOptions,
    run_id: int,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        result = snapshot_scores(
            conn,
            limit=options.watchlist_limit,
            min_score=options.watchlist_min_score,
            run_id=run_id,
        )
    return (
        result.snapshots_written,
        f"Snapshotted {result.tickers_snapshotted} ticker score/review state row(s) at {result.snapshot_at}.",
    )


def _build_snapshot_change_report(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_snapshot_change_report(
            conn,
            limit=options.watchlist_limit,
        )
        write_snapshot_change_csv(rows, options.change_report_output)
    return len(rows), f"Wrote {len(rows)} snapshot change row(s) to {options.change_report_output}."


def _build_ops_report(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        content = build_ops_report(
            conn,
            source_input_import_path=options.source_input_import_output,
            source_input_plan_path=options.source_input_plan_output,
        )
        write_ops_report(content, options.ops_report_output)
    return 1, f"Wrote operations report to {options.ops_report_output}."


def _build_mvp_readiness(
    db_path: Path,
    options: PipelineOptions,
) -> tuple[int, str]:
    with open_db(db_path) as conn:
        rows = build_mvp_readiness(
            conn,
            reports_dir=options.mvp_readiness_output.parent,
            design_doc=options.mvp_readiness_design_doc,
            report_paths=_mvp_report_paths(options),
            cards_dir=options.cards_dir,
        )
        write_mvp_readiness_csv(rows, options.mvp_readiness_output)
    return len(rows), f"Wrote {len(rows)} MVP readiness row(s) to {options.mvp_readiness_output}."


def _mvp_report_paths(options: PipelineOptions) -> dict[str, Path]:
    return {
        "report_watchlist": options.watchlist_output,
        "report_evidence_audit": options.evidence_audit_output,
        "report_thesis_checklist": options.thesis_checklist_output,
        "report_review_queue": options.review_queue_output,
        "report_research_pool": options.research_pool_output,
        "report_ai_chain_coverage": options.ai_chain_coverage_output,
        "report_score_provenance": options.score_provenance_output,
        "report_data_mining_leads": options.data_mining_leads_output,
        "report_card_artifact_audit": options.card_artifact_audit_output,
        "report_card_index": options.card_index_output,
        "report_refresh_plan": options.refresh_plan_output,
        "report_source_failures": options.source_failure_output,
        "report_source_input_audit": options.source_input_audit_output,
        "report_source_input_import": options.source_input_import_output,
        "report_source_input_plan": options.source_input_plan_output,
        "report_snapshot_changes": options.change_report_output,
        "report_ops_report": options.ops_report_output,
    }


def _start_run(db_path: Path, options: PipelineOptions) -> int:
    started_at = utc_now_iso()
    options_payload = asdict(options)
    options_payload["watchlist_output"] = str(options.watchlist_output)
    options_payload["evidence_audit_output"] = str(options.evidence_audit_output)
    options_payload["thesis_checklist_output"] = str(options.thesis_checklist_output)
    options_payload["review_queue_output"] = str(options.review_queue_output)
    options_payload["research_pool_output"] = str(options.research_pool_output)
    options_payload["ai_chain_coverage_output"] = str(options.ai_chain_coverage_output)
    options_payload["score_provenance_output"] = str(options.score_provenance_output)
    options_payload["data_mining_leads_output"] = str(options.data_mining_leads_output)
    options_payload["refresh_plan_output"] = str(options.refresh_plan_output)
    options_payload["source_failure_output"] = str(options.source_failure_output)
    options_payload["market_source_dir"] = str(options.market_source_dir)
    options_payload["finra_short_sale_tickers"] = list(options.finra_short_sale_tickers)
    options_payload["gdelt_queries"] = list(options.gdelt_queries)
    options_payload["sec_companyfacts_failure_limit"] = options.sec_companyfacts_failure_limit
    options_payload["source_input_dir"] = str(options.source_input_dir)
    options_payload["source_input_audit_output"] = str(options.source_input_audit_output)
    options_payload["source_input_import_output"] = str(options.source_input_import_output)
    options_payload["source_input_plan_output"] = str(options.source_input_plan_output)
    options_payload["change_report_output"] = str(options.change_report_output)
    options_payload["ops_report_output"] = str(options.ops_report_output)
    options_payload["mvp_readiness_output"] = str(options.mvp_readiness_output)
    options_payload["mvp_readiness_design_doc"] = str(options.mvp_readiness_design_doc)
    options_payload["cards_dir"] = str(options.cards_dir)
    options_payload["card_index_output"] = str(options.card_index_output)
    options_payload["card_artifact_audit_output"] = str(options.card_artifact_audit_output)
    with open_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO pipeline_runs (run_type, status, started_at, options_json)
            VALUES (?, ?, ?, ?)
            """,
            ("mvp_pipeline", "running", started_at, json.dumps(options_payload, sort_keys=True)),
        )
        return int(cursor.lastrowid)


def _finish_run(
    db_path: Path,
    run_id: int,
    status: str,
    steps: list[PipelineStepResult],
) -> None:
    summary = "; ".join(f"{step.step_name}={step.status}" for step in steps)
    with open_db(db_path) as conn:
        conn.execute(
            """
            UPDATE pipeline_runs
            SET status = ?, completed_at = ?, summary = ?
            WHERE id = ?
            """,
            (status, utc_now_iso(), summary, run_id),
        )


def _record_step(
    db_path: Path,
    *,
    run_id: int,
    step_name: str,
    status: str,
    records_changed: int | None,
    message: str,
    error: str | None = None,
) -> PipelineStepResult:
    started_at = utc_now_iso()
    completed_at = utc_now_iso()
    with open_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_steps (
                run_id, step_name, status, started_at, completed_at,
                records_changed, message, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, step_name, status, started_at, completed_at, records_changed, message, error),
        )
    return PipelineStepResult(
        step_name=step_name,
        status=status,
        records_changed=records_changed,
        message=message,
        error=error,
    )


def _mark_source_status(db_path: Path, source_name: str, status: str, reason: str) -> None:
    with open_db(db_path) as conn:
        sec.mark_source_status(conn, source_name=source_name, status=status, reason=reason)


def _parse_forms(raw: str) -> set[str]:
    if raw.strip().lower() in {"", "all", "*"}:
        return set()
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def _sql_placeholders(values: set[str]) -> str:
    if not values:
        return "NULL"
    return ",".join("?" for _ in values)


def _short_message(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _upsert_queue_filing(conn: sqlite3.Connection, row: sqlite3.Row, document_url: str) -> None:
    conn.execute(
        """
        INSERT INTO filings (
            ticker, cik, form, filed_at, accession_number, filing_url,
            document_url, parsed_status, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(ticker, accession_number) DO UPDATE SET
            form=excluded.form,
            filing_url=excluded.filing_url,
            document_url=excluded.document_url,
            parsed_status=excluded.parsed_status,
            updated_at=excluded.updated_at
        """,
        (
            row["ticker"],
            row["cik"],
            row["form"],
            None,
            row["accession_number"] or row["filing_url"],
            row["filing_url"],
            document_url,
            "scanned",
        ),
    )


def _user_agent(settings: Settings) -> str:
    return settings.sec_user_agent or "ai-stock-discovery-mvp/0.1 local-research"
