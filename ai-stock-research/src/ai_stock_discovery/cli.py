from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

from ai_stock_discovery.analysis.catalysts import (
    extract_catalysts_from_evidence,
    store_catalysts,
)
from ai_stock_discovery.analysis.industry_tags import (
    infer_ai_chain_tags,
    import_ai_tags_csv,
    infer_tags_for_ticker,
    replace_inferred_tags_for_ticker,
)
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
from ai_stock_discovery.analysis.scoring import score_ticker, upsert_research_card
from ai_stock_discovery.analysis import universe_filters
from ai_stock_discovery.ai_chain_coverage import build_ai_chain_coverage, write_ai_chain_coverage_csv
from ai_stock_discovery.backtest import build_snapshot_backtest, write_backtest_csv
from ai_stock_discovery.card_artifact_audit import (
    archive_card_artifacts,
    build_card_artifact_audit,
    write_card_artifact_archive_csv,
    write_card_artifact_audit_csv,
)
from ai_stock_discovery.card_batch import build_research_cards, write_card_index_csv
from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.change_report import build_snapshot_change_report, write_snapshot_change_csv
from ai_stock_discovery.config import Settings, load_settings
from ai_stock_discovery.data_mining_leads import build_data_mining_leads, write_data_mining_leads_csv
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.evidence_audit import build_evidence_audit, write_evidence_audit_csv
from ai_stock_discovery.free_source_refresh import refresh_free_source_context, select_free_source_refresh_tickers
from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.market_lead_refresh import (
    refresh_yahoo_market_bars_from_leads,
    select_market_bar_fallback_tickers,
)
from ai_stock_discovery.mvp_readiness import build_mvp_readiness, write_mvp_readiness_csv
from ai_stock_discovery.ops_report import build_ops_report, write_ops_report
from ai_stock_discovery.pipeline import OWNERSHIP_FORMS, PipelineOptions, list_pipeline_runs, list_pipeline_steps, run_pipeline
from ai_stock_discovery.portfolio import build_portfolio_risk_report, import_portfolio_positions_csv, write_portfolio_risk_csv
from ai_stock_discovery.review_queue import build_review_queue, write_review_queue_csv
from ai_stock_discovery.refresh_plan import build_refresh_plan, write_refresh_plan_csv
from ai_stock_discovery.research_pool import build_research_pool, write_research_pool_csv
from ai_stock_discovery.sec_companyfacts_backfill import (
    backfill_sec_companyfacts,
    record_sec_companyfacts_backfill_unavailable,
    select_sec_companyfacts_backfill_tickers,
)
from ai_stock_discovery.score_provenance import build_score_provenance, write_score_provenance_csv
from ai_stock_discovery.seed_refresh import refresh_ai_seed_data, select_ai_seed_tickers
from ai_stock_discovery.source_input_templates import (
    build_source_input_audit,
    filter_source_input_templates,
    select_source_input_templates,
    write_source_input_audit_csv,
    write_source_input_template_pack,
)
from ai_stock_discovery.source_input_import import (
    import_source_input_pack,
    write_source_input_import_csv,
)
from ai_stock_discovery.source_input_plan import (
    build_source_input_plan,
    write_source_input_plan_csv,
)
from ai_stock_discovery.source_failures import (
    build_source_failure_report,
    mark_source_success,
    record_source_failure,
    write_source_failure_csv,
)
from ai_stock_discovery.sources import (
    analysts,
    catalyst_calendar,
    expectations,
    finra,
    gdelt,
    guidance,
    insiders,
    institutions,
    ir,
    local_research,
    macro,
    market,
    nasdaq,
    news_rss,
    peer_valuation,
    profile,
    risks,
    sec,
    sec_rss,
    transcripts,
    valuation,
)
from ai_stock_discovery.thesis_checklist import build_thesis_checklist, write_thesis_checklist_csv
from ai_stock_discovery.timeutils import utc_now_iso
from ai_stock_discovery.tracking import list_score_snapshots, snapshot_scores
from ai_stock_discovery.watchlist import build_watchlist, write_watchlist_csv


def main(argv: list[str] | None = None) -> None:
    settings = load_settings()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args, settings)
    except FetchError as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-stock-discovery",
        description="Traceable MVP for AI-related US equity discovery research.",
    )
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path.")
    subparsers = parser.add_subparsers(required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize SQLite database.")
    init_parser.set_defaults(func=cmd_init_db)

    universe_parser = subparsers.add_parser("fetch-universe", help="Fetch Nasdaq Trader universe.")
    universe_parser.add_argument("--limit", type=int, default=None)
    universe_parser.set_defaults(func=cmd_fetch_universe)

    universe_filters_parser = subparsers.add_parser(
        "apply-universe-filters",
        help="Apply source-backed universe exclusion rules.",
    )
    universe_filters_parser.add_argument(
        "--min-market-cap",
        type=float,
        default=universe_filters.DEFAULT_MIN_MARKET_CAP,
        help="Exclude latest valuation snapshots below this market cap when the market-cap filter is enabled.",
    )
    universe_filters_parser.add_argument(
        "--no-market-cap-filter",
        action="store_true",
        help="Skip microcap filtering when valuation snapshots are unavailable or not desired.",
    )
    universe_filters_parser.set_defaults(func=cmd_apply_universe_filters)

    universe_exclusions_parser = subparsers.add_parser(
        "list-universe-exclusions",
        help="List active universe exclusion reasons.",
    )
    universe_exclusions_parser.add_argument("--ticker", default=None)
    universe_exclusions_parser.add_argument("--limit", type=int, default=50)
    universe_exclusions_parser.add_argument("--all", action="store_true", help="Include inactive exclusions.")
    universe_exclusions_parser.set_defaults(func=cmd_list_universe_exclusions)

    sec_tickers_parser = subparsers.add_parser("sync-sec-tickers", help="Sync SEC ticker/CIK mapping.")
    sec_tickers_parser.set_defaults(func=cmd_sync_sec_tickers)

    filings_parser = subparsers.add_parser("sync-filings", help="Sync recent SEC filing metadata.")
    filings_parser.add_argument("--ticker", required=True)
    filings_parser.add_argument("--forms", default="10-K,10-Q,8-K")
    filings_parser.add_argument("--limit", type=int, default=20)
    filings_parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Also enqueue synced filings for process-filing-queue.",
    )
    filings_parser.set_defaults(func=cmd_sync_filings)

    facts_parser = subparsers.add_parser("fetch-companyfacts", help="Fetch and parse SEC companyfacts.")
    facts_parser.add_argument("--ticker", required=True)
    facts_parser.set_defaults(func=cmd_fetch_companyfacts)

    facts_backfill_parser = subparsers.add_parser(
        "backfill-sec-companyfacts",
        help="Backfill SEC companyfacts for explicit tickers or open FMP financial-statement failures.",
    )
    facts_backfill_parser.add_argument("--ticker", action="append", default=[])
    facts_backfill_parser.add_argument(
        "--from-source-failures",
        action="store_true",
        help="Select tickers from open FMP financial statements source failures. Used by default when --ticker is omitted.",
    )
    facts_backfill_parser.add_argument("--limit", type=int, default=25)
    facts_backfill_parser.set_defaults(func=cmd_backfill_sec_companyfacts)

    scan_parser = subparsers.add_parser("scan-sec-filings", help="Scan recent SEC filings for AI context.")
    scan_parser.add_argument("--ticker", required=True)
    scan_parser.add_argument("--forms", default="10-K,10-Q,8-K")
    scan_parser.add_argument("--limit", type=int, default=2)
    scan_parser.set_defaults(func=cmd_scan_sec_filings)

    local_ai_context_parser = subparsers.add_parser(
        "scan-local-ai-context",
        help="Scan local source-backed company profiles and AI tags into conservative AI context signals.",
    )
    local_ai_context_parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker to scan; repeat or comma-separate. Defaults to local AI-tag/profile candidates.",
    )
    local_ai_context_parser.add_argument("--limit", type=int, default=500)
    local_ai_context_parser.set_defaults(func=cmd_scan_local_ai_context)

    score_parser = subparsers.add_parser("score", help="Score a ticker from local evidence.")
    score_parser.add_argument("--ticker", required=True)
    score_parser.set_defaults(func=cmd_score)

    card_parser = subparsers.add_parser("card", help="Generate a research card from local evidence.")
    card_parser.add_argument("--ticker", required=True)
    card_parser.add_argument("--output-dir", type=Path, default=Path("reports/cards"))
    card_parser.add_argument("--stdout", action="store_true")
    card_parser.set_defaults(func=cmd_card)

    card_batch_parser = subparsers.add_parser(
        "build-research-cards",
        help="Generate local Markdown research cards and a card index from the review queue.",
    )
    card_batch_parser.add_argument("--ticker", action="append", default=[])
    card_batch_parser.add_argument("--limit", type=int, default=200)
    card_batch_parser.add_argument("--min-score", type=float, default=None)
    card_batch_parser.add_argument("--output-dir", type=Path, default=Path("reports/cards"))
    card_batch_parser.add_argument("--index-output", type=Path, default=Path("reports/card_index.csv"))
    card_batch_parser.set_defaults(func=cmd_build_research_cards)

    card_audit_parser = subparsers.add_parser(
        "build-card-artifact-audit",
        help="Build a CSV auditing generated research cards against the current card index and exclusions.",
    )
    card_audit_parser.add_argument("--cards-dir", type=Path, default=Path("reports/cards"))
    card_audit_parser.add_argument("--card-index", type=Path, default=Path("reports/card_index.csv"))
    card_audit_parser.add_argument("--limit", type=int, default=200)
    card_audit_parser.add_argument("--min-score", type=float, default=None)
    card_audit_parser.add_argument("--output", type=Path, default=Path("reports/card_artifact_audit.csv"))
    card_audit_parser.set_defaults(func=cmd_build_card_artifact_audit)

    card_archive_parser = subparsers.add_parser(
        "archive-card-artifacts",
        help="Archive stale generated research card files reported by the card artifact audit.",
    )
    card_archive_parser.add_argument("--cards-dir", type=Path, default=Path("reports/cards"))
    card_archive_parser.add_argument("--card-index", type=Path, default=Path("reports/card_index.csv"))
    card_archive_parser.add_argument("--archive-dir", type=Path, default=Path("reports/cards/_archive"))
    card_archive_parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Artifact status to archive; repeat or comma-separate. Defaults to stale excluded/unindexed cards.",
    )
    card_archive_parser.add_argument("--limit", type=int, default=200)
    card_archive_parser.add_argument("--min-score", type=float, default=None)
    card_archive_parser.add_argument("--output", type=Path, default=Path("reports/card_artifact_archive_plan.csv"))
    card_archive_parser.add_argument("--apply", action="store_true", help="Move files to the archive directory.")
    card_archive_parser.set_defaults(func=cmd_archive_card_artifacts)

    source_parser = subparsers.add_parser("source-status", help="Record data source status.")
    source_parser.add_argument("--source", required=True)
    source_parser.add_argument("--status", required=True, choices=["ok", "unavailable", "degraded"])
    source_parser.add_argument("--reason", default=None)
    source_parser.set_defaults(func=cmd_source_status)

    set_ir_parser = subparsers.add_parser("set-ir-url", help="Set or update a company IR URL.")
    set_ir_parser.add_argument("--ticker", required=True)
    set_ir_parser.add_argument("--url", required=True)
    set_ir_parser.add_argument("--page-type", default="ir_home")
    set_ir_parser.add_argument("--source", default="manual")
    set_ir_parser.add_argument("--notes", default=None)
    set_ir_parser.set_defaults(func=cmd_set_ir_url)

    import_ir_parser = subparsers.add_parser("import-ir-urls", help="Import IR URLs from CSV.")
    import_ir_parser.add_argument("--csv", required=True, type=Path)
    import_ir_parser.set_defaults(func=cmd_import_ir_urls)

    discover_ir_parser = subparsers.add_parser(
        "discover-ir-pages",
        help="Discover company IR pages from source-backed company_profile.website values.",
    )
    discover_ir_parser.add_argument("--ticker", action="append", default=[])
    discover_ir_parser.add_argument("--limit", type=int, default=25)
    discover_ir_parser.add_argument("--include-existing", action="store_true")
    discover_ir_parser.add_argument("--max-candidate-urls", type=int, default=8)
    discover_ir_parser.set_defaults(func=cmd_discover_ir_pages)

    check_ir_parser = subparsers.add_parser("check-ir-url", help="Fetch an IR page title and hash.")
    check_ir_parser.add_argument("--ticker", required=True)
    check_ir_parser.add_argument("--url", default=None)
    check_ir_parser.set_defaults(func=cmd_check_ir_url)

    check_ir_pages_parser = subparsers.add_parser(
        "check-ir-pages",
        help="Batch-check active IR pages by title and hash without storing full page text.",
    )
    check_ir_pages_parser.add_argument("--ticker", default=None)
    check_ir_pages_parser.add_argument("--limit", type=int, default=50)
    check_ir_pages_parser.set_defaults(func=cmd_check_ir_pages)

    rss_parser = subparsers.add_parser("fetch-sec-rss", help="Fetch SEC latest filings Atom feed.")
    rss_parser.add_argument("--limit", type=int, default=40)
    rss_parser.add_argument("--forms", default="10-K,10-Q,8-K")
    rss_parser.set_defaults(func=cmd_fetch_sec_rss)

    queue_parser = subparsers.add_parser("list-filing-queue", help="List queued filing events.")
    queue_parser.add_argument("--status", default="queued")
    queue_parser.add_argument("--limit", type=int, default=20)
    queue_parser.set_defaults(func=cmd_list_filing_queue)

    process_queue_parser = subparsers.add_parser(
        "process-filing-queue",
        help="Fetch queued SEC filing documents and scan them for evidence.",
    )
    process_queue_parser.add_argument("--limit", type=int, default=5)
    process_queue_parser.add_argument("--forms", default="10-K,10-Q,8-K")
    process_queue_parser.set_defaults(func=cmd_process_filing_queue)

    valuation_csv_parser = subparsers.add_parser(
        "import-valuation-csv",
        help="Import valuation snapshots from CSV.",
    )
    valuation_csv_parser.add_argument("--csv", required=True, type=Path)
    valuation_csv_parser.set_defaults(func=cmd_import_valuation_csv)

    peer_valuation_csv_parser = subparsers.add_parser(
        "import-peer-valuation-csv",
        help="Import source-backed peer valuation comparison records from CSV.",
    )
    peer_valuation_csv_parser.add_argument("--csv", required=True, type=Path)
    peer_valuation_csv_parser.set_defaults(func=cmd_import_peer_valuation_csv)

    guidance_csv_parser = subparsers.add_parser(
        "import-guidance-events-csv",
        help="Import source-backed management guidance events from CSV.",
    )
    guidance_csv_parser.add_argument("--csv", required=True, type=Path)
    guidance_csv_parser.set_defaults(func=cmd_import_guidance_events_csv)

    transcript_csv_parser = subparsers.add_parser(
        "import-transcript-snippets-csv",
        help="Import source-backed earnings call or presentation transcript snippets from CSV.",
    )
    transcript_csv_parser.add_argument("--csv", required=True, type=Path)
    transcript_csv_parser.add_argument(
        "--no-ai-scan",
        action="store_true",
        help="Store transcript snippets without scanning them for AI keyword candidates.",
    )
    transcript_csv_parser.add_argument(
        "--no-risk-scan",
        action="store_true",
        help="Store transcript snippets without scanning them for risk flag candidates.",
    )
    transcript_csv_parser.set_defaults(func=cmd_import_transcript_snippets_csv)

    short_sale_parser = subparsers.add_parser(
        "import-finra-short-sale-csv",
        help="Import FINRA daily short sale volume from CSV or pipe-delimited file.",
    )
    short_sale_parser.add_argument("--csv", required=True, type=Path)
    short_sale_parser.add_argument("--source-name", default=finra.DEFAULT_SHORT_SALE_SOURCE)
    short_sale_parser.add_argument("--source-url", default=None)
    short_sale_parser.set_defaults(func=cmd_import_finra_short_sale_csv)

    fetch_short_sale_parser = subparsers.add_parser(
        "fetch-finra-short-sale-volume",
        help="Fetch FINRA daily short sale volume from the public daily file.",
    )
    fetch_short_sale_parser.add_argument("--ticker", action="append", default=[])
    fetch_short_sale_parser.add_argument("--date", default=None, help="Trade date in YYYY-MM-DD format.")
    fetch_short_sale_parser.add_argument("--lookback-days", type=int, default=10)
    fetch_short_sale_parser.add_argument("--file-code", default=finra.DEFAULT_DAILY_FILE_CODE)
    fetch_short_sale_parser.set_defaults(func=cmd_fetch_finra_short_sale_volume)

    insider_csv_parser = subparsers.add_parser(
        "import-insider-transactions-csv",
        help="Import source-backed insider transaction/Form 4 events from CSV.",
    )
    insider_csv_parser.add_argument("--csv", required=True, type=Path)
    insider_csv_parser.set_defaults(func=cmd_import_insider_transactions_csv)

    institutional_csv_parser = subparsers.add_parser(
        "import-institutional-holdings-csv",
        help="Import source-backed institutional holding change events from CSV.",
    )
    institutional_csv_parser.add_argument("--csv", required=True, type=Path)
    institutional_csv_parser.set_defaults(func=cmd_import_institutional_holdings_csv)

    insider_list_parser = subparsers.add_parser(
        "list-insider-transactions",
        help="List source-backed insider transaction/Form 4 events.",
    )
    insider_list_parser.add_argument("--ticker", default=None)
    insider_list_parser.add_argument("--limit", type=int, default=20)
    insider_list_parser.set_defaults(func=cmd_list_insider_transactions)

    yahoo_parser = subparsers.add_parser(
        "fetch-yahoo-quote",
        help="Fetch a Yahoo Finance quote snapshot for prototype valuation data.",
    )
    yahoo_parser.add_argument("--ticker", required=True)
    yahoo_parser.set_defaults(func=cmd_fetch_yahoo_quote)

    eastmoney_parser = subparsers.add_parser(
        "fetch-eastmoney-quote",
        help="Fetch a public Eastmoney US quote snapshot as an FMP quote 402 fallback.",
    )
    eastmoney_parser.add_argument("--ticker", required=True)
    eastmoney_parser.set_defaults(func=cmd_fetch_eastmoney_quote)

    eastmoney_failures_parser = subparsers.add_parser(
        "refresh-eastmoney-quotes-from-failures",
        help="Backfill valuation price snapshots for open FMP quote failures with Eastmoney US quotes.",
    )
    eastmoney_failures_parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker to refresh; repeat or comma-separate. Defaults to open FMP quote failures.",
    )
    eastmoney_failures_parser.add_argument("--limit", type=int, default=25)
    eastmoney_failures_parser.set_defaults(func=cmd_refresh_eastmoney_quotes_from_failures)

    fmp_parser = subparsers.add_parser(
        "fetch-fmp-quote",
        help="Fetch an FMP quote snapshot when FMP_API_KEY is configured.",
    )
    fmp_parser.add_argument("--ticker", required=True)
    fmp_parser.set_defaults(func=cmd_fetch_fmp_quote)

    profile_csv_parser = subparsers.add_parser(
        "import-company-profiles",
        help="Import company profile fields from CSV.",
    )
    profile_csv_parser.add_argument("--csv", required=True, type=Path)
    profile_csv_parser.set_defaults(func=cmd_import_company_profiles)

    fmp_profile_parser = subparsers.add_parser(
        "fetch-fmp-profile",
        help="Fetch FMP company profile when FMP_API_KEY is configured.",
    )
    fmp_profile_parser.add_argument("--ticker", required=True)
    fmp_profile_parser.set_defaults(func=cmd_fetch_fmp_profile)

    enrich_profiles_parser = subparsers.add_parser(
        "enrich-fmp-profiles",
        help="Select local AI-chain name candidates and enrich their FMP company profiles.",
    )
    enrich_profiles_parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker to enrich; repeat or comma-separate. Defaults to name-scored local candidates.",
    )
    enrich_profiles_parser.add_argument("--limit", type=int, default=10)
    enrich_profiles_parser.add_argument("--include-existing", action="store_true")
    enrich_profiles_parser.add_argument("--dry-run", action="store_true")
    enrich_profiles_parser.add_argument(
        "--no-valuation-snapshot",
        action="store_true",
        help="Do not store FMP profile marketCap as a valuation snapshot fallback.",
    )
    enrich_profiles_parser.set_defaults(func=cmd_enrich_fmp_profiles)

    import_tags_parser = subparsers.add_parser(
        "import-ai-tags",
        help="Import AI industry-chain tags from CSV.",
    )
    import_tags_parser.add_argument("--csv", required=True, type=Path)
    import_tags_parser.set_defaults(func=cmd_import_ai_tags)

    tag_parser = subparsers.add_parser(
        "tag-ai-chain",
        help="Infer AI industry-chain tags from local profile and evidence.",
    )
    tag_parser.add_argument("--ticker", required=True)
    tag_parser.set_defaults(func=cmd_tag_ai_chain)

    infer_tags_parser = subparsers.add_parser(
        "infer-ai-chain-tags",
        help="Batch-infer AI industry-chain tags for local candidates.",
    )
    infer_tags_parser.add_argument("--ticker", action="append", default=[])
    infer_tags_parser.add_argument("--limit", type=int, default=500)
    infer_tags_parser.set_defaults(func=cmd_infer_ai_chain_tags)

    list_tags_parser = subparsers.add_parser("list-ai-tags", help="List AI industry-chain tags.")
    list_tags_parser.add_argument("--ticker", required=True)
    list_tags_parser.add_argument("--limit", type=int, default=20)
    list_tags_parser.set_defaults(func=cmd_list_ai_tags)

    catalyst_parser = subparsers.add_parser(
        "extract-catalysts",
        help="Extract candidate catalysts from stored evidence.",
    )
    catalyst_parser.add_argument("--ticker", default=None)
    catalyst_parser.add_argument("--limit", type=int, default=200)
    catalyst_parser.set_defaults(func=cmd_extract_catalysts)

    list_catalyst_parser = subparsers.add_parser("list-catalysts", help="List candidate catalysts.")
    list_catalyst_parser.add_argument("--ticker", required=True)
    list_catalyst_parser.add_argument("--limit", type=int, default=20)
    list_catalyst_parser.set_defaults(func=cmd_list_catalysts)

    catalyst_calendar_parser = subparsers.add_parser(
        "import-catalyst-calendar-csv",
        help="Import source-backed catalyst calendar events from CSV.",
    )
    catalyst_calendar_parser.add_argument("--csv", required=True, type=Path)
    catalyst_calendar_parser.set_defaults(func=cmd_import_catalyst_calendar_csv)

    fmp_earnings_calendar_parser = subparsers.add_parser(
        "fetch-fmp-earnings-calendar",
        help="Fetch FMP earnings calendar dates into catalyst records.",
    )
    fmp_earnings_calendar_parser.add_argument("--ticker", action="append", required=True)
    fmp_earnings_calendar_parser.add_argument("--from-date", default=None)
    fmp_earnings_calendar_parser.add_argument("--to-date", default=None)
    fmp_earnings_calendar_parser.add_argument("--limit-per-ticker", type=int, default=2)
    fmp_earnings_calendar_parser.set_defaults(func=cmd_fetch_fmp_earnings_calendar)

    upcoming_catalyst_parser = subparsers.add_parser(
        "list-upcoming-catalysts",
        help="List upcoming catalyst calendar events.",
    )
    upcoming_catalyst_parser.add_argument("--ticker", default=None)
    upcoming_catalyst_parser.add_argument("--from-date", default=None)
    upcoming_catalyst_parser.add_argument("--days", type=int, default=365)
    upcoming_catalyst_parser.add_argument("--limit", type=int, default=50)
    upcoming_catalyst_parser.set_defaults(func=cmd_list_upcoming_catalysts)

    watchlist_parser = subparsers.add_parser(
        "build-watchlist",
        help="Build a research watchlist CSV from local scores and evidence.",
    )
    watchlist_parser.add_argument("--limit", type=int, default=200)
    watchlist_parser.add_argument("--min-score", type=float, default=None)
    watchlist_parser.add_argument("--output", type=Path, default=Path("reports/watchlist.csv"))
    watchlist_parser.set_defaults(func=cmd_build_watchlist)

    audit_parser = subparsers.add_parser(
        "build-evidence-audit",
        help="Build a CSV showing source-backed evidence coverage and missing review inputs.",
    )
    audit_parser.add_argument("--ticker", action="append", default=[])
    audit_parser.add_argument("--limit", type=int, default=200)
    audit_parser.add_argument("--min-score", type=float, default=None)
    audit_parser.add_argument("--output", type=Path, default=Path("reports/evidence_audit.csv"))
    audit_parser.set_defaults(func=cmd_build_evidence_audit)

    thesis_checklist_parser = subparsers.add_parser(
        "build-thesis-checklist",
        help="Build a source-chain checklist for the five core thesis validation questions.",
    )
    thesis_checklist_parser.add_argument("--ticker", action="append", default=[])
    thesis_checklist_parser.add_argument("--limit", type=int, default=200)
    thesis_checklist_parser.add_argument("--min-score", type=float, default=None)
    thesis_checklist_parser.add_argument("--output", type=Path, default=Path("reports/thesis_checklist.csv"))
    thesis_checklist_parser.set_defaults(func=cmd_build_thesis_checklist)

    review_queue_parser = subparsers.add_parser(
        "build-review-queue",
        help="Build a human review queue with tracking cadence and required source checks.",
    )
    review_queue_parser.add_argument("--ticker", action="append", default=[])
    review_queue_parser.add_argument("--limit", type=int, default=200)
    review_queue_parser.add_argument("--min-score", type=float, default=None)
    review_queue_parser.add_argument("--output", type=Path, default=Path("reports/review_queue.csv"))
    review_queue_parser.add_argument("--cards-dir", type=Path, default=Path("reports/cards"))
    review_queue_parser.set_defaults(func=cmd_build_review_queue)

    research_pool_parser = subparsers.add_parser(
        "build-research-pool",
        help="Build a score-layered research pool from thesis gates and review queue evidence.",
    )
    research_pool_parser.add_argument("--ticker", action="append", default=[])
    research_pool_parser.add_argument("--limit", type=int, default=200)
    research_pool_parser.add_argument("--min-score", type=float, default=None)
    research_pool_parser.add_argument("--output", type=Path, default=Path("reports/research_pool.csv"))
    research_pool_parser.add_argument("--cards-dir", type=Path, default=Path("reports/cards"))
    research_pool_parser.set_defaults(func=cmd_build_research_pool)

    ai_chain_coverage_parser = subparsers.add_parser(
        "build-ai-chain-coverage",
        help="Build an AI industry-chain coverage report from local tags and research pool state.",
    )
    ai_chain_coverage_parser.add_argument("--ticker", action="append", default=[])
    ai_chain_coverage_parser.add_argument("--limit", type=int, default=200)
    ai_chain_coverage_parser.add_argument("--min-score", type=float, default=None)
    ai_chain_coverage_parser.add_argument("--top-limit", type=int, default=8)
    ai_chain_coverage_parser.add_argument("--output", type=Path, default=Path("reports/ai_chain_coverage.csv"))
    ai_chain_coverage_parser.set_defaults(func=cmd_build_ai_chain_coverage)

    score_provenance_parser = subparsers.add_parser(
        "build-score-provenance",
        help="Build a CSV tracing score components to local source links and thesis gates.",
    )
    score_provenance_parser.add_argument("--ticker", action="append", default=[])
    score_provenance_parser.add_argument("--limit", type=int, default=200)
    score_provenance_parser.add_argument("--min-score", type=float, default=None)
    score_provenance_parser.add_argument("--source-link-limit", type=int, default=12)
    score_provenance_parser.add_argument("--output", type=Path, default=Path("reports/score_provenance.csv"))
    score_provenance_parser.set_defaults(func=cmd_build_score_provenance)

    data_mining_parser = subparsers.add_parser(
        "build-data-mining-leads",
        help="Build an actionable data-mining queue from source-backed scores, failures, and refresh actions.",
    )
    data_mining_parser.add_argument("--ticker", action="append", default=[])
    data_mining_parser.add_argument("--limit", type=int, default=200)
    data_mining_parser.add_argument("--min-score", type=float, default=None)
    data_mining_parser.add_argument("--command-limit", type=int, default=5)
    data_mining_parser.add_argument("--output", type=Path, default=Path("reports/data_mining_leads.csv"))
    data_mining_parser.set_defaults(func=cmd_build_data_mining_leads)

    refresh_plan_parser = subparsers.add_parser(
        "build-refresh-plan",
        help="Build a local source refresh plan from review cadence and evidence gaps.",
    )
    refresh_plan_parser.add_argument("--ticker", action="append", default=[])
    refresh_plan_parser.add_argument("--limit", type=int, default=200)
    refresh_plan_parser.add_argument("--min-score", type=float, default=None)
    refresh_plan_parser.add_argument("--as-of", default=None)
    refresh_plan_parser.add_argument("--include-scheduled", action="store_true")
    refresh_plan_parser.add_argument("--output", type=Path, default=Path("reports/refresh_plan.csv"))
    refresh_plan_parser.set_defaults(func=cmd_build_refresh_plan)

    source_failure_parser = subparsers.add_parser(
        "build-source-failure-report",
        help="Build a CSV of ticker-level source failures such as HTTP 402 entitlement gaps.",
    )
    source_failure_parser.add_argument("--status", choices=["open", "resolved", "all"], default="open")
    source_failure_parser.add_argument("--limit", type=int, default=200)
    source_failure_parser.add_argument("--output", type=Path, default=Path("reports/source_failures.csv"))
    source_failure_parser.set_defaults(func=cmd_build_source_failure_report)

    ops_report_parser = subparsers.add_parser(
        "build-ops-report",
        help="Build a local operations report for pipeline, source status, evidence, and review queue.",
    )
    ops_report_parser.add_argument("--output", type=Path, default=Path("reports/ops_report.md"))
    ops_report_parser.add_argument("--review-limit", type=int, default=20)
    ops_report_parser.add_argument("--source-limit", type=int, default=80)
    ops_report_parser.add_argument("--evidence-limit", type=int, default=30)
    ops_report_parser.add_argument(
        "--source-input-import-report",
        type=Path,
        default=Path("reports/source_input_import.csv"),
    )
    ops_report_parser.add_argument(
        "--source-input-plan-report",
        type=Path,
        default=Path("reports/source_input_plan.csv"),
    )
    ops_report_parser.add_argument("--source-input-limit", type=int, default=20)
    ops_report_parser.add_argument("--source-input-plan-limit", type=int, default=20)
    ops_report_parser.add_argument("--stdout", action="store_true")
    ops_report_parser.set_defaults(func=cmd_build_ops_report)

    mvp_readiness_parser = subparsers.add_parser(
        "build-mvp-readiness",
        help="Build a local MVP readiness and gap checklist from source status, tables, and reports.",
    )
    mvp_readiness_parser.add_argument("--output", type=Path, default=Path("reports/mvp_readiness.csv"))
    mvp_readiness_parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    mvp_readiness_parser.add_argument("--design-doc", type=Path, default=Path("ai_potential_stock_discovery_system.md"))
    mvp_readiness_parser.set_defaults(func=cmd_build_mvp_readiness)

    source_templates_parser = subparsers.add_parser(
        "build-source-input-templates",
        help="Write blank local CSV templates and a manifest for source-backed data gap imports.",
    )
    source_templates_parser.add_argument("--output-dir", type=Path, default=Path("data/input_templates"))
    source_templates_parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("reports/source_input_templates.csv"),
    )
    source_templates_parser.add_argument(
        "--only-readiness-gaps",
        action="store_true",
        help="Only write templates whose mapped readiness area is not currently ready.",
    )
    source_templates_parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Limit to one source input template name or CSV filename. Repeat or comma-separate.",
    )
    source_templates_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing blank templates. Existing files are preserved by default.",
    )
    source_templates_parser.set_defaults(func=cmd_build_source_input_templates)

    source_input_audit_parser = subparsers.add_parser(
        "build-source-input-audit",
        help="Audit local source input templates before importing source-backed data.",
    )
    source_input_audit_parser.add_argument("--input-dir", type=Path, default=Path("data/input_templates"))
    source_input_audit_parser.add_argument("--output", type=Path, default=Path("reports/source_input_audit.csv"))
    source_input_audit_parser.add_argument(
        "--only-readiness-gaps",
        action="store_true",
        help="Only audit templates whose mapped readiness area is not currently ready.",
    )
    source_input_audit_parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Limit to one source input template name or CSV filename. Repeat or comma-separate.",
    )
    source_input_audit_parser.set_defaults(func=cmd_build_source_input_audit)

    source_input_import_parser = subparsers.add_parser(
        "import-source-input-pack",
        help="Dry-run or explicitly import source-backed rows from local source input templates.",
    )
    source_input_import_parser.add_argument("--input-dir", type=Path, default=Path("data/input_templates"))
    source_input_import_parser.add_argument("--output", type=Path, default=Path("reports/source_input_import.csv"))
    source_input_import_parser.add_argument(
        "--only-readiness-gaps",
        action="store_true",
        help="Only process templates whose mapped readiness area is not currently ready.",
    )
    source_input_import_parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Limit to one source input template name or CSV filename. Repeat or comma-separate.",
    )
    source_input_import_parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually import rows that pass audit. Without this flag the command is a dry run.",
    )
    source_input_import_parser.add_argument(
        "--finra-source-url",
        default=None,
        help="Optional original FINRA source URL/path for FINRA short-sale template imports.",
    )
    source_input_import_parser.add_argument(
        "--no-transcript-ai-scan",
        action="store_true",
        help="Import transcript snippets without scanning them for AI keyword candidates.",
    )
    source_input_import_parser.add_argument(
        "--no-transcript-risk-scan",
        action="store_true",
        help="Import transcript snippets without scanning them for risk flag candidates.",
    )
    source_input_import_parser.set_defaults(func=cmd_import_source_input_pack)

    source_input_plan_parser = subparsers.add_parser(
        "build-source-input-plan",
        help="Build a local action plan for filling, auditing, and importing source input templates.",
    )
    source_input_plan_parser.add_argument("--input-dir", type=Path, default=Path("data/input_templates"))
    source_input_plan_parser.add_argument("--output", type=Path, default=Path("reports/source_input_plan.csv"))
    source_input_plan_parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/source_input_audit.csv"),
    )
    source_input_plan_parser.add_argument(
        "--import-output",
        type=Path,
        default=Path("reports/source_input_import.csv"),
    )
    source_input_plan_parser.add_argument(
        "--only-readiness-gaps",
        action="store_true",
        help="Only plan templates whose mapped readiness area is not currently ready.",
    )
    source_input_plan_parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Limit to one source input template name or CSV filename. Repeat or comma-separate.",
    )
    source_input_plan_parser.set_defaults(func=cmd_build_source_input_plan)

    snapshot_parser = subparsers.add_parser(
        "snapshot-scores",
        help="Persist a point-in-time score, evidence coverage, and review-state snapshot.",
    )
    snapshot_parser.add_argument("--ticker", action="append", default=[])
    snapshot_parser.add_argument("--limit", type=int, default=200)
    snapshot_parser.add_argument("--min-score", type=float, default=None)
    snapshot_parser.set_defaults(func=cmd_snapshot_scores)

    list_snapshots_parser = subparsers.add_parser(
        "list-score-snapshots",
        help="List persisted score/review-state snapshots.",
    )
    list_snapshots_parser.add_argument("--ticker", default=None)
    list_snapshots_parser.add_argument("--limit", type=int, default=20)
    list_snapshots_parser.set_defaults(func=cmd_list_score_snapshots)

    change_report_parser = subparsers.add_parser(
        "build-snapshot-change-report",
        help="Build a local score/review-state change report from the latest two score snapshots.",
    )
    change_report_parser.add_argument("--ticker", action="append", default=[])
    change_report_parser.add_argument("--limit", type=int, default=200)
    change_report_parser.add_argument("--min-score-delta", type=float, default=0.0)
    change_report_parser.add_argument("--changed-only", action="store_true")
    change_report_parser.add_argument("--output", type=Path, default=Path("reports/snapshot_changes.csv"))
    change_report_parser.set_defaults(func=cmd_build_snapshot_change_report)

    backtest_parser = subparsers.add_parser(
        "build-backtest-report",
        help="Build a retrospective local price-bar outcome report from score snapshots.",
    )
    backtest_parser.add_argument("--ticker", action="append", default=[])
    backtest_parser.add_argument("--limit", type=int, default=500)
    backtest_parser.add_argument("--horizon-bars", type=int, default=5)
    backtest_parser.add_argument("--min-score", type=float, default=None)
    backtest_parser.add_argument("--bucket", action="append", default=[])
    backtest_parser.add_argument("--complete-only", action="store_true")
    backtest_parser.add_argument("--output", type=Path, default=Path("reports/backtest.csv"))
    backtest_parser.set_defaults(func=cmd_build_backtest_report)

    portfolio_import_parser = subparsers.add_parser(
        "import-portfolio-positions-csv",
        help="Import local portfolio/watchlist positions for risk review.",
    )
    portfolio_import_parser.add_argument("--csv", required=True, type=Path)
    portfolio_import_parser.add_argument("--portfolio", default="default")
    portfolio_import_parser.set_defaults(func=cmd_import_portfolio_positions_csv)

    portfolio_risk_parser = subparsers.add_parser(
        "build-portfolio-risk-report",
        help="Build a local portfolio exposure and evidence-risk review report.",
    )
    portfolio_risk_parser.add_argument("--portfolio", default=None)
    portfolio_risk_parser.add_argument("--limit", type=int, default=500)
    portfolio_risk_parser.add_argument("--output", type=Path, default=Path("reports/portfolio_risk.csv"))
    portfolio_risk_parser.set_defaults(func=cmd_build_portfolio_risk_report)

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run the repeatable MVP refresh pipeline and record step status.",
    )
    pipeline_parser.add_argument("--skip-network", action="store_true")
    pipeline_parser.add_argument("--universe-limit", type=int, default=500)
    pipeline_parser.add_argument("--sec-rss-limit", type=int, default=40)
    pipeline_parser.add_argument(
        "--sec-submissions-ticker",
        action="append",
        default=[],
        help="Ticker to backfill from SEC submissions into filing_queue; repeat or comma-separate.",
    )
    pipeline_parser.add_argument("--sec-submissions-limit", type=int, default=10)
    pipeline_parser.add_argument(
        "--sec-companyfacts-failure-limit",
        type=int,
        default=0,
        help="Backfill SEC companyfacts for this many open FMP financial-statements failures.",
    )
    pipeline_parser.add_argument("--process-filing-limit", type=int, default=0)
    pipeline_parser.add_argument("--ir-limit", type=int, default=20)
    pipeline_parser.add_argument("--news-limit", type=int, default=25)
    pipeline_parser.add_argument(
        "--gdelt-query",
        action="append",
        default=[],
        help="Optional GDELT DOC API query to run; repeat for multiple focused news searches.",
    )
    pipeline_parser.add_argument("--gdelt-limit", type=int, default=25)
    pipeline_parser.add_argument("--gdelt-timespan", default="7d")
    pipeline_parser.add_argument("--macro-limit", type=int, default=12)
    pipeline_parser.add_argument(
        "--finra-short-sale-ticker",
        action="append",
        default=[],
        help="Ticker to refresh from FINRA daily short sale volume; repeat or comma-separate.",
    )
    pipeline_parser.add_argument("--finra-short-sale-lookback-days", type=int, default=10)
    pipeline_parser.add_argument("--finra-short-sale-file-code", default=finra.DEFAULT_DAILY_FILE_CODE)
    pipeline_parser.add_argument("--market-source-dir", type=Path, default=None)
    pipeline_parser.add_argument("--market-anomaly-limit", type=int, default=200)
    pipeline_parser.add_argument("--market-anomaly-lookback", type=int, default=20)
    pipeline_parser.add_argument("--catalyst-limit", type=int, default=200)
    pipeline_parser.add_argument("--risk-limit", type=int, default=200)
    pipeline_parser.add_argument("--fmp-profile-enrichment-limit", type=int, default=0)
    pipeline_parser.add_argument("--fmp-profile-enrichment-include-existing", action="store_true")
    pipeline_parser.add_argument("--no-profile-valuation-snapshot", action="store_true")
    pipeline_parser.add_argument("--ai-tag-limit", type=int, default=500)
    pipeline_parser.add_argument("--ai-context-limit", type=int, default=500)
    pipeline_parser.add_argument("--expectation-gap-limit", type=int, default=500)
    pipeline_parser.add_argument("--watchlist-limit", type=int, default=200)
    pipeline_parser.add_argument("--min-score", type=float, default=None)
    pipeline_parser.add_argument("--output", type=Path, default=Path("reports/watchlist.csv"))
    pipeline_parser.add_argument("--evidence-audit-output", type=Path, default=Path("reports/evidence_audit.csv"))
    pipeline_parser.add_argument("--thesis-checklist-output", type=Path, default=Path("reports/thesis_checklist.csv"))
    pipeline_parser.add_argument("--review-queue-output", type=Path, default=Path("reports/review_queue.csv"))
    pipeline_parser.add_argument("--research-pool-output", type=Path, default=Path("reports/research_pool.csv"))
    pipeline_parser.add_argument("--ai-chain-coverage-output", type=Path, default=Path("reports/ai_chain_coverage.csv"))
    pipeline_parser.add_argument("--score-provenance-output", type=Path, default=Path("reports/score_provenance.csv"))
    pipeline_parser.add_argument("--data-mining-leads-output", type=Path, default=Path("reports/data_mining_leads.csv"))
    pipeline_parser.add_argument("--refresh-plan-output", type=Path, default=Path("reports/refresh_plan.csv"))
    pipeline_parser.add_argument("--source-failure-output", type=Path, default=Path("reports/source_failures.csv"))
    pipeline_parser.add_argument("--source-input-dir", type=Path, default=Path("data/input_templates"))
    pipeline_parser.add_argument("--source-input-audit-output", type=Path, default=Path("reports/source_input_audit.csv"))
    pipeline_parser.add_argument("--source-input-import-output", type=Path, default=Path("reports/source_input_import.csv"))
    pipeline_parser.add_argument("--source-input-plan-output", type=Path, default=Path("reports/source_input_plan.csv"))
    pipeline_parser.add_argument("--change-report-output", type=Path, default=Path("reports/snapshot_changes.csv"))
    pipeline_parser.add_argument("--ops-report-output", type=Path, default=Path("reports/ops_report.md"))
    pipeline_parser.add_argument("--mvp-readiness-output", type=Path, default=Path("reports/mvp_readiness.csv"))
    pipeline_parser.add_argument("--design-doc", type=Path, default=Path("ai_potential_stock_discovery_system.md"))
    pipeline_parser.add_argument("--cards-dir", type=Path, default=Path("reports/cards"))
    pipeline_parser.add_argument("--card-index-output", type=Path, default=Path("reports/card_index.csv"))
    pipeline_parser.add_argument("--card-artifact-audit-output", type=Path, default=Path("reports/card_artifact_audit.csv"))
    pipeline_parser.add_argument("--forms", default="10-K,10-Q,8-K")
    pipeline_parser.add_argument("--min-market-cap", type=float, default=universe_filters.DEFAULT_MIN_MARKET_CAP)
    pipeline_parser.add_argument("--no-market-cap-filter", action="store_true")
    pipeline_parser.add_argument("--stop-on-error", action="store_true")
    pipeline_parser.set_defaults(func=cmd_run_pipeline)

    seed_refresh_parser = subparsers.add_parser(
        "refresh-ai-seed-data",
        help="Refresh source-backed FMP/FINRA data for AI industry-chain seed tickers.",
    )
    seed_refresh_parser.add_argument("--ticker", action="append", default=[])
    seed_refresh_parser.add_argument("--limit", type=int, default=25)
    seed_refresh_parser.add_argument("--analyst-period", default="annual", choices=["annual", "quarter"])
    seed_refresh_parser.add_argument("--analyst-limit", type=int, default=3)
    seed_refresh_parser.add_argument("--financial-period", default="annual", choices=["annual", "quarter"])
    seed_refresh_parser.add_argument("--financial-limit", type=int, default=4)
    seed_refresh_parser.add_argument("--finra-lookback-days", type=int, default=10)
    seed_refresh_parser.add_argument("--finra-file-code", default=finra.DEFAULT_DAILY_FILE_CODE)
    seed_refresh_parser.add_argument("--skip-profile", action="store_true")
    seed_refresh_parser.add_argument("--skip-quote", action="store_true")
    seed_refresh_parser.add_argument("--skip-analyst", action="store_true")
    seed_refresh_parser.add_argument("--skip-finra", action="store_true")
    seed_refresh_parser.add_argument("--skip-sec-facts", action="store_true")
    seed_refresh_parser.add_argument("--skip-fmp-financials", action="store_true")
    seed_refresh_parser.set_defaults(func=cmd_refresh_ai_seed_data)

    free_refresh_parser = subparsers.add_parser(
        "refresh-free-source-context",
        help="Batch-refresh free/public source context selected from the refresh plan.",
    )
    free_refresh_parser.add_argument("--ticker", action="append", default=[])
    free_refresh_parser.add_argument("--limit", type=int, default=20)
    free_refresh_parser.add_argument("--plan-limit", type=int, default=300)
    free_refresh_parser.add_argument("--finra-lookback-days", type=int, default=10)
    free_refresh_parser.add_argument("--finra-file-code", default=finra.DEFAULT_DAILY_FILE_CODE)
    free_refresh_parser.add_argument("--news-limit", type=int, default=50)
    free_refresh_parser.add_argument("--include-gdelt", action="store_true")
    free_refresh_parser.add_argument("--gdelt-limit", type=int, default=10)
    free_refresh_parser.add_argument("--gdelt-timespan", default="30d")
    free_refresh_parser.add_argument("--extraction-limit", type=int, default=200)
    free_refresh_parser.add_argument("--skip-finra", action="store_true")
    free_refresh_parser.add_argument("--skip-news", action="store_true")
    free_refresh_parser.add_argument("--skip-catalysts", action="store_true")
    free_refresh_parser.add_argument("--skip-risks", action="store_true")
    free_refresh_parser.set_defaults(func=cmd_refresh_free_source_context)

    pipeline_runs_parser = subparsers.add_parser(
        "list-pipeline-runs",
        help="List recent pipeline runs and recorded step outcomes.",
    )
    pipeline_runs_parser.add_argument("--limit", type=int, default=10)
    pipeline_runs_parser.add_argument("--run-id", type=int, default=None)
    pipeline_runs_parser.set_defaults(func=cmd_list_pipeline_runs)

    detect_market_parser = subparsers.add_parser(
        "detect-market-sources",
        help="Detect local options-flow and premarket confirmation sources without modifying them.",
    )
    detect_market_parser.add_argument("--base-dir", type=Path, default=None)
    detect_market_parser.set_defaults(func=cmd_detect_market_sources)

    market_csv_parser = subparsers.add_parser(
        "import-market-confirmation-csv",
        help="Import source-backed market confirmation signals from CSV.",
    )
    market_csv_parser.add_argument("--csv", required=True, type=Path)
    market_csv_parser.set_defaults(func=cmd_import_market_confirmation_csv)

    market_bars_parser = subparsers.add_parser(
        "import-market-price-bars-csv",
        help="Import local source-backed price/volume bars for market anomaly detection.",
    )
    market_bars_parser.add_argument("--csv", required=True, type=Path)
    market_bars_parser.set_defaults(func=cmd_import_market_price_bars_csv)

    fmp_market_bars_parser = subparsers.add_parser(
        "fetch-fmp-market-bars",
        help="Fetch FMP historical EOD price/volume bars for market confirmation.",
    )
    fmp_market_bars_parser.add_argument("--ticker", action="append", required=True)
    fmp_market_bars_parser.add_argument("--benchmark", default="QQQ")
    fmp_market_bars_parser.add_argument("--lookback-days", type=int, default=90)
    fmp_market_bars_parser.set_defaults(func=cmd_fetch_fmp_market_bars)

    yahoo_market_bars_parser = subparsers.add_parser(
        "fetch-yahoo-market-bars",
        help="Fetch Yahoo chart EOD price/volume bars as a prototype market-data fallback.",
    )
    yahoo_market_bars_parser.add_argument("--ticker", action="append", required=True)
    yahoo_market_bars_parser.add_argument("--benchmark", default="QQQ")
    yahoo_market_bars_parser.add_argument("--lookback-days", type=int, default=90)
    yahoo_market_bars_parser.set_defaults(func=cmd_fetch_yahoo_market_bars)

    market_lead_refresh_parser = subparsers.add_parser(
        "refresh-market-bars-from-leads",
        help="Batch-refresh Yahoo prototype EOD bars for P0 market-data blockers from data-mining leads.",
    )
    market_lead_refresh_parser.add_argument("--ticker", action="append", default=[])
    market_lead_refresh_parser.add_argument("--limit", type=int, default=20)
    market_lead_refresh_parser.add_argument("--lead-limit", type=int, default=300)
    market_lead_refresh_parser.add_argument("--min-score", type=float, default=None)
    market_lead_refresh_parser.add_argument("--benchmark", default="QQQ")
    market_lead_refresh_parser.add_argument("--lookback-days", type=int, default=90)
    market_lead_refresh_parser.add_argument("--skip-anomalies", action="store_true")
    market_lead_refresh_parser.add_argument("--anomaly-lookback", type=int, default=20)
    market_lead_refresh_parser.set_defaults(func=cmd_refresh_market_bars_from_leads)

    market_anomaly_parser = subparsers.add_parser(
        "detect-market-anomalies",
        help="Detect volume, gap, relative-strength, and moving-average signals from local price bars.",
    )
    market_anomaly_parser.add_argument("--ticker", action="append", default=[])
    market_anomaly_parser.add_argument("--limit", type=int, default=200)
    market_anomaly_parser.add_argument("--lookback", type=int, default=20)
    market_anomaly_parser.add_argument("--min-volume-multiple", type=float, default=2.0)
    market_anomaly_parser.add_argument("--min-gap-pct", type=float, default=0.03)
    market_anomaly_parser.add_argument("--min-relative-strength-pct", type=float, default=0.03)
    market_anomaly_parser.set_defaults(func=cmd_detect_market_anomalies)

    expectation_csv_parser = subparsers.add_parser(
        "import-expectation-gap-csv",
        help="Import source-backed expectation-gap signals from CSV.",
    )
    expectation_csv_parser.add_argument("--csv", required=True, type=Path)
    expectation_csv_parser.set_defaults(func=cmd_import_expectation_gap_csv)

    infer_expectation_parser = subparsers.add_parser(
        "infer-expectation-gaps",
        help="Infer review-only expectation-gap candidates from local profiles and AI tags.",
    )
    infer_expectation_parser.add_argument("--ticker", action="append", default=[])
    infer_expectation_parser.add_argument("--limit", type=int, default=500)
    infer_expectation_parser.set_defaults(func=cmd_infer_expectation_gaps)

    analyst_estimate_csv_parser = subparsers.add_parser(
        "import-analyst-estimate-events-csv",
        help="Import source-backed analyst estimate or coverage events from CSV.",
    )
    analyst_estimate_csv_parser.add_argument("--csv", required=True, type=Path)
    analyst_estimate_csv_parser.set_defaults(func=cmd_import_analyst_estimate_events_csv)

    fmp_analyst_parser = subparsers.add_parser(
        "fetch-fmp-analyst-events",
        help="Fetch FMP analyst estimate snapshots into source-backed analyst events.",
    )
    fmp_analyst_parser.add_argument("--ticker", action="append", required=True)
    fmp_analyst_parser.add_argument("--period", default="annual", choices=["annual", "quarter"])
    fmp_analyst_parser.add_argument("--limit", type=int, default=5)
    fmp_analyst_parser.add_argument("--low-coverage-threshold", type=int, default=3)
    fmp_analyst_parser.set_defaults(func=cmd_fetch_fmp_analyst_events)

    risk_flags_parser = subparsers.add_parser(
        "import-risk-flags-csv",
        help="Import source-backed non-financial risk flags from CSV.",
    )
    risk_flags_parser.add_argument("--csv", required=True, type=Path)
    risk_flags_parser.set_defaults(func=cmd_import_risk_flags_csv)

    macro_csv_parser = subparsers.add_parser(
        "import-macro-csv",
        help="Import source-backed macro/electricity indicator observations from CSV.",
    )
    macro_csv_parser.add_argument("--csv", required=True, type=Path)
    macro_csv_parser.set_defaults(func=cmd_import_macro_csv)

    fred_parser = subparsers.add_parser(
        "fetch-fred-series",
        help="Fetch recent FRED series observations when FRED_API_KEY is configured.",
    )
    fred_parser.add_argument("--series-id", required=True)
    fred_parser.add_argument("--metric-name", default=None)
    fred_parser.add_argument("--category", default="macro")
    fred_parser.add_argument("--limit", type=int, default=12)
    fred_parser.add_argument("--observation-start", default=None)
    fred_parser.set_defaults(func=cmd_fetch_fred_series)

    fred_public_parser = subparsers.add_parser(
        "fetch-fred-public-series",
        help="Fetch recent FRED public CSV observations without an API key.",
    )
    fred_public_parser.add_argument("--series-id", required=True)
    fred_public_parser.add_argument("--metric-name", default=None)
    fred_public_parser.add_argument("--category", default="macro")
    fred_public_parser.add_argument("--limit", type=int, default=12)
    fred_public_parser.add_argument("--observation-start", default=None)
    fred_public_parser.set_defaults(func=cmd_fetch_fred_public_series)

    eia_parser = subparsers.add_parser(
        "fetch-eia-electricity-retail-sales",
        help="Fetch EIA electricity retail sales observations when EIA_API_KEY is configured.",
    )
    eia_parser.add_argument("--limit", type=int, default=12)
    eia_parser.add_argument("--frequency", default="monthly")
    eia_parser.add_argument("--stateid", default=None)
    eia_parser.add_argument("--sectorid", default=None)
    eia_parser.set_defaults(func=cmd_fetch_eia_electricity_retail_sales)

    macro_list_parser = subparsers.add_parser("list-macro-indicators", help="List recent macro/electricity observations.")
    macro_list_parser.add_argument("--limit", type=int, default=10)
    macro_list_parser.add_argument("--category", default=None)
    macro_list_parser.set_defaults(func=cmd_list_macro_indicators)

    extract_risks_parser = subparsers.add_parser(
        "extract-risk-flags",
        help="Extract conservative source-text risk flag candidates from stored evidence.",
    )
    extract_risks_parser.add_argument("--ticker", default=None)
    extract_risks_parser.add_argument("--limit", type=int, default=200)
    extract_risks_parser.set_defaults(func=cmd_extract_risk_flags)

    news_rss_parser = subparsers.add_parser(
        "fetch-news-rss",
        help="Fetch a news/announcement RSS feed and store mapped evidence.",
    )
    news_rss_parser.add_argument("--url", default=news_rss.GLOBENEWSWIRE_PRESS_RELEASES_RSS)
    news_rss_parser.add_argument("--source-name", default="GlobeNewswire Press Releases")
    news_rss_parser.add_argument("--limit", type=int, default=50)
    news_rss_parser.add_argument("--all-defaults", action="store_true")
    news_rss_parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker for ticker-scoped Yahoo/Nasdaq RSS defaults; repeat or comma-separate.",
    )
    news_rss_parser.add_argument(
        "--ticker-defaults",
        action="store_true",
        help="Fetch free ticker-scoped Yahoo Finance and Nasdaq RSS sources for --ticker values.",
    )
    news_rss_parser.set_defaults(func=cmd_fetch_news_rss)

    gdelt_parser = subparsers.add_parser(
        "fetch-gdelt-doc-news",
        help="Fetch GDELT DOC API article metadata and store mapped news evidence.",
    )
    gdelt_parser.add_argument("--query", required=True)
    gdelt_parser.add_argument("--limit", type=int, default=25)
    gdelt_parser.add_argument("--timespan", default="7d")
    gdelt_parser.set_defaults(func=cmd_fetch_gdelt_doc_news)

    local_research_parser = subparsers.add_parser(
        "import-local-research-md",
        help="Import source-linked local Markdown research leads as external verification cues.",
    )
    local_research_parser.add_argument("--path", required=True, type=Path)
    local_research_parser.add_argument("--source-name", default=local_research.LOCAL_RESEARCH_SOURCE_NAME)
    local_research_parser.add_argument("--limit", type=int, default=500)
    local_research_parser.set_defaults(func=cmd_import_local_research_md)
    return parser


def cmd_init_db(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    init_db(db_path)
    print(f"Initialized database: {db_path}")


def cmd_fetch_universe(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=_user_agent(settings))
    with open_db(db_path) as conn:
        records = nasdaq.fetch_universe(client, limit=args.limit)
        count = nasdaq.upsert_universe(conn, records)
        sec.mark_source_status(conn, source_name="Nasdaq Trader Symbol Directory", status="ok")
    print(f"Upserted {count} Nasdaq Trader universe record(s).")


def cmd_apply_universe_filters(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    min_market_cap = None if args.no_market_cap_filter else args.min_market_cap
    with open_db(db_path) as conn:
        result = universe_filters.apply_universe_filters(conn, min_market_cap=min_market_cap)
        sec.mark_source_status(
            conn,
            source_name="Universe exclusion rules",
            status="ok",
            reason=(
                f"Recorded {result.recorded} rule hit(s); "
                f"{result.active_rule_exclusions} active exclusion(s)."
            ),
        )
    print(
        f"Recorded {result.recorded} universe exclusion rule hit(s); "
        f"{result.active_rule_exclusions} active rule exclusion(s)."
    )


def cmd_list_universe_exclusions(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = universe_filters.list_universe_exclusions(
            conn,
            ticker=args.ticker,
            active_only=not args.all,
            limit=args.limit,
        )
    if not rows:
        scope = args.ticker.upper() if args.ticker else "the current database"
        print(f"No universe exclusions found for {scope}.")
        return
    for row in rows:
        state = "active" if int(row["is_active"] or 0) == 1 else "inactive"
        print(
            f"{row['ticker']} {state} {row['severity']} {row['reason_type']}: "
            f"{row['reason']} source={row['source_name']} url={row['source_url']}"
        )


def cmd_sync_sec_tickers(args: argparse.Namespace, settings: Settings) -> None:
    _require_sec_user_agent(settings)
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=settings.sec_user_agent)
    with open_db(db_path) as conn:
        count = sec.sync_company_tickers(conn, client)
        sec.mark_source_status(conn, source_name="SEC company_tickers", status="ok")
    print(f"Upserted {count} SEC ticker/CIK mapping record(s).")


def cmd_sync_filings(args: argparse.Namespace, settings: Settings) -> None:
    _require_sec_user_agent(settings)
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    forms = _parse_forms(args.forms)
    client = HttpClient(user_agent=settings.sec_user_agent)
    with open_db(db_path) as conn:
        cik = _ensure_cik(conn, client, ticker)
        submissions = sec.fetch_submissions(client, cik)
        filings = sec.list_recent_filings(
            ticker=ticker,
            cik=cik,
            submissions=submissions,
            forms=forms,
        )
        if args.limit:
            filings = filings[: args.limit]
        count = sec.upsert_filings(conn, filings)
        queued = sec.enqueue_recent_filings(conn, filings, source="SEC submissions") if args.enqueue else 0
        sec.mark_source_status(conn, source_name="SEC submissions", status="ok")
    suffix = f" Queued/refreshed {queued} filing(s)." if args.enqueue else ""
    print(f"Upserted {count} SEC filing metadata record(s) for {ticker}.{suffix}")


def cmd_fetch_companyfacts(args: argparse.Namespace, settings: Settings) -> None:
    _require_sec_user_agent(settings)
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    client = HttpClient(user_agent=settings.sec_user_agent)
    with open_db(db_path) as conn:
        cik = _ensure_cik(conn, client, ticker)
        payload = sec.fetch_companyfacts(client, cik)
        rows = sec.extract_financial_facts(ticker=ticker, cik=cik, companyfacts=payload)
        count = sec.upsert_financial_facts(conn, rows)
        sec.mark_source_status(conn, source_name="SEC companyfacts", status="ok")
    print(f"Upserted {count} SEC financial fact period(s) for {ticker}.")


def cmd_backfill_sec_companyfacts(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    db_path = _db_path(args, settings)
    explicit_tickers = _parse_ticker_args(args.ticker)
    from_source_failures = args.from_source_failures or not explicit_tickers
    with open_db(db_path) as conn:
        tickers = select_sec_companyfacts_backfill_tickers(
            conn,
            tickers=explicit_tickers,
            from_source_failures=from_source_failures,
            limit=args.limit,
        )
        if not tickers:
            print("No SEC companyfacts backfill tickers selected.")
            return
        if not settings.sec_user_agent:
            recorded = record_sec_companyfacts_backfill_unavailable(conn, tickers=tickers)
            print(
                "SEC_USER_AGENT is not configured; recorded "
                f"{recorded} ticker-level SEC companyfacts failure(s) without fetching."
            )
            return
        client = HttpClient(user_agent=settings.sec_user_agent, min_interval_seconds=0.2)
        result = backfill_sec_companyfacts(conn, client, tickers=tickers)
    print(
        f"Backfilled SEC companyfacts for {result.tickers_considered} ticker(s): "
        f"financial_fact_periods={result.financial_fact_periods}, "
        f"successes={len(result.successes)}, failures={len(result.failures)}."
    )
    for failure in result.failures[:5]:
        print(f"- {failure}")


def cmd_scan_sec_filings(args: argparse.Namespace, settings: Settings) -> None:
    _require_sec_user_agent(settings)
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    forms = _parse_forms(args.forms)
    client = HttpClient(user_agent=settings.sec_user_agent, min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        cik = _ensure_cik(conn, client, ticker)
        submissions = sec.fetch_submissions(client, cik)
        filing_rows = sec.list_recent_filings(ticker=ticker, cik=cik, submissions=submissions, forms=forms)
        signal_count = 0
        risk_flag_count = 0
        for filing in filing_rows[: args.limit]:
            doc = sec.fetch_filing_document(
                client,
                ticker=ticker,
                cik=cik,
                form=filing["form"],
                filed_at=filing["filed_at"],
                accession_number=filing["accession_number"],
                document_url=filing["document_url"],
            )
            signals = scan_text(
                ticker=ticker,
                text=doc.text,
                source_type=f"SEC {doc.form}",
                source_url=doc.document_url,
                signal_date=doc.filed_at,
            )
            signal_count += store_keyword_signals(conn, signals)
            risk_flags = scan_risk_text(
                ticker=ticker,
                text=doc.text,
                source_type=f"SEC {doc.form}",
                source_name="SEC filing documents",
                source_url=doc.document_url,
                risk_date=doc.filed_at,
            )
            risk_flag_count += store_extracted_risk_flags(conn, risk_flags)
        sec.upsert_recent_filings(
            conn,
            ticker=ticker,
            cik=cik,
            submissions=submissions,
            forms=forms,
            limit=args.limit,
        )
        sec.mark_source_status(conn, source_name="SEC filing documents", status="ok")
    print(
        f"Stored {signal_count} AI keyword signal candidate(s) and "
        f"{risk_flag_count} risk flag candidate(s) for {ticker}."
    )


def cmd_scan_local_ai_context(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative.")
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        result = scan_local_ai_context(conn, tickers=tickers or None, limit=args.limit)
        sec.mark_source_status(
            conn,
            source_name=LOCAL_AI_CONTEXT_SOURCE_NAME,
            status="ok" if result.signals_written else "degraded",
            reason=(
                f"Considered {result.tickers_considered} local ticker(s); "
                f"scanned {result.profiles_scanned} source-backed profile(s) and "
                f"{result.tag_rows_scanned} AI tag row(s); "
                f"wrote {result.signals_written} conservative AI context signal(s)."
            ),
        )
    print(
        f"Scanned {result.tickers_considered} ticker(s): "
        f"profiles={result.profiles_scanned}, profile_signals={result.profile_signals}, "
        f"tag_rows={result.tag_rows_scanned}, tag_signals={result.tag_signals}, "
        f"stored={result.signals_written}."
    )


def cmd_score(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        result = score_ticker(conn, args.ticker)
        upsert_research_card(conn, result)
    print(f"{result.ticker} score={result.score_total} status={result.status}")
    for note in result.notes:
        print(f"- {note}")


def cmd_card(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        output_dir = None if args.stdout else args.output_dir
        content = generate_card(conn, args.ticker, output_dir=output_dir)
    if args.stdout:
        print(content)
    else:
        print(f"Generated card: {args.output_dir / (args.ticker.upper() + '.md')}")


def cmd_build_research_cards(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_research_cards(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            output_dir=args.output_dir,
        )
        write_card_index_csv(rows, args.index_output)
    print(f"Generated {len(rows)} research card(s); index written to {args.index_output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} bucket={row.review_bucket} priority={row.review_priority} "
            f"card={row.card_path}"
        )


def cmd_build_card_artifact_audit(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = build_card_artifact_audit(
            conn,
            cards_dir=args.cards_dir,
            card_index_path=args.card_index,
            limit=args.limit,
            min_score=args.min_score,
        )
        write_card_artifact_audit_csv(rows, args.output)
    issue_count = sum(1 for row in rows if row.artifact_status != "current_indexed_card")
    print(f"Wrote {len(rows)} card artifact audit row(s) to {args.output}; issues={issue_count}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} status={row.artifact_status} "
            f"indexed={row.indexed} file_exists={row.file_exists}"
        )


def cmd_archive_card_artifacts(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    statuses = {status.lower() for status in _parse_csv_args(args.status)}
    with open_db(db_path) as conn:
        rows = archive_card_artifacts(
            conn,
            cards_dir=args.cards_dir,
            card_index_path=args.card_index,
            archive_dir=args.archive_dir,
            statuses=statuses or None,
            apply=args.apply,
            limit=args.limit,
            min_score=args.min_score,
        )
        write_card_artifact_archive_csv(rows, args.output)
    archived = sum(1 for row in rows if row.action_status == "archived")
    blocked = sum(1 for row in rows if row.action_status.startswith("blocked"))
    mode = "applied" if args.apply else "dry-run"
    print(
        f"Wrote {len(rows)} card artifact archive row(s) to {args.output}; "
        f"mode={mode}; archived={archived}; blocked={blocked}."
    )
    for row in rows[:5]:
        print(f"{row.ticker} {row.action_status} {row.source_path} -> {row.archive_path}")


def cmd_source_status(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        sec.mark_source_status(
            conn,
            source_name=args.source,
            status=args.status,
            reason=args.reason,
        )
    print(f"Recorded source status for {args.source}: {args.status}")


def cmd_set_ir_url(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        ir.upsert_ir_url(
            conn,
            ticker=args.ticker,
            url=args.url,
            page_type=args.page_type,
            source=args.source,
            notes=args.notes,
        )
    print(f"Recorded IR URL for {args.ticker.upper()}: {args.url}")


def cmd_import_ir_urls(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = ir.import_ir_urls(conn, args.csv)
    print(f"Imported {count} IR URL record(s).")


def cmd_discover_ir_pages(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.5)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        result = ir.discover_ir_pages_from_profiles(
            conn,
            client,
            tickers=tickers or None,
            limit=args.limit,
            include_existing=args.include_existing,
            max_candidate_urls=args.max_candidate_urls,
        )
        status = "ok" if result.discovered else "degraded"
        reason = (
            f"Inspected {result.candidates} source-backed company website profile(s); "
            f"checked={result.checked}; discovered={result.discovered}; errors={len(result.errors)}. "
            "Only URL/title/hash evidence is stored; full page content is not stored."
        )
        sec.mark_source_status(conn, source_name="Company IR page", status=status, reason=reason)
    print(
        f"Inspected {result.candidates} company website profile(s); "
        f"checked={result.checked}; discovered={result.discovered}; errors={len(result.errors)}."
    )
    for error in result.errors[:10]:
        print(f"- error {error.ticker} {error.discovered_url or error.website}: {error.reason}")


def cmd_check_ir_url(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.5)
    with open_db(db_path) as conn:
        url = args.url or ir.get_ir_url(conn, ticker)
        if not url:
            raise SystemExit(f"No IR URL configured for {ticker}. Use set-ir-url first.")
        result = ir.check_ir_page(conn, client, ticker=ticker, url=url)
        sec.mark_source_status(conn, source_name="Company IR page", status="ok")
    changed = "changed" if result.changed else "unchanged_or_first_check"
    print(f"Checked IR URL for {ticker}: {changed}; title={result.title}")


def cmd_check_ir_pages(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.5)
    with open_db(db_path) as conn:
        result = ir.check_ir_pages(conn, client, ticker=args.ticker, limit=args.limit)
        status = "degraded" if result.errors else "ok"
        reason = (
            f"Checked {result.checked} active IR page(s); changed={result.changed}; errors={len(result.errors)}."
        )
        sec.mark_source_status(conn, source_name="Company IR page", status=status, reason=reason)
    print(
        f"Checked {result.checked} active IR page(s); "
        f"changed={result.changed}; errors={len(result.errors)}."
    )
    for error in result.errors[:10]:
        print(f"- error {error.ticker} {error.url}: {error.error}")


def cmd_fetch_sec_rss(args: argparse.Namespace, settings: Settings) -> None:
    _require_sec_user_agent(settings)
    db_path = _db_path(args, settings)
    forms = _parse_forms(args.forms)
    client = HttpClient(user_agent=settings.sec_user_agent)
    with open_db(db_path) as conn:
        events = sec_rss.fetch_current_filings(client, count=args.limit)
        result = sec_rss.store_rss_events(conn, events, forms=forms)
        sec.mark_source_status(conn, source_name="SEC latest filings Atom feed", status="ok")
    print(
        f"Stored/updated {result.events_seen} SEC RSS filing event(s); "
        f"queued {result.queued} mapped filing(s)."
    )


def cmd_list_filing_queue(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, ticker, cik, form, accession_number, status, filing_url, queued_at
            FROM filing_queue
            WHERE (? = 'all' OR status = ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (args.status, args.status, args.limit),
        ).fetchall()
    if not rows:
        print("No filing queue rows matched.")
        return
    for row in rows:
        print(
            f"#{row['id']} {row['status']} {row['ticker'] or 'NA'} "
            f"{row['form'] or 'NA'} {row['accession_number'] or 'NA'} "
            f"{row['filing_url']}"
        )


def cmd_process_filing_queue(args: argparse.Namespace, settings: Settings) -> None:
    _require_sec_user_agent(settings)
    db_path = _db_path(args, settings)
    forms = _parse_forms(args.forms)
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
            (1 if not forms else 0, *sorted(forms), args.limit),
        ).fetchall()
        if not rows:
            print("No queued filing rows matched.")
            return
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
    print(
        f"Processed {processed} queued filing(s), errors={errors}, "
        f"stored {signals_stored} AI signal candidate(s) and "
        f"{risk_flags_stored} risk flag candidate(s), plus "
        f"{insider_transactions_stored} insider transaction event(s)."
    )


def cmd_import_valuation_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = valuation.import_valuation_csv(conn, args.csv)
        sec.mark_source_status(conn, source_name="Manual valuation CSV", status="ok")
    print(f"Imported {count} valuation snapshot(s).")


def cmd_import_peer_valuation_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = peer_valuation.import_peer_valuation_csv(conn, args.csv)
        sec.mark_source_status(conn, source_name="Manual peer valuation CSV", status="ok")
    print(f"Imported {count} peer valuation comparison record(s).")


def cmd_import_guidance_events_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = guidance.import_guidance_events_csv(conn, args.csv)
        sec.mark_source_status(conn, source_name="Manual guidance events CSV", status="ok")
    print(f"Imported {count} management guidance event(s).")


def cmd_import_transcript_snippets_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        result = transcripts.import_transcript_snippets_csv(
            conn,
            args.csv,
            scan_ai=not args.no_ai_scan,
            scan_risks=not args.no_risk_scan,
        )
        sec.mark_source_status(
            conn,
            source_name="Manual transcript snippets CSV",
            status="ok",
            reason="Imported source-backed transcript snippets; snippets require original-source review.",
        )
    print(
        f"Imported {result.snippets_imported} transcript snippet(s), "
        f"stored {result.ai_signals_stored} AI signal candidate(s) and "
        f"{result.risk_flags_stored} risk flag candidate(s)."
    )


def cmd_import_finra_short_sale_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = finra.import_short_sale_volume_file(
            conn,
            args.csv,
            source_name=args.source_name,
            source_url=args.source_url,
        )
        sec.mark_source_status(
            conn,
            source_name=args.source_name,
            status="ok",
            reason="Imported FINRA daily short sale volume; this is not short interest.",
        )
    print(f"Imported {count} FINRA short sale volume record(s).")


def cmd_fetch_finra_short_sale_volume(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    if args.lookback_days <= 0:
        raise SystemExit("--lookback-days must be positive.")
    trade_date = None
    if args.date:
        try:
            trade_date = date.fromisoformat(args.date)
        except ValueError as exc:
            raise SystemExit("--date must use YYYY-MM-DD format.") from exc
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        try:
            result = (
                finra.fetch_finra_daily_short_sale_volume(
                    client,
                    trade_date=trade_date,
                    tickers=tickers or None,
                    file_code=args.file_code,
                )
                if trade_date
                else finra.fetch_latest_finra_daily_short_sale_volume(
                    client,
                    lookback_days=args.lookback_days,
                    tickers=tickers or None,
                    file_code=args.file_code,
                )
            )
            count = finra.upsert_short_sale_volume(conn, result.records)
        except Exception as exc:
            sec.mark_source_status(
                conn,
                source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
                status="degraded",
                reason=_short_message(str(exc), 500),
            )
            print(f"FINRA daily short sale volume unavailable: {exc}")
            return
        status = "ok" if count else "degraded"
        reason = (
            f"Fetched FINRA daily short sale volume for {result.trade_date}; "
            f"file_code={args.file_code.strip().upper()}; imported {count} row(s). "
            "This is not short interest."
        )
        if tickers:
            reason += " ticker_filter=" + ",".join(tickers) + "."
        sec.mark_source_status(
            conn,
            source_name=finra.DEFAULT_SHORT_SALE_SOURCE,
            status=status,
            reason=reason,
        )
    print(
        f"Imported {count} FINRA short sale volume record(s) from {result.source_url}; "
        "daily short sale volume is not short interest."
    )


def cmd_import_insider_transactions_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = insiders.import_insider_transactions_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Insider transactions CSV",
            status="ok",
            reason=f"Imported from {args.csv}; events are not automatically treated as risk conclusions.",
        )
    print(f"Imported {count} insider transaction event(s).")


def cmd_import_institutional_holdings_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = institutions.import_institutional_holding_events_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Institutional holding events CSV",
            status="ok",
            reason=f"Imported from {args.csv}; holding changes are review inputs, not investment conclusions.",
        )
    print(f"Imported {count} institutional holding event(s).")


def cmd_list_insider_transactions(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = insiders.list_insider_transactions(conn, ticker=args.ticker, limit=args.limit)
    if not rows:
        print("No insider transaction events found.")
        return
    for row in rows:
        shares = "NA" if row["transaction_shares"] is None else f"{row['transaction_shares']:g}"
        price = "NA" if row["transaction_price"] is None else f"{row['transaction_price']:g}"
        print(
            f"{row['transaction_date']} {row['ticker']} {row['owner_name']} "
            f"code={row['transaction_code'] or 'NA'} acquired_disposed={row['acquired_disposed_code'] or 'NA'} "
            f"shares={shares} price={price} url={row['source_url']}"
        )


def cmd_fetch_yahoo_quote(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    client = HttpClient(user_agent=_user_agent(settings))
    with open_db(db_path) as conn:
        try:
            snapshot = valuation.fetch_yahoo_snapshot(client, ticker)
        except Exception as exc:
            sec.mark_source_status(
                conn,
                source_name="Yahoo Finance endpoint prototype",
                status="degraded",
                reason=str(exc),
            )
            print(f"Yahoo Finance endpoint prototype unavailable for {ticker}: {exc}")
            return
        count = valuation.upsert_valuation_snapshots(conn, [snapshot])
        sec.mark_source_status(
            conn,
            source_name="Yahoo Finance endpoint prototype",
            status="ok",
            reason="Prototype-only source; not a production-critical data source.",
        )
    print(f"Stored {count} Yahoo quote valuation snapshot(s) for {ticker}.")


def cmd_fetch_eastmoney_quote(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        try:
            snapshot = valuation.fetch_eastmoney_quote_snapshot(client, ticker)
        except Exception as exc:
            reason = _short_message(str(exc), 500)
            sec.mark_source_status(
                conn,
                source_name=valuation.EASTMONEY_US_QUOTE_SOURCE_NAME,
                status="degraded",
                reason=reason,
            )
            record_source_failure(
                conn,
                source_name=valuation.EASTMONEY_US_QUOTE_SOURCE_NAME,
                ticker=ticker,
                endpoint="quote",
                reason=f"{ticker} Eastmoney US quote: {reason}",
                source_url=valuation.eastmoney_quote_url(ticker),
            )
            print(f"Eastmoney US quote API unavailable for {ticker}: {exc}")
            return
        count = valuation.upsert_valuation_snapshots(conn, [snapshot])
        resolved = mark_source_success(
            conn,
            source_name=valuation.FMP_QUOTE_SOURCE_NAME,
            ticker=ticker,
            endpoint="quote",
        )
        sec.mark_source_status(
            conn,
            source_name=valuation.EASTMONEY_US_QUOTE_SOURCE_NAME,
            status="ok",
            reason=(
                f"Stored {count} public delayed quote valuation snapshot(s) for {ticker}; "
                f"resolved_open_fmp_quote_failures={resolved}. "
                "Source is a fallback for FMP quote entitlement gaps and is not company fundamentals."
            ),
        )
    print(
        f"Stored {count} Eastmoney US quote valuation snapshot(s) for {ticker}; "
        f"resolved_open_fmp_quote_failures={resolved}."
    )


def cmd_refresh_eastmoney_quotes_from_failures(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    db_path = _db_path(args, settings)
    explicit_tickers = _parse_ticker_args(args.ticker)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        tickers = _select_fmp_quote_failure_tickers(conn, tickers=explicit_tickers, limit=args.limit)
        if not tickers:
            print("No FMP quote failure tickers selected for Eastmoney fallback.")
            return
        snapshots: list[valuation.ValuationSnapshot] = []
        failures: list[str] = []
        for ticker in tickers:
            try:
                snapshots.append(valuation.fetch_eastmoney_quote_snapshot(client, ticker))
            except Exception as exc:
                reason = _short_message(str(exc), 500)
                failures.append(f"{ticker}: {reason}")
                record_source_failure(
                    conn,
                    source_name=valuation.EASTMONEY_US_QUOTE_SOURCE_NAME,
                    ticker=ticker,
                    endpoint="quote",
                    reason=f"{ticker} Eastmoney US quote: {reason}",
                    source_url=valuation.eastmoney_quote_url(ticker),
                )
        written = valuation.upsert_valuation_snapshots(conn, snapshots)
        resolved = 0
        for snapshot in snapshots:
            resolved += mark_source_success(
                conn,
                source_name=valuation.FMP_QUOTE_SOURCE_NAME,
                ticker=snapshot.ticker,
                endpoint="quote",
            )
        if snapshots and not failures:
            status = "ok"
        elif snapshots:
            status = "degraded"
        else:
            status = "unavailable"
        reason = (
            f"Eastmoney fallback considered {len(tickers)} ticker(s); "
            f"stored {written} valuation snapshot(s); resolved_open_fmp_quote_failures={resolved}. "
            "Source is public delayed quote context only and not company fundamentals."
        )
        if failures:
            reason += " Failures: " + _short_message("; ".join(failures), 300)
        sec.mark_source_status(
            conn,
            source_name=valuation.EASTMONEY_US_QUOTE_SOURCE_NAME,
            status=status,
            reason=_short_message(reason, 500),
        )
    print(
        f"Eastmoney fallback considered {len(tickers)} ticker(s); "
        f"stored {written} valuation snapshot(s); resolved_open_fmp_quote_failures={resolved}; "
        f"failures={len(failures)}."
    )
    for failure in failures[:5]:
        print(f"failure: {failure}")


def cmd_fetch_fmp_quote(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    with open_db(db_path) as conn:
        if not settings.fmp_api_key:
            sec.mark_source_status(
                conn,
                source_name="Financial Modeling Prep quote API",
                status="unavailable",
                reason="FMP_API_KEY is not configured.",
            )
            print("FMP_API_KEY is not configured; recorded source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings))
        try:
            snapshot = valuation.fetch_fmp_quote_snapshot(client, ticker, settings.fmp_api_key)
        except Exception as exc:
            sec.mark_source_status(
                conn,
                source_name="Financial Modeling Prep quote API",
                status="degraded",
                reason=str(exc),
            )
            print(f"Financial Modeling Prep quote API unavailable for {ticker}: {exc}")
            return
        count = valuation.upsert_valuation_snapshots(conn, [snapshot])
        sec.mark_source_status(conn, source_name="Financial Modeling Prep quote API", status="ok")
    print(f"Stored {count} FMP quote valuation snapshot(s) for {ticker}.")


def cmd_import_company_profiles(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = profile.import_company_profiles_csv(conn, args.csv)
        sec.mark_source_status(conn, source_name="Manual company profile CSV", status="ok")
    print(f"Imported {count} company profile record(s).")


def cmd_fetch_fmp_profile(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    with open_db(db_path) as conn:
        if not settings.fmp_api_key:
            sec.mark_source_status(
                conn,
                source_name="Financial Modeling Prep profile API",
                status="unavailable",
                reason="FMP_API_KEY is not configured.",
            )
            print("FMP_API_KEY is not configured; recorded profile source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings))
        try:
            company_profile = profile.fetch_fmp_profile(client, ticker, settings.fmp_api_key)
        except Exception as exc:
            sec.mark_source_status(
                conn,
                source_name="Financial Modeling Prep profile API",
                status="degraded",
                reason=str(exc),
            )
            print(f"Financial Modeling Prep profile API unavailable for {ticker}: {exc}")
            return
        count = profile.upsert_company_profiles(conn, [company_profile])
        sec.mark_source_status(conn, source_name="Financial Modeling Prep profile API", status="ok")
    print(f"Stored {count} FMP company profile record(s) for {ticker}.")


def cmd_enrich_fmp_profiles(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        candidates = profile.select_ai_profile_enrichment_candidates(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            include_existing=args.include_existing,
        )
        if args.dry_run:
            for candidate in candidates:
                terms = ",".join(candidate.matched_terms) or "explicit_ticker"
                print(
                    f"{candidate.ticker}\t{candidate.score}\t{terms}\t"
                    f"{candidate.company_name}\t{candidate.source}"
                )
            print(f"Selected {len(candidates)} FMP profile enrichment candidate(s); dry-run only.")
            return
        if not settings.fmp_api_key:
            sec.mark_source_status(
                conn,
                source_name=profile.FMP_PROFILE_SOURCE_NAME,
                status="unavailable",
                reason="FMP_API_KEY is not configured; profile enrichment skipped.",
            )
            print("FMP_API_KEY is not configured; recorded profile enrichment source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
        result = profile.enrich_fmp_profile_candidates(
            conn,
            client,
            api_key=settings.fmp_api_key,
            tickers=tickers or None,
            limit=args.limit,
            include_existing=args.include_existing,
            write_valuation_snapshot=not args.no_valuation_snapshot,
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
    print(
        f"Selected {result.candidates_considered} candidate(s); "
        f"stored profiles={result.profiles_written}, "
        f"profile_market_cap_valuation_snapshots={result.valuation_snapshots_written}, "
        f"failures={len(result.failures)}."
    )
    for failure in result.failures[:5]:
        print(f"- {failure}")


def cmd_import_ai_tags(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = import_ai_tags_csv(conn, args.csv)
        sec.mark_source_status(conn, source_name="Manual AI industry tags CSV", status="ok")
    print(f"Imported {count} AI industry-chain tag record(s).")


def cmd_tag_ai_chain(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    with open_db(db_path) as conn:
        tags = infer_tags_for_ticker(conn, ticker)
        count = replace_inferred_tags_for_ticker(conn, ticker, tags)
    print(f"Stored {count} inferred AI industry-chain tag candidate(s) for {ticker}.")


def cmd_infer_ai_chain_tags(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        result = infer_ai_chain_tags(
            conn,
            tickers=tickers or None,
            limit=args.limit,
        )
        sec.mark_source_status(
            conn,
            source_name="Automated AI industry tag inference",
            status="ok",
            reason=(
                f"Considered {result.tickers_considered} local candidate ticker(s); "
                f"tagged {result.tickers_tagged}; wrote {result.tags_written} inferred tag(s)."
            ),
        )
    print(
        f"Considered {result.tickers_considered} local candidate ticker(s); "
        f"tagged {result.tickers_tagged}; wrote {result.tags_written} inferred AI industry-chain tag(s)."
    )


def cmd_list_ai_tags(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    with open_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT tag, confidence, source_type, source_url, evidence_snippet
            FROM ai_industry_tags
            WHERE ticker = ?
            ORDER BY confidence DESC, tag
            LIMIT ?
            """,
            (ticker, args.limit),
        ).fetchall()
    if not rows:
        print(f"No AI industry-chain tags found for {ticker}.")
        return
    for row in rows:
        print(
            f"{ticker} {row['tag']} confidence={row['confidence']:.2f} "
            f"source={row['source_type']} url={row['source_url']}"
        )


def cmd_extract_catalysts(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper() if args.ticker else None
    with open_db(db_path) as conn:
        candidates = extract_catalysts_from_evidence(conn, ticker=ticker, limit=args.limit)
        inserted = store_catalysts(conn, candidates)
    scope = ticker or "all tickers"
    print(f"Extracted {len(candidates)} catalyst candidate(s) for {scope}; inserted {inserted}.")


def cmd_list_catalysts(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper()
    with open_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT catalyst_type, catalyst_date, confidence, status, source_url, description
            FROM catalysts
            WHERE ticker = ?
            ORDER BY COALESCE(catalyst_date, ''), confidence DESC
            LIMIT ?
            """,
            (ticker, args.limit),
        ).fetchall()
    if not rows:
        print(f"No catalysts found for {ticker}.")
        return
    for row in rows:
        print(
            f"{ticker} {row['catalyst_type']} {row['catalyst_date'] or 'NA'} "
            f"confidence={row['confidence']:.2f} status={row['status']} url={row['source_url']}"
        )


def cmd_import_catalyst_calendar_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        inserted = catalyst_calendar.import_catalyst_calendar_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Manual catalyst calendar CSV",
            status="ok",
            reason=f"Imported from {args.csv}.",
        )
    print(f"Imported {inserted} new catalyst calendar event(s).")


def cmd_fetch_fmp_earnings_calendar(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit_per_ticker <= 0:
        raise SystemExit("--limit-per-ticker must be positive.")
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    if not tickers:
        raise SystemExit("--ticker is required.")
    if not settings.fmp_api_key:
        with open_db(db_path) as conn:
            sec.mark_source_status(
                conn,
                source_name=catalyst_calendar.FMP_EARNINGS_CALENDAR_SOURCE_NAME,
                status="unavailable",
                reason="FMP_API_KEY is not configured.",
            )
        print("FMP_API_KEY is not configured; recorded FMP earnings calendar source as unavailable.")
        return
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        try:
            entries = catalyst_calendar.fetch_fmp_earnings_calendar_entries(
                client,
                tickers,
                settings.fmp_api_key,
                from_date=args.from_date,
                to_date=args.to_date,
                limit_per_ticker=args.limit_per_ticker,
            )
            inserted = catalyst_calendar.upsert_catalyst_calendar_entries(conn, entries)
            sec.mark_source_status(
                conn,
                source_name=catalyst_calendar.FMP_EARNINGS_CALENDAR_SOURCE_NAME,
                status="ok" if entries else "degraded",
                reason=f"Fetched {len(entries)} earnings calendar event(s); inserted {inserted}.",
            )
        except Exception as exc:
            message = str(exc).replace(settings.fmp_api_key, "***")
            sec.mark_source_status(
                conn,
                source_name=catalyst_calendar.FMP_EARNINGS_CALENDAR_SOURCE_NAME,
                status="degraded",
                reason=message[:500],
            )
            print(f"FMP earnings calendar unavailable: {message}")
            return
    print(f"Stored {len(entries)} FMP earnings calendar event(s); inserted {inserted}.")


def cmd_list_upcoming_catalysts(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = catalyst_calendar.list_upcoming_catalysts(
            conn,
            ticker=args.ticker,
            from_date=args.from_date,
            days=args.days,
            limit=args.limit,
        )
    if not rows:
        print("No upcoming catalysts found.")
        return
    for row in rows:
        print(
            f"{row['catalyst_date']} {row['ticker']} {row['catalyst_type']} "
            f"confidence={row['confidence']:.2f} status={row['status']} url={row['source_url']}"
        )


def cmd_build_watchlist(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = build_watchlist(conn, limit=args.limit, min_score=args.min_score)
        write_watchlist_csv(rows, args.output)
    print(f"Wrote {len(rows)} watchlist row(s) to {args.output}.")


def cmd_build_evidence_audit(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_evidence_audit(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
        )
        write_evidence_audit_csv(rows, args.output)
    print(f"Wrote {len(rows)} evidence audit row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} score={row.score_total:g} coverage={row.evidence_coverage_score:g} "
            f"priority={row.review_priority} missing={row.missing_core_evidence or 'none'}"
        )


def cmd_build_thesis_checklist(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_thesis_checklist(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
        )
        write_thesis_checklist_csv(rows, args.output)
    print(f"Wrote {len(rows)} thesis checklist row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} gate={row.thesis_gate} score={row.score_total:g} "
            f"actions={row.next_source_actions}"
        )


def cmd_build_review_queue(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_review_queue(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            cards_dir=args.cards_dir,
        )
        write_review_queue_csv(rows, args.output)
    print(f"Wrote {len(rows)} review queue row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} bucket={row.review_bucket} frequency={row.tracking_frequency} "
            f"checks={row.required_review_checks}"
        )


def cmd_build_research_pool(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_research_pool(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            cards_dir=args.cards_dir,
        )
        write_research_pool_csv(rows, args.output)
    print(f"Wrote {len(rows)} research pool row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} layer={row.score_layer} status={row.research_pool_status} "
            f"score={row.score_total:g} action={row.next_action}"
        )


def cmd_build_ai_chain_coverage(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_ai_chain_coverage(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            top_limit=args.top_limit,
        )
        write_ai_chain_coverage_csv(rows, args.output)
    print(f"Wrote {len(rows)} AI chain coverage row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.chain_category} candidates={row.candidate_count} "
            f"source_blocked={row.source_blocked_count} top={row.top_tickers or 'none'}"
        )


def cmd_build_score_provenance(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_score_provenance(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            source_link_limit=args.source_link_limit,
        )
        write_score_provenance_csv(rows, args.output)
    print(f"Wrote {len(rows)} score provenance row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} trace={row.score_trace_status} score={row.score_total:g} "
            f"gate={row.thesis_gate}"
        )


def cmd_build_data_mining_leads(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.command_limit <= 0:
        raise SystemExit("--command-limit must be positive.")
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_data_mining_leads(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            command_limit=args.command_limit,
        )
        write_data_mining_leads_csv(rows, args.output)
    print(f"Wrote {len(rows)} data mining lead row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} priority={row.mining_priority} type={row.lead_type} "
            f"score={row.score_total:g}"
        )


def cmd_build_refresh_plan(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_refresh_plan(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
            as_of=args.as_of,
            include_scheduled=args.include_scheduled,
        )
        write_refresh_plan_csv(rows, args.output)
    print(f"Wrote {len(rows)} refresh plan row(s) to {args.output}.")
    for row in rows[:5]:
        print(
            f"{row.ticker} action={row.action_type} status={row.due_status} "
            f"source={row.source_name}"
        )


def cmd_build_source_failure_report(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = build_source_failure_report(
            conn,
            status=None if args.status == "all" else args.status,
            limit=args.limit,
        )
        write_source_failure_csv(rows, args.output)
    open_count = sum(1 for row in rows if row.status == "open")
    print(f"Wrote {len(rows)} source failure row(s) to {args.output}; open={open_count}.")
    for row in rows[:5]:
        ticker = row.ticker or "NA"
        print(f"{row.status} {ticker} {row.source_name} {row.endpoint} {row.failure_type}: {row.reason}")


def cmd_build_ops_report(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        content = build_ops_report(
            conn,
            review_limit=args.review_limit,
            source_limit=args.source_limit,
            evidence_limit=args.evidence_limit,
            source_input_import_path=args.source_input_import_report,
            source_input_plan_path=args.source_input_plan_report,
            source_input_limit=args.source_input_limit,
            source_input_plan_limit=args.source_input_plan_limit,
        )
    if args.stdout:
        print(content, end="")
    else:
        write_ops_report(content, args.output)
        print(f"Wrote operations report to {args.output}.")


def cmd_build_mvp_readiness(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = build_mvp_readiness(
            conn,
            reports_dir=args.reports_dir,
            design_doc=args.design_doc,
        )
        write_mvp_readiness_csv(rows, args.output)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    print(f"Wrote {len(rows)} MVP readiness row(s) to {args.output}; {summary}.")
    for row in [item for item in rows if item.status != "ready"][:5]:
        print(f"{row.area} status={row.status} action={row.next_action}")


def cmd_build_source_input_templates(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    readiness_statuses: dict[str, str] = {}
    if args.only_readiness_gaps:
        with open_db(db_path) as conn:
            readiness_statuses = {row.area: row.status for row in build_mvp_readiness(conn)}
    templates = select_source_input_templates(
        readiness_statuses=readiness_statuses,
        only_gaps=args.only_readiness_gaps,
    )
    templates = _filter_source_input_templates_for_args(templates, args.template)
    rows = write_source_input_template_pack(
        templates,
        output_dir=args.output_dir,
        manifest_path=args.manifest_output,
        overwrite=args.overwrite,
    )
    written = sum(1 for row in rows if row.write_status == "written")
    print(
        f"Wrote {len(rows)} source input template manifest row(s) to {args.manifest_output}; "
        f"templates_written={written}."
    )
    for row in rows[:5]:
        print(f"{row.template_name} {row.write_status} {row.template_path}")


def cmd_build_source_input_audit(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    readiness_statuses: dict[str, str] = {}
    if args.only_readiness_gaps:
        with open_db(db_path) as conn:
            readiness_statuses = {row.area: row.status for row in build_mvp_readiness(conn)}
    templates = select_source_input_templates(
        readiness_statuses=readiness_statuses,
        only_gaps=args.only_readiness_gaps,
    )
    templates = _filter_source_input_templates_for_args(templates, args.template)
    rows = build_source_input_audit(templates, input_dir=args.input_dir)
    write_source_input_audit_csv(rows, args.output)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.audit_status] = counts.get(row.audit_status, 0) + 1
    summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    print(f"Wrote {len(rows)} source input audit row(s) to {args.output}; {summary}.")
    for row in rows[:5]:
        print(f"{row.template_name} status={row.audit_status} rows={row.data_rows}")


def cmd_import_source_input_pack(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    readiness_statuses: dict[str, str] = {}
    with open_db(db_path) as conn:
        if args.only_readiness_gaps:
            readiness_statuses = {row.area: row.status for row in build_mvp_readiness(conn)}
        templates = select_source_input_templates(
            readiness_statuses=readiness_statuses,
            only_gaps=args.only_readiness_gaps,
        )
        templates = _filter_source_input_templates_for_args(templates, args.template)
        rows = import_source_input_pack(
            conn,
            templates,
            input_dir=args.input_dir,
            apply=args.apply,
            finra_source_url=args.finra_source_url,
            transcript_scan_ai=not args.no_transcript_ai_scan,
            transcript_scan_risks=not args.no_transcript_risk_scan,
        )
        write_source_input_import_csv(rows, args.output)
    counts: dict[str, int] = {}
    imported_rows = 0
    for row in rows:
        counts[row.import_status] = counts.get(row.import_status, 0) + 1
        imported_rows += row.rows_imported
    summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    mode = "apply" if args.apply else "dry-run"
    print(
        f"Wrote {len(rows)} source input import row(s) to {args.output}; "
        f"mode={mode}; imported_rows={imported_rows}; {summary}."
    )
    for row in rows[:5]:
        print(f"{row.template_name} status={row.import_status} rows_imported={row.rows_imported}")


def cmd_build_source_input_plan(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        readiness_statuses = {row.area: row.status for row in build_mvp_readiness(conn)}
    templates = select_source_input_templates(
        readiness_statuses=readiness_statuses,
        only_gaps=args.only_readiness_gaps,
    )
    templates = _filter_source_input_templates_for_args(templates, args.template)
    rows = build_source_input_plan(
        templates,
        input_dir=args.input_dir,
        readiness_statuses=readiness_statuses,
        audit_output=args.audit_output,
        import_output=args.import_output,
    )
    write_source_input_plan_csv(rows, args.output)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.audit_status] = counts.get(row.audit_status, 0) + 1
    summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
    print(f"Wrote {len(rows)} source input plan row(s) to {args.output}; {summary}.")
    for row in rows[:5]:
        print(f"{row.template_name} audit={row.audit_status} import={row.import_status}")


def cmd_snapshot_scores(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        result = snapshot_scores(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score=args.min_score,
        )
    print(
        f"Snapshotted {result.tickers_snapshotted} ticker score/review state row(s); "
        f"inserted {result.snapshots_written} new snapshot row(s) at {result.snapshot_at}."
    )


def cmd_list_score_snapshots(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = list_score_snapshots(conn, ticker=args.ticker, limit=args.limit)
    if not rows:
        print("No score snapshots found.")
        return
    for row in rows:
        run_text = f" run=#{row['run_id']}" if row["run_id"] is not None else ""
        print(
            f"{row['snapshot_at']} {row['ticker']}{run_text} "
            f"score={row['score_total']} bucket={row['review_bucket']} "
            f"coverage={row['evidence_coverage_score']} priority={row['review_priority']}"
        )


def cmd_build_snapshot_change_report(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        rows = build_snapshot_change_report(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            min_score_delta=args.min_score_delta,
            changed_only=args.changed_only,
        )
        write_snapshot_change_csv(rows, args.output)
    print(f"Wrote {len(rows)} snapshot change row(s) to {args.output}.")
    for row in rows[:5]:
        delta = "NA" if row.score_delta is None else f"{row.score_delta:g}"
        print(
            f"{row.ticker} change={row.change_type} attention={row.attention_level} "
            f"score_delta={delta}"
        )


def cmd_build_backtest_report(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    buckets = [bucket.strip() for item in args.bucket for bucket in item.split(",") if bucket.strip()]
    with open_db(db_path) as conn:
        rows = build_snapshot_backtest(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            horizon_bars=args.horizon_bars,
            min_score=args.min_score,
            buckets=buckets or None,
            include_incomplete=not args.complete_only,
        )
        write_backtest_csv(rows, args.output)
    complete_count = sum(1 for row in rows if row.outcome_status == "complete")
    print(
        f"Wrote {len(rows)} backtest row(s) to {args.output}; "
        f"{complete_count} complete outcome row(s)."
    )


def cmd_import_portfolio_positions_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = import_portfolio_positions_csv(
            conn,
            args.csv,
            default_portfolio=args.portfolio,
        )
        sec.mark_source_status(
            conn,
            source_name="Portfolio positions CSV",
            status="ok",
            reason=f"Imported {count} local portfolio position row(s) from {args.csv}.",
        )
    print(f"Imported {count} portfolio position row(s).")


def cmd_build_portfolio_risk_report(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = build_portfolio_risk_report(
            conn,
            portfolio_name=args.portfolio,
            limit=args.limit,
        )
        write_portfolio_risk_csv(rows, args.output)
    print(f"Wrote {len(rows)} portfolio risk review row(s) to {args.output}.")


def cmd_run_pipeline(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    options = PipelineOptions(
        skip_network=args.skip_network,
        universe_limit=args.universe_limit,
        sec_rss_limit=args.sec_rss_limit,
        sec_submissions_tickers=tuple(_parse_ticker_args(args.sec_submissions_ticker)),
        sec_submissions_limit=args.sec_submissions_limit,
        sec_companyfacts_failure_limit=args.sec_companyfacts_failure_limit,
        process_filing_limit=args.process_filing_limit,
        ir_limit=args.ir_limit,
        news_limit=args.news_limit,
        gdelt_queries=tuple(query.strip() for query in args.gdelt_query if query.strip()),
        gdelt_limit=args.gdelt_limit,
        gdelt_timespan=args.gdelt_timespan,
        macro_limit=args.macro_limit,
        finra_short_sale_tickers=tuple(_parse_ticker_args(args.finra_short_sale_ticker)),
        finra_short_sale_lookback_days=args.finra_short_sale_lookback_days,
        finra_short_sale_file_code=args.finra_short_sale_file_code,
        market_source_dir=args.market_source_dir or settings.local_market_source_dir,
        market_anomaly_limit=args.market_anomaly_limit,
        market_anomaly_lookback=args.market_anomaly_lookback,
        catalyst_limit=args.catalyst_limit,
        risk_limit=args.risk_limit,
        fmp_profile_enrichment_limit=args.fmp_profile_enrichment_limit,
        fmp_profile_enrichment_include_existing=args.fmp_profile_enrichment_include_existing,
        profile_valuation_snapshot=not args.no_profile_valuation_snapshot,
        ai_tag_limit=args.ai_tag_limit,
        ai_context_limit=args.ai_context_limit,
        expectation_gap_limit=args.expectation_gap_limit,
        watchlist_limit=args.watchlist_limit,
        watchlist_min_score=args.min_score,
        watchlist_output=args.output,
        evidence_audit_output=args.evidence_audit_output,
        thesis_checklist_output=args.thesis_checklist_output,
        review_queue_output=args.review_queue_output,
        research_pool_output=args.research_pool_output,
        ai_chain_coverage_output=args.ai_chain_coverage_output,
        score_provenance_output=args.score_provenance_output,
        data_mining_leads_output=args.data_mining_leads_output,
        refresh_plan_output=args.refresh_plan_output,
        source_failure_output=args.source_failure_output,
        source_input_dir=args.source_input_dir,
        source_input_audit_output=args.source_input_audit_output,
        source_input_import_output=args.source_input_import_output,
        source_input_plan_output=args.source_input_plan_output,
        change_report_output=args.change_report_output,
        ops_report_output=args.ops_report_output,
        mvp_readiness_output=args.mvp_readiness_output,
        mvp_readiness_design_doc=args.design_doc,
        cards_dir=args.cards_dir,
        card_index_output=args.card_index_output,
        card_artifact_audit_output=args.card_artifact_audit_output,
        forms=args.forms,
        min_market_cap=None if args.no_market_cap_filter else args.min_market_cap,
        continue_on_error=not args.stop_on_error,
    )
    result = run_pipeline(db_path, settings, options=options)
    print(f"Pipeline run #{result.run_id} status={result.status}")
    for step in result.steps:
        changed = "NA" if step.records_changed is None else str(step.records_changed)
        print(f"- {step.step_name}: {step.status} records={changed} {step.message}")
        if step.error:
            print(f"  error={step.error}")


def cmd_refresh_ai_seed_data(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.analyst_limit <= 0:
        raise SystemExit("--analyst-limit must be positive.")
    if args.financial_limit <= 0:
        raise SystemExit("--financial-limit must be positive.")
    if args.finra_lookback_days <= 0:
        raise SystemExit("--finra-lookback-days must be positive.")
    db_path = _db_path(args, settings)
    explicit_tickers = _parse_ticker_args(args.ticker)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        include_sec_facts = not args.skip_sec_facts and bool(settings.sec_user_agent)
        if not args.skip_sec_facts and not settings.sec_user_agent:
            sec.mark_source_status(
                conn,
                source_name="SEC companyfacts",
                status="unavailable",
                reason="SEC_USER_AGENT is not configured; seed refresh skipped SEC companyfacts.",
            )
        tickers = select_ai_seed_tickers(conn, tickers=explicit_tickers, limit=args.limit)
        if not tickers:
            print("No AI seed tickers found. Load AI industry tags or pass --ticker.")
            return
        result = refresh_ai_seed_data(
            conn,
            client,
            tickers=tickers,
            fmp_api_key=settings.fmp_api_key,
            include_profile=not args.skip_profile,
            include_quote=not args.skip_quote,
            include_analyst=not args.skip_analyst,
            include_finra=not args.skip_finra,
            include_sec_facts=include_sec_facts,
            include_fmp_financials=not args.skip_fmp_financials,
            analyst_period=args.analyst_period,
            analyst_limit=args.analyst_limit,
            financial_period=args.financial_period,
            financial_limit=args.financial_limit,
            finra_lookback_days=args.finra_lookback_days,
            finra_file_code=args.finra_file_code,
        )
    print(
        f"Refreshed {result.tickers_considered} AI seed ticker(s): "
        f"financial_fact_periods={result.financial_fact_periods}, "
        f"profiles={result.profile_records}, valuations={result.valuation_snapshots}, "
        f"analyst_events={result.analyst_events}, finra_rows={result.finra_records}, "
        f"ai_tags={result.ai_tags_written}, failures={len(result.failures)}."
    )
    for failure in result.failures[:5]:
        print(f"- {failure}")


def cmd_refresh_free_source_context(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.plan_limit <= 0:
        raise SystemExit("--plan-limit must be positive.")
    if args.finra_lookback_days <= 0:
        raise SystemExit("--finra-lookback-days must be positive.")
    if args.news_limit <= 0:
        raise SystemExit("--news-limit must be positive.")
    if args.gdelt_limit <= 0:
        raise SystemExit("--gdelt-limit must be positive.")
    if args.extraction_limit <= 0:
        raise SystemExit("--extraction-limit must be positive.")
    db_path = _db_path(args, settings)
    explicit_tickers = _parse_ticker_args(args.ticker)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        tickers = select_free_source_refresh_tickers(
            conn,
            tickers=explicit_tickers,
            limit=args.limit,
            plan_limit=args.plan_limit,
        )
        if not tickers:
            print("No free-source refresh tickers found. Build refresh_plan or pass --ticker.")
            return
        result = refresh_free_source_context(
            conn,
            client,
            tickers=tickers,
            include_finra=not args.skip_finra,
            include_news=not args.skip_news,
            include_gdelt=args.include_gdelt,
            extract_catalysts=not args.skip_catalysts,
            extract_risks=not args.skip_risks,
            finra_lookback_days=args.finra_lookback_days,
            finra_file_code=args.finra_file_code,
            news_limit=args.news_limit,
            gdelt_limit=args.gdelt_limit,
            gdelt_timespan=args.gdelt_timespan,
            extraction_limit=args.extraction_limit,
        )
    print(
        f"Refreshed free/public source context for {result.tickers_considered} ticker(s): "
        f"finra_rows={result.finra_records}, rss_events={result.news_events_seen}, "
        f"rss_mapped={result.news_events_mapped}, gdelt_events={result.gdelt_events_seen}, "
        f"gdelt_mapped={result.gdelt_events_mapped}, catalysts_inserted={result.catalysts_inserted}, "
        f"risk_flags_inserted={result.risk_flags_inserted}, failures={len(result.failures)}."
    )
    for failure in result.failures[:5]:
        print(f"- {failure}")


def cmd_list_pipeline_runs(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        if args.run_id is not None:
            steps = list_pipeline_steps(conn, run_id=args.run_id)
            if not steps:
                print(f"No pipeline steps found for run #{args.run_id}.")
                return
            for step in steps:
                changed = "NA" if step["records_changed"] is None else str(step["records_changed"])
                print(
                    f"#{args.run_id} {step['step_name']} {step['status']} "
                    f"records={changed} message={step['message']}"
                )
                if step["error"]:
                    print(f"  error={step['error']}")
            return
        runs = list_pipeline_runs(conn, limit=args.limit)
    if not runs:
        print("No pipeline runs found.")
        return
    for row in runs:
        print(
            f"#{row['id']} {row['status']} {row['started_at']} -> "
            f"{row['completed_at'] or 'running'} {row['summary'] or ''}"
        )


def cmd_detect_market_sources(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    base_dir = args.base_dir or settings.local_market_source_dir
    sources = market.detect_local_market_sources(base_dir)
    with open_db(db_path) as conn:
        for source in sources:
            status = "ok" if source.exists else "unavailable"
            reason = (
                f"Detected local source at {source.path}; no read/write performed."
                if source.exists
                else f"Local source not found at {source.path}."
            )
            sec.mark_source_status(conn, source_name=source.source_name, status=status, reason=reason)
    for source in sources:
        found = "found" if source.exists else "missing"
        print(f"{source.source_name}: {found} ({source.path})")


def cmd_import_market_confirmation_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = market.import_market_confirmation_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Local market confirmation CSV",
            status="ok",
            reason=f"Imported from {args.csv}.",
        )
    print(f"Imported {count} market confirmation signal(s).")


def cmd_import_market_price_bars_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = market.import_market_price_bars_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Local market price bars CSV",
            status="ok",
            reason=f"Imported {count} price/volume bar(s) from {args.csv}.",
        )
    print(f"Imported {count} market price bar(s).")


def cmd_fetch_fmp_market_bars(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    benchmark = args.benchmark.strip().upper()
    if not tickers:
        raise SystemExit("--ticker is required.")
    if args.lookback_days <= 0:
        raise SystemExit("--lookback-days must be positive.")
    if not benchmark:
        raise SystemExit("--benchmark must not be empty.")
    with open_db(db_path) as conn:
        if not settings.fmp_api_key:
            sec.mark_source_status(
                conn,
                source_name=market.FMP_HISTORICAL_PRICE_SOURCE_NAME,
                status="unavailable",
                reason="FMP_API_KEY is not configured.",
            )
            for ticker in tickers:
                record_source_failure(
                    conn,
                    source_name=market.FMP_HISTORICAL_PRICE_SOURCE_NAME,
                    ticker=ticker,
                    endpoint="historical-price-eod",
                    reason="FMP_API_KEY is not configured.",
                    source_url=_fmp_market_bars_source_url(
                        ticker,
                        benchmark=benchmark,
                        lookback_days=args.lookback_days,
                    ),
                    failure_type="missing_configuration",
                )
            print("FMP_API_KEY is not configured; recorded market price source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings))
        total = 0
        failures: list[str] = []
        for ticker in tickers:
            try:
                bars = market.fetch_fmp_market_price_bars(
                    client,
                    ticker,
                    settings.fmp_api_key,
                    benchmark_ticker=benchmark,
                    lookback_days=args.lookback_days,
                )
            except Exception as exc:
                failure = f"{ticker}: {exc}"
                failures.append(failure)
                record_source_failure(
                    conn,
                    source_name=market.FMP_HISTORICAL_PRICE_SOURCE_NAME,
                    ticker=ticker,
                    endpoint="historical-price-eod",
                    reason=failure,
                    source_url=_fmp_market_bars_source_url(
                        ticker,
                        benchmark=benchmark,
                        lookback_days=args.lookback_days,
                    ),
                )
                continue
            total += market.upsert_market_price_bars(conn, bars)
            mark_source_success(
                conn,
                source_name=market.FMP_HISTORICAL_PRICE_SOURCE_NAME,
                ticker=ticker,
                endpoint="historical-price-eod",
            )
        if failures:
            status = "degraded" if total else "unavailable"
            reason = _short_message("; ".join(failures), 500)
            sec.mark_source_status(
                conn,
                source_name=market.FMP_HISTORICAL_PRICE_SOURCE_NAME,
                status=status,
                reason=reason,
            )
            print(
                f"Stored {total} FMP market price bar(s); "
                f"{len(failures)} ticker(s) failed: {reason}"
            )
            return
        sec.mark_source_status(
            conn,
            source_name=market.FMP_HISTORICAL_PRICE_SOURCE_NAME,
            status="ok",
            reason=(
                f"Stored {total} EOD price/volume bar(s) for {len(tickers)} ticker(s); "
                f"benchmark={benchmark}; lookback_days={args.lookback_days}."
            ),
        )
    print(
        f"Stored {total} FMP market price bar(s) for {len(tickers)} ticker(s); "
        f"benchmark={benchmark}."
    )


def cmd_fetch_yahoo_market_bars(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    benchmark = args.benchmark.strip().upper()
    if not tickers:
        raise SystemExit("--ticker is required.")
    if args.lookback_days <= 0:
        raise SystemExit("--lookback-days must be positive.")
    if not benchmark:
        raise SystemExit("--benchmark must not be empty.")
    client = HttpClient(user_agent=_user_agent(settings))
    with open_db(db_path) as conn:
        total = 0
        failures: list[str] = []
        for ticker in tickers:
            try:
                bars = market.fetch_yahoo_market_price_bars(
                    client,
                    ticker,
                    benchmark_ticker=benchmark,
                    lookback_days=args.lookback_days,
                )
            except Exception as exc:
                failure = f"{ticker}: {exc}"
                failures.append(failure)
                record_source_failure(
                    conn,
                    source_name=market.YAHOO_CHART_SOURCE_NAME,
                    ticker=ticker,
                    endpoint="chart-1d",
                    reason=failure,
                    source_url=_yahoo_market_bars_source_url(
                        ticker,
                        benchmark=benchmark,
                        lookback_days=args.lookback_days,
                    ),
                )
                continue
            total += market.upsert_market_price_bars(conn, bars)
            mark_source_success(
                conn,
                source_name=market.YAHOO_CHART_SOURCE_NAME,
                ticker=ticker,
                endpoint="chart-1d",
            )
        if failures:
            status = "degraded" if total else "unavailable"
            reason = _short_message(
                "Prototype-only source; "
                f"stored {total} EOD price/volume bar(s); failures: {'; '.join(failures)}",
                500,
            )
            sec.mark_source_status(
                conn,
                source_name=market.YAHOO_CHART_SOURCE_NAME,
                status=status,
                reason=reason,
            )
            print(
                f"Stored {total} Yahoo prototype market price bar(s); "
                f"{len(failures)} ticker(s) failed: {reason}"
            )
            return
        sec.mark_source_status(
            conn,
            source_name=market.YAHOO_CHART_SOURCE_NAME,
            status="ok",
            reason=(
                "Prototype-only source; "
                f"stored {total} EOD price/volume bar(s) for {len(tickers)} ticker(s); "
                f"benchmark={benchmark}; lookback_days={args.lookback_days}."
            ),
        )
    print(
        f"Stored {total} Yahoo prototype market price bar(s) for {len(tickers)} ticker(s); "
        f"benchmark={benchmark}."
    )


def cmd_refresh_market_bars_from_leads(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.lead_limit <= 0:
        raise SystemExit("--lead-limit must be positive.")
    if args.lookback_days <= 0:
        raise SystemExit("--lookback-days must be positive.")
    if args.anomaly_lookback <= 0:
        raise SystemExit("--anomaly-lookback must be positive.")
    benchmark = args.benchmark.strip().upper()
    if not benchmark:
        raise SystemExit("--benchmark must not be empty.")
    db_path = _db_path(args, settings)
    explicit_tickers = _parse_ticker_args(args.ticker)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    with open_db(db_path) as conn:
        tickers = select_market_bar_fallback_tickers(
            conn,
            tickers=explicit_tickers,
            limit=args.limit,
            lead_limit=args.lead_limit,
            min_score=args.min_score,
        )
        if not tickers:
            print("No P0 market-bar fallback tickers found. Build data-mining leads or pass --ticker.")
            return
        result = refresh_yahoo_market_bars_from_leads(
            conn,
            client,
            tickers=tickers,
            benchmark_ticker=benchmark,
            lookback_days=args.lookback_days,
            detect_anomalies=not args.skip_anomalies,
            anomaly_lookback=args.anomaly_lookback,
        )
    print(
        f"Refreshed market bars from data-mining leads for {result.tickers_considered} ticker(s): "
        f"bars_written={result.bars_written}, anomaly_tickers_checked={result.anomaly_tickers_checked}, "
        f"anomaly_signals_written={result.anomaly_signals_written}, failures={len(result.failures)}."
    )
    for failure in result.failures[:5]:
        print(f"- {failure}")


def cmd_detect_market_anomalies(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        result = market.detect_market_anomalies(
            conn,
            tickers=tickers or None,
            limit=args.limit,
            lookback=args.lookback,
            min_volume_multiple=args.min_volume_multiple,
            min_gap_pct=args.min_gap_pct,
            min_relative_strength_pct=args.min_relative_strength_pct,
        )
        status = "ok" if result.tickers_checked else "unavailable"
        reason = (
            f"Checked {result.tickers_checked} ticker(s); wrote {result.signals_written} signal(s)."
            if result.tickers_checked
            else "No local market price bars are available for anomaly detection."
        )
        sec.mark_source_status(conn, source_name="Local market anomaly detector", status=status, reason=reason)
    print(
        f"Checked {result.tickers_checked} ticker(s); "
        f"wrote {result.signals_written} market anomaly signal(s)."
    )


def cmd_import_expectation_gap_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = expectations.import_expectation_gap_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Expectation gap CSV",
            status="ok",
            reason=f"Imported from {args.csv}.",
        )
    print(f"Imported {count} expectation-gap signal(s).")


def cmd_infer_expectation_gaps(args: argparse.Namespace, settings: Settings) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    with open_db(db_path) as conn:
        signals = expectations.infer_profile_ai_tag_expectation_gaps(
            conn,
            tickers=tickers or None,
            limit=args.limit,
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
    scope = ",".join(tickers) if tickers else "local candidates"
    print(f"Inferred {len(signals)} expectation-gap candidate(s) for {scope}; stored {count}.")


def cmd_import_analyst_estimate_events_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = analysts.import_analyst_estimate_events_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Analyst estimate events CSV",
            status="ok",
            reason=f"Imported from {args.csv}; no analyst estimate data was inferred.",
        )
    print(f"Imported {count} analyst estimate event(s).")


def cmd_fetch_fmp_analyst_events(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    tickers = _parse_ticker_args(args.ticker)
    if not tickers:
        raise SystemExit("--ticker is required.")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.limit > analysts.FMP_ANALYST_ESTIMATES_MAX_LIMIT:
        raise SystemExit(f"--limit must be <= {analysts.FMP_ANALYST_ESTIMATES_MAX_LIMIT}.")
    if args.low_coverage_threshold < 0:
        raise SystemExit("--low-coverage-threshold must be non-negative.")
    with open_db(db_path) as conn:
        if not settings.fmp_api_key:
            sec.mark_source_status(
                conn,
                source_name=analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
                status="unavailable",
                reason="FMP_API_KEY is not configured.",
            )
            print("FMP_API_KEY is not configured; recorded analyst estimates source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings))
        total = 0
        failures: list[str] = []
        for ticker in tickers:
            try:
                events = analysts.fetch_fmp_analyst_estimate_events(
                    client,
                    ticker,
                    settings.fmp_api_key,
                    period=args.period,
                    limit=args.limit,
                    low_coverage_threshold=args.low_coverage_threshold,
                )
            except Exception as exc:
                failures.append(f"{ticker}: {str(exc).replace(settings.fmp_api_key, '***')}")
                continue
            total += analysts.replace_fmp_analyst_estimate_events(conn, ticker, events)
        if failures:
            status = "degraded" if total else "unavailable"
            reason = _short_message("; ".join(failures), 500)
            sec.mark_source_status(
                conn,
                source_name=analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
                status=status,
                reason=reason,
            )
            print(
                f"Stored {total} FMP analyst estimate event(s); "
                f"{len(failures)} ticker(s) failed: {reason}"
            )
            return
        sec.mark_source_status(
            conn,
            source_name=analysts.FMP_ANALYST_ESTIMATES_SOURCE_NAME,
            status="ok",
            reason=(
                f"Stored {total} analyst estimate event(s) for {len(tickers)} ticker(s); "
                f"period={args.period}; limit={args.limit}; no expectation-gap conclusion was inferred."
            ),
        )
    print(
        f"Stored {total} FMP analyst estimate event(s) for {len(tickers)} ticker(s); "
        f"period={args.period}."
    )


def cmd_import_risk_flags_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = risks.import_risk_flags_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Risk flags CSV",
            status="ok",
            reason=f"Imported from {args.csv}.",
        )
    print(f"Imported {count} risk flag(s).")


def cmd_import_macro_csv(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        count = macro.import_macro_indicators_csv(conn, args.csv)
        sec.mark_source_status(
            conn,
            source_name="Manual macro indicator CSV",
            status="ok",
            reason=f"Imported from {args.csv}.",
        )
    print(f"Imported {count} macro/electricity indicator observation(s).")


def cmd_fetch_fred_series(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        if not settings.fred_api_key:
            sec.mark_source_status(
                conn,
                source_name="FRED",
                status="unavailable",
                reason="FRED_API_KEY is not configured.",
            )
            print("FRED_API_KEY is not configured; recorded source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings))
        try:
            observations = macro.fetch_fred_observations(
                client,
                api_key=settings.fred_api_key,
                series_id=args.series_id,
                metric_name=args.metric_name,
                category=args.category,
                limit=args.limit,
                observation_start=args.observation_start,
            )
            count = macro.upsert_macro_indicators(conn, observations)
        except Exception as exc:
            sec.mark_source_status(conn, source_name="FRED", status="degraded", reason=str(exc))
            print(f"FRED unavailable for {args.series_id}: {exc}")
            return
        sec.mark_source_status(
            conn,
            source_name="FRED",
            status="ok",
            reason="Macro observations stored as non-company background evidence.",
        )
    print(f"Stored {count} FRED observation(s) for {args.series_id}.")


def cmd_fetch_fred_public_series(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=_user_agent(settings))
    with open_db(db_path) as conn:
        try:
            observations = macro.fetch_fred_public_observations(
                client,
                series_id=args.series_id,
                metric_name=args.metric_name,
                category=args.category,
                limit=args.limit,
                observation_start=args.observation_start,
            )
            count = macro.upsert_macro_indicators(conn, observations)
        except Exception as exc:
            sec.mark_source_status(conn, source_name="FRED", status="degraded", reason=str(exc))
            print(f"FRED public CSV unavailable for {args.series_id}: {exc}")
            return
        sec.mark_source_status(
            conn,
            source_name="FRED",
            status="ok",
            reason="Public CSV macro observations stored as non-company background evidence.",
        )
    print(f"Stored {count} FRED public CSV observation(s) for {args.series_id}.")


def cmd_fetch_eia_electricity_retail_sales(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        if not settings.eia_api_key:
            sec.mark_source_status(
                conn,
                source_name="EIA Open Data",
                status="unavailable",
                reason="EIA_API_KEY is not configured.",
            )
            print("EIA_API_KEY is not configured; recorded source as unavailable.")
            return
        client = HttpClient(user_agent=_user_agent(settings))
        try:
            observations = macro.fetch_eia_electricity_retail_sales(
                client,
                api_key=settings.eia_api_key,
                limit=args.limit,
                frequency=args.frequency,
                stateid=args.stateid,
                sectorid=args.sectorid,
            )
            count = macro.upsert_macro_indicators(conn, observations)
        except Exception as exc:
            sec.mark_source_status(conn, source_name="EIA Open Data", status="degraded", reason=str(exc))
            print(f"EIA Open Data unavailable: {exc}")
            return
        sec.mark_source_status(
            conn,
            source_name="EIA Open Data",
            status="ok",
            reason="Electricity observations stored as non-company background evidence.",
        )
    print(f"Stored {count} EIA electricity retail sales observation(s).")


def cmd_list_macro_indicators(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    with open_db(db_path) as conn:
        rows = macro.list_latest_macro_indicators(conn, limit=args.limit, category=args.category)
    if not rows:
        print("No macro/electricity indicators found.")
        return
    for row in rows:
        unit = f" {row['unit']}" if row["unit"] else ""
        geo = row["geography"] or "NA"
        print(
            f"{row['period']} {row['source_name']} {row['series_id']} "
            f"{row['metric_name']}={row['value']:g}{unit} geography={geo} url={row['source_url']}"
        )


def cmd_extract_risk_flags(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    ticker = args.ticker.upper() if args.ticker else None
    with open_db(db_path) as conn:
        flags = extract_risk_flags_from_evidence(conn, ticker=ticker, limit=args.limit)
        count = store_extracted_risk_flags(conn, flags)
        sec.mark_source_status(
            conn,
            source_name="Automated risk text extraction",
            status="ok",
            reason=f"Extracted {count} watch-status risk flag candidate(s) from stored evidence.",
        )
    scope = ticker or "all tickers"
    print(f"Extracted {count} risk flag candidate(s) for {scope}.")


def cmd_fetch_news_rss(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=0.2)
    sources: list[news_rss.NewsRssSource] = []
    if args.all_defaults:
        sources.extend(news_rss.DEFAULT_NEWS_RSS_SOURCES)
    if getattr(args, "ticker_defaults", False):
        tickers = _dedupe_tickers(_parse_ticker_args(getattr(args, "ticker", [])))
        if not tickers:
            raise SystemExit("--ticker-defaults requires at least one --ticker.")
        for ticker in tickers:
            sources.extend(news_rss.ticker_news_rss_sources(ticker))
    if not sources:
        sources.append(news_rss.NewsRssSource(args.source_name, args.url))
    total_seen = 0
    total_mapped = 0
    failures = 0
    with open_db(db_path) as conn:
        for source in sources:
            try:
                events = news_rss.fetch_news_rss(
                    client,
                    url=source.url,
                    source_name=source.source_name,
                    limit=args.limit,
                    related_ticker=source.related_ticker,
                )
                result = news_rss.store_news_events(conn, events)
            except Exception as exc:
                failures += 1
                for status_source_name in _news_status_source_names(source):
                    sec.mark_source_status(
                        conn,
                        source_name=status_source_name,
                        status="degraded",
                        reason=_short_message(str(exc), 500),
                    )
                print(f"{source.source_name} unavailable: {exc}")
                continue
            total_seen += result.events_seen
            total_mapped += result.mapped
            for status_source_name in _news_status_source_names(source):
                sec.mark_source_status(
                    conn,
                    source_name=status_source_name,
                    status="ok",
                    reason="RSS announcements stored as title/summary/url evidence only.",
                )
    print(
        f"Stored/updated {total_seen} news RSS event(s) from {len(sources)} source(s); "
        f"mapped {total_mapped} to local tickers; failures={failures}."
    )


def cmd_fetch_gdelt_doc_news(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    client = HttpClient(user_agent=_user_agent(settings), min_interval_seconds=1.0)
    with open_db(db_path) as conn:
        try:
            result = gdelt.fetch_gdelt_doc_news(
                client,
                query=args.query,
                limit=args.limit,
                timespan=args.timespan,
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
            sec.mark_source_status(
                conn,
                source_name=gdelt.GDELT_DOC_API_SOURCE_NAME,
                status="degraded",
                reason=_short_message(str(exc), 500),
            )
            print(f"{gdelt.GDELT_DOC_API_SOURCE_NAME} unavailable: {exc}")
            return
        sec.mark_source_status(
            conn,
            source_name=gdelt.GDELT_DOC_API_SOURCE_NAME,
            status="ok",
            reason=(
                "GDELT DOC API article metadata stored as title/url evidence only; "
                f"query={args.query!r}; source_url={result.source_url}"
            ),
        )
    print(
        f"Stored/updated {stored.events_seen} GDELT DOC article metadata event(s); "
        f"mapped {stored.mapped} to local tickers."
    )


def cmd_import_local_research_md(args: argparse.Namespace, settings: Settings) -> None:
    db_path = _db_path(args, settings)
    path = args.path
    if not path.exists():
        raise SystemExit(f"Local research Markdown file not found: {path}")
    with open_db(db_path) as conn:
        result = local_research.import_local_research_markdown(
            conn,
            path,
            source_name=args.source_name,
            limit=args.limit,
        )
        status = "ok" if result.leads_seen else "degraded"
        reason = (
            f"Imported {result.leads_seen} external research lead(s) from local Markdown; "
            f"mapped_ticker_leads={result.mapped_ticker_leads}; "
            f"distinct_tickers={result.distinct_tickers}. Leads require primary-source verification."
        )
        sec.mark_source_status(
            conn,
            source_name=args.source_name,
            status=status,
            reason=reason,
        )
    print(
        f"Imported {result.leads_seen} local research lead(s); "
        f"mapped_ticker_leads={result.mapped_ticker_leads}; "
        f"distinct_tickers={result.distinct_tickers}."
    )


def _ensure_cik(conn, client: HttpClient, ticker: str) -> str:
    cik = sec.get_cik(conn, ticker)
    if cik:
        return cik
    sec.sync_company_tickers(conn, client)
    cik = sec.get_cik(conn, ticker)
    if not cik:
        raise SystemExit(f"No CIK found for {ticker}. Confirm ticker is listed in SEC company_tickers.")
    return cik


def _parse_forms(raw: str) -> set[str]:
    if raw.strip().lower() in {"", "all", "*"}:
        return set()
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def _select_fmp_quote_failure_tickers(conn, *, tickers: list[str], limit: int) -> list[str]:
    explicit = _dedupe_tickers(tickers)
    if explicit:
        return explicit[:limit]
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM source_failures
        WHERE status = 'open'
          AND source_name = ?
          AND endpoint = 'quote'
          AND ticker IS NOT NULL
          AND ticker != ''
        ORDER BY last_seen_at DESC, ticker
        LIMIT ?
        """,
        (valuation.FMP_QUOTE_SOURCE_NAME, limit),
    ).fetchall()
    return [str(row["ticker"]).upper() for row in rows]


def _parse_ticker_args(values: list[str]) -> list[str]:
    return _parse_csv_args(values)


def _filter_source_input_templates_for_args(templates, names: list[str]):
    try:
        return filter_source_input_templates(templates, names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_csv_args(values: list[str]) -> list[str]:
    tickers: list[str] = []
    for value in values:
        tickers.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return tickers


def _dedupe_tickers(values: list[str]) -> list[str]:
    seen: set[str] = set()
    tickers: list[str] = []
    for value in values:
        ticker = value.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _news_status_source_names(source: news_rss.NewsRssSource) -> list[str]:
    names = [source.source_name]
    if source.related_ticker:
        if source.source_name.startswith("Yahoo Finance RSS"):
            names.append("Yahoo Finance ticker RSS")
        elif source.source_name.startswith("Nasdaq RSS"):
            names.append("Nasdaq ticker RSS")
    return names


def _short_message(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _fmp_market_bars_source_url(ticker: str, *, benchmark: str, lookback_days: int) -> str:
    to_date = date.fromisoformat(utc_now_iso()[:10])
    from_date = to_date - timedelta(days=lookback_days)
    ticker_url = market.FMP_HISTORICAL_PRICE_EOD_URL.format(
        ticker=ticker.upper(),
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        api_key="***",
    )
    benchmark_url = market.FMP_HISTORICAL_PRICE_EOD_URL.format(
        ticker=benchmark.upper(),
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        api_key="***",
    )
    return f"{ticker_url}; benchmark={benchmark.upper()}:{benchmark_url}"


def _yahoo_market_bars_source_url(ticker: str, *, benchmark: str, lookback_days: int) -> str:
    to_date = date.fromisoformat(utc_now_iso()[:10])
    from_date = to_date - timedelta(days=lookback_days)
    ticker_url = market.yahoo_chart_url(ticker, from_date, to_date)
    benchmark_url = market.yahoo_chart_url(benchmark, from_date, to_date)
    return f"{ticker_url}; benchmark={benchmark.upper()}:{benchmark_url}"


def _sql_placeholders(values: set[str]) -> str:
    if not values:
        return "NULL"
    return ",".join("?" for _ in values)


def _upsert_queue_filing(conn, row, document_url: str) -> None:
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


def _db_path(args: argparse.Namespace, settings: Settings) -> Path:
    return args.db if args.db else settings.db_path


def _user_agent(settings: Settings) -> str:
    return settings.sec_user_agent or "ai-stock-discovery-mvp/0.1 local-research"


def _require_sec_user_agent(settings: Settings) -> None:
    if not settings.sec_user_agent:
        raise SystemExit(
            "SEC_USER_AGENT is required for SEC commands. Set it in .env, for example: "
            "SEC_USER_AGENT=\"AI Stock Discovery MVP your.email@example.com\""
        )


if __name__ == "__main__":
    main()
