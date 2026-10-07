from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Mapping


@dataclass(frozen=True)
class MvpReadinessRow:
    area: str
    requirement: str
    status: str
    evidence: str
    next_action: str
    notes: str


REQUIRED_TABLES: tuple[str, ...] = (
    "universe",
    "company_profile",
    "filings",
    "financial_facts",
    "ai_relevance_signals",
    "ai_industry_tags",
    "catalysts",
    "valuation_snapshots",
    "expectation_gap_signals",
    "risk_flags",
    "market_confirmation_signals",
    "evidence_items",
    "research_cards",
    "score_snapshots",
    "data_source_status",
    "source_failures",
    "pipeline_runs",
    "pipeline_steps",
)

TABLE_REQUIREMENTS: tuple[tuple[str, str, str, str], ...] = (
    ("table_universe", "Clean US equity universe has local rows.", "universe", "fetch_universe_or_import_universe"),
    ("table_company_profile", "Company profile, CIK, sector, and IR context has local rows.", "company_profile", "sync_sec_tickers_or_import_company_profiles"),
    ("table_filings", "SEC filing metadata has local rows.", "filings", "sync_filings_or_fetch_sec_rss"),
    ("table_financial_facts", "Source-backed financial facts have local rows.", "financial_facts", "fetch_companyfacts_or_fmp_financials_for_priority_tickers"),
    ("table_ai_signals", "AI keyword/context evidence has local rows.", "ai_relevance_signals", "scan_sec_filings_or_import_transcripts"),
    ("table_ai_chain_tags", "AI industry-chain tags have source-backed local rows.", "ai_industry_tags", "tag_ai_chain_or_import_ai_tags"),
    ("table_valuation", "Valuation snapshots have local rows.", "valuation_snapshots", "import_valuation_csv_or_fetch_quote_source"),
    ("table_catalysts", "Catalyst candidates or calendar events have local rows.", "catalysts", "extract_catalysts_or_import_catalyst_calendar"),
    ("table_expectation_gap", "Expectation-gap evidence has local rows.", "expectation_gap_signals", "import_expectation_gap_or_analyst_events"),
    ("table_risks", "Risk flags have local rows.", "risk_flags", "extract_risk_flags_or_import_risk_flags"),
    ("table_market_confirmation", "Market confirmation signals have local rows.", "market_confirmation_signals", "import_market_confirmation_or_detect_anomalies"),
    ("table_evidence_items", "Evidence inventory has local rows.", "evidence_items", "refresh_source_backed_evidence"),
    ("table_score_snapshots", "Score snapshots exist for change tracking.", "score_snapshots", "snapshot_scores_or_run_pipeline"),
)

SOURCE_GROUPS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "source_stock_universe",
        "P0 stock universe source is recorded.",
        ("Nasdaq Trader Symbol Directory",),
        "run_fetch_universe_or_run_pipeline_without_skip_network",
    ),
    (
        "source_sec_mapping",
        "SEC ticker/CIK mapping source is recorded.",
        ("SEC company_tickers",),
        "run_sync_sec_tickers_with_sec_user_agent",
    ),
    (
        "source_financial_facts",
        "Financial facts source is recorded.",
        ("SEC companyfacts", "Financial Modeling Prep financial statements API"),
        "run_fetch_companyfacts_or_fmp_financials_for_priority_tickers",
    ),
    (
        "source_sec_filing_triggers",
        "SEC filing trigger source is recorded.",
        ("SEC latest filings Atom feed", "SEC submissions"),
        "run_fetch_sec_rss_or_sync_filings",
    ),
    (
        "source_sec_filing_documents",
        "SEC filing document processing source is recorded.",
        ("SEC filing documents",),
        "process_filing_queue_for_priority_tickers",
    ),
    (
        "source_ir_and_news",
        "IR/news announcement sources are recorded.",
        (
            "Company IR page",
            "GlobeNewswire Press Releases",
            "PR Newswire News Releases",
            "Yahoo Finance ticker RSS",
            "Nasdaq ticker RSS",
            "GDELT DOC API",
        ),
        "check_ir_pages_and_fetch_news_rss",
    ),
    (
        "source_ai_chain_tags",
        "AI industry-chain tag source or inference is recorded.",
        ("Manual AI industry tags CSV", "Automated AI industry tag inference"),
        "infer_ai_chain_tags_or_import_manual_source_backed_tags",
    ),
    (
        "source_valuation",
        "Valuation/profile source is recorded.",
        (
            "Manual valuation CSV",
            "Eastmoney US quote API",
            "Yahoo Finance endpoint prototype",
            "Financial Modeling Prep quote API",
            "Financial Modeling Prep profile API",
        ),
        "import_valuation_csv_or_configure_optional_market_data_source",
    ),
    (
        "source_market_confirmation",
        "Market confirmation source is recorded.",
        (
            "Financial Modeling Prep historical price API",
            "Yahoo Finance chart endpoint prototype",
            "Local market confirmation CSV",
            "Local market price bars CSV",
            "Local market anomaly detector",
        ),
        "fetch_fmp_market_bars_or_import_local_market_bars",
    ),
    (
        "source_short_sale",
        "FINRA short-sale volume source is recorded.",
        ("FINRA Daily Short Sale Volume",),
        "import_finra_short_sale_volume_file",
    ),
    (
        "source_macro",
        "Macro/electricity background source is recorded.",
        ("FRED", "EIA Open Data", "Manual macro indicator CSV"),
        "configure_optional_api_keys_or_import_macro_csv",
    ),
    (
        "source_expectation_gap",
        "Expectation-gap and thesis-context sources are recorded.",
        (
            "Financial Modeling Prep analyst estimates API",
            "Expectation gap CSV",
            "Analyst estimate events CSV",
            "Manual peer valuation CSV",
            "Manual guidance events CSV",
            "Manual transcript snippets CSV",
            "Local profile/AI tag expectation gap inference",
        ),
        "import_source_backed_expectation_gap_guidance_or_transcript_inputs",
    ),
    (
        "source_risk",
        "Risk source or automated risk extraction is recorded.",
        ("Risk flags CSV", "Automated risk text extraction", "SEC filing documents"),
        "extract_risk_flags_or_import_source_backed_risk_flags",
    ),
)

REPORT_REQUIREMENTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("report_watchlist", "Watchlist CSV exists and has rows.", "watchlist.csv", "csv", "build_watchlist"),
    ("report_evidence_audit", "Evidence audit CSV exists and has rows.", "evidence_audit.csv", "csv", "build_evidence_audit"),
    ("report_thesis_checklist", "Thesis source-chain checklist exists and has rows.", "thesis_checklist.csv", "csv", "build_thesis_checklist"),
    ("report_review_queue", "Human review queue exists and has rows.", "review_queue.csv", "csv", "build_review_queue"),
    ("report_research_pool", "Score-layered research pool exists and has rows.", "research_pool.csv", "csv", "build_research_pool"),
    ("report_ai_chain_coverage", "AI chain coverage report exists and has rows.", "ai_chain_coverage.csv", "csv", "build_ai_chain_coverage"),
    ("report_score_provenance", "Score provenance report exists and has rows.", "score_provenance.csv", "csv", "build_score_provenance"),
    ("report_data_mining_leads", "Actionable data-mining lead queue exists and has rows.", "data_mining_leads.csv", "csv", "build_data_mining_leads"),
    ("report_source_failures", "Ticker-level source failure report exists.", "source_failures.csv", "csv", "build_source_failure_report"),
    ("report_research_cards", "Markdown research cards directory exists and has cards.", "cards", "directory", "build_research_cards"),
    ("report_card_artifact_audit", "Research card artifact audit CSV exists and has rows.", "card_artifact_audit.csv", "csv", "build_card_artifact_audit"),
    ("report_card_index", "Research card index CSV exists and has rows.", "card_index.csv", "csv", "build_research_cards"),
    ("report_refresh_plan", "Refresh plan CSV exists and has rows.", "refresh_plan.csv", "csv", "build_refresh_plan"),
    ("report_source_input_audit", "Source input template audit CSV exists and has rows.", "source_input_audit.csv", "csv", "build_source_input_audit"),
    ("report_source_input_import", "Source input import dry-run report exists and has rows.", "source_input_import.csv", "csv", "build_source_input_import_plan"),
    ("report_source_input_plan", "Source input action plan CSV exists and has rows.", "source_input_plan.csv", "csv", "build_source_input_action_plan"),
    ("report_snapshot_changes", "Snapshot change report exists and has rows.", "snapshot_changes.csv", "csv", "build_snapshot_change_report"),
    ("report_ops_report", "Operations report exists.", "ops_report.md", "file", "build_ops_report"),
)

EMPTY_OK_REPORTS = {
    "report_source_input_audit",
    "report_source_input_import",
    "report_source_input_plan",
    "report_source_failures",
}


def build_mvp_readiness(
    conn: sqlite3.Connection,
    *,
    reports_dir: Path = Path("reports"),
    design_doc: Path = Path("ai_potential_stock_discovery_system.md"),
    report_paths: Mapping[str, Path] | None = None,
    cards_dir: Path | None = None,
) -> list[MvpReadinessRow]:
    source_rows = _source_rows(conn)
    rows: list[MvpReadinessRow] = [
        MvpReadinessRow(
            area="research_boundary",
            requirement="Readiness report is only a local research-operations checklist.",
            status="ready",
            evidence="No external data is fetched and no company-level conclusion is generated.",
            next_action="verify_original_sources_before_company_level_conclusions",
            notes="Research auxiliary only; not investment advice.",
        ),
        _design_doc_row(design_doc),
        _schema_row(conn),
        _latest_pipeline_row(conn),
    ]
    rows.extend(_source_group_row(source_rows, *requirement) for requirement in SOURCE_GROUPS)
    rows.extend(_table_requirement_row(conn, *requirement) for requirement in TABLE_REQUIREMENTS)
    rows.extend(
        _report_requirement_row(
            reports_dir=reports_dir,
            report_paths=report_paths or {},
            cards_dir=cards_dir,
            area=area,
            requirement=requirement,
            relative_path=relative_path,
            kind=kind,
            build_action=build_action,
        )
        for area, requirement, relative_path, kind, build_action in REPORT_REQUIREMENTS
    )
    return rows


def write_mvp_readiness_csv(rows: list[MvpReadinessRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MvpReadinessRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _design_doc_row(design_doc: Path) -> MvpReadinessRow:
    if not design_doc.exists():
        return MvpReadinessRow(
            area="design_document",
            requirement="Markdown design document is present and readable.",
            status="blocked",
            evidence=f"path={_display_path(design_doc)}; exists=false",
            next_action="restore_or_provide_ai_potential_stock_discovery_system_markdown",
            notes="Do not rebuild requirements from memory without the source document.",
        )
    try:
        text = design_doc.read_text(encoding="utf-8")[:8000]
    except (OSError, UnicodeDecodeError) as exc:
        return MvpReadinessRow(
            area="design_document",
            requirement="Markdown design document is present and readable.",
            status="blocked",
            evidence=f"path={_display_path(design_doc)}; read_error={type(exc).__name__}",
            next_action="fix_design_document_permissions_or_path",
            notes=str(exc),
        )
    matches_goal = (
        "AI" in text
        and ("\u6f5c\u529b\u80a1" in text or "stock discovery" in text.lower())
        and ("\u53d1\u6398\u7cfb\u7edf" in text or "MVP" in text)
    )
    return MvpReadinessRow(
        area="design_document",
        requirement="Markdown design document is present and readable.",
        status="ready" if matches_goal else "review",
        evidence=f"path={_display_path(design_doc)}; bytes={design_doc.stat().st_size}",
        next_action="keep_design_doc_as_priority_source_for_architecture" if matches_goal else "confirm_design_doc_matches_ai_stock_discovery_goal",
        notes="Document signature matched expected AI stock discovery MVP wording." if matches_goal else "Document is readable but should be manually confirmed.",
    )


def _schema_row(conn: sqlite3.Connection) -> MvpReadinessRow:
    existing = {
        str(row["name"])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    missing = [table for table in REQUIRED_TABLES if table not in existing]
    return MvpReadinessRow(
        area="schema",
        requirement="Required MVP tables exist in SQLite.",
        status="ready" if not missing else "blocked",
        evidence=f"required_tables={len(REQUIRED_TABLES)}; missing={';'.join(missing) if missing else 'none'}",
        next_action="continue_pipeline_runs" if not missing else "run_init_db_and_migrations",
        notes="Schema check only verifies table presence, not data quality.",
    )


def _latest_pipeline_row(conn: sqlite3.Connection) -> MvpReadinessRow:
    row = conn.execute(
        """
        SELECT id, status, started_at, completed_at
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return MvpReadinessRow(
            area="latest_pipeline_run",
            requirement="At least one repeatable MVP pipeline run has been recorded.",
            status="needs_data",
            evidence="pipeline_runs=0",
            next_action="run_pipeline",
            notes="No pipeline health can be inferred yet.",
        )
    counts = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM pipeline_steps
        WHERE run_id = ?
        GROUP BY status
        ORDER BY status
        """,
        (row["id"],),
    ).fetchall()
    step_counts = ";".join(f"{count_row['status']}={count_row['count']}" for count_row in counts) or "none"
    status = str(row["status"])
    readiness_status = {
        "success": "ready",
        "degraded": "degraded",
        "failed": "blocked",
        "running": "review",
    }.get(status, "review")
    return MvpReadinessRow(
        area="latest_pipeline_run",
        requirement="Latest repeatable MVP pipeline run is recorded.",
        status=readiness_status,
        evidence=(
            f"run_id={row['id']}; status={status}; completed_at={row['completed_at'] or 'NA'}; "
            f"step_counts={step_counts}"
        ),
        next_action="inspect_pipeline_errors" if readiness_status in {"blocked", "degraded"} else "continue_scheduled_refresh",
        notes="Skipped network steps may be acceptable for offline validation, but source gaps remain visible below.",
    )


def _source_group_row(
    source_rows: Mapping[str, Mapping[str, object]],
    area: str,
    requirement: str,
    source_names: tuple[str, ...],
    next_action: str,
) -> MvpReadinessRow:
    statuses: list[str] = []
    reasons: list[str] = []
    status_values: list[str] = []
    for source_name in source_names:
        row = source_rows.get(source_name)
        if row is None:
            statuses.append(f"{source_name}=missing")
            continue
        status = str(row["status"])
        status_values.append(status)
        statuses.append(f"{source_name}={status}")
        if row["reason"]:
            reasons.append(f"{source_name}: {_shorten(str(row['reason']), 120)}")

    if "ok" in status_values:
        readiness_status = "ready"
        recommended_action = "keep_refreshing_on_schedule"
    elif "degraded" in status_values:
        readiness_status = "degraded"
        recommended_action = next_action
    elif "unavailable" in status_values:
        readiness_status = "unavailable"
        recommended_action = next_action
    else:
        readiness_status = "needs_data"
        recommended_action = next_action

    return MvpReadinessRow(
        area=area,
        requirement=requirement,
        status=readiness_status,
        evidence="; ".join(statuses),
        next_action=recommended_action,
        notes=" | ".join(reasons) if reasons else "No source-status reason is recorded.",
    )


def _table_requirement_row(
    conn: sqlite3.Connection,
    area: str,
    requirement: str,
    table_name: str,
    next_action: str,
) -> MvpReadinessRow:
    if area == "table_expectation_gap":
        return _expectation_gap_table_row(conn, requirement, next_action)
    try:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    except sqlite3.OperationalError as exc:
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="blocked",
            evidence=f"table={table_name}; error={type(exc).__name__}",
            next_action="run_init_db_and_migrations",
            notes=str(exc),
        )
    count = int(row["count"] or 0)
    return MvpReadinessRow(
        area=area,
        requirement=requirement,
        status="ready" if count > 0 else "needs_data",
        evidence=f"table={table_name}; rows={count}",
        next_action="continue_refresh_and_review" if count > 0 else next_action,
        notes="Local row count only; verify source links before using company-level conclusions.",
    )


def _expectation_gap_table_row(
    conn: sqlite3.Connection,
    requirement: str,
    next_action: str,
) -> MvpReadinessRow:
    try:
        expectation_gap_count = _table_count(conn, "expectation_gap_signals")
        analyst_event_count = _table_count(conn, "analyst_estimate_events")
    except sqlite3.OperationalError as exc:
        return MvpReadinessRow(
            area="table_expectation_gap",
            requirement=requirement,
            status="blocked",
            evidence=f"table=expectation_gap_signals|analyst_estimate_events; error={type(exc).__name__}",
            next_action="run_init_db_and_migrations",
            notes=str(exc),
        )
    count = expectation_gap_count + analyst_event_count
    return MvpReadinessRow(
        area="table_expectation_gap",
        requirement=requirement,
        status="ready" if count > 0 else "needs_data",
        evidence=(
            f"expectation_gap_signals={expectation_gap_count}; "
            f"analyst_estimate_events={analyst_event_count}; rows={count}"
        ),
        next_action="continue_refresh_and_review" if count > 0 else next_action,
        notes="Analyst events are context evidence only; verify original source links before company-level conclusions.",
    )


def _table_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"] or 0)


def _report_requirement_row(
    *,
    reports_dir: Path,
    report_paths: Mapping[str, Path],
    cards_dir: Path | None,
    area: str,
    requirement: str,
    relative_path: str,
    kind: str,
    build_action: str,
) -> MvpReadinessRow:
    if area == "report_research_cards" and cards_dir is not None:
        path = cards_dir
    else:
        path = report_paths.get(area, reports_dir / relative_path)

    if kind == "directory":
        if not path.exists():
            return _missing_report_row(area, requirement, path, build_action)
        if area == "report_research_cards":
            return _research_cards_directory_row(
                area=area,
                requirement=requirement,
                path=path,
                index_path=report_paths.get("report_card_index", reports_dir / "card_index.csv"),
                build_action=build_action,
            )
        count = len(list(path.glob("*.md")))
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="ready" if count > 0 else "needs_data",
            evidence=f"path={_display_path(path)}; markdown_files={count}",
            next_action="review_generated_cards" if count > 0 else build_action,
            notes="Cards are local research aids and must not be treated as investment advice.",
        )

    if not path.exists():
        return _missing_report_row(area, requirement, path, build_action)

    if kind == "file":
        size = path.stat().st_size
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="ready" if size > 0 else "needs_data",
            evidence=f"path={_display_path(path)}; bytes={size}",
            next_action="review_report" if size > 0 else build_action,
            notes="Generated report presence does not prove data completeness.",
        )

    try:
        row_count = _csv_row_count(path)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="degraded",
            evidence=f"path={_display_path(path)}; read_error={type(exc).__name__}",
            next_action=build_action,
            notes=str(exc),
        )
    if row_count == 0 and area in EMPTY_OK_REPORTS:
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="ready",
            evidence=f"path={_display_path(path)}; rows=0",
            next_action="review_report",
            notes="No source-input gap rows were generated; this is expected when upstream readiness gaps are closed.",
        )
    return MvpReadinessRow(
        area=area,
        requirement=requirement,
        status="ready" if row_count > 0 else "needs_data",
        evidence=f"path={_display_path(path)}; rows={row_count}",
        next_action="review_report" if row_count > 0 else build_action,
        notes="Generated report rows are local checklist artifacts, not company conclusions.",
        )


def _research_cards_directory_row(
    *,
    area: str,
    requirement: str,
    path: Path,
    index_path: Path,
    build_action: str,
) -> MvpReadinessRow:
    card_files = sorted(path.glob("*.md"))
    card_names = {card_file.name.upper() for card_file in card_files}
    count = len(card_files)
    if count == 0:
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="needs_data",
            evidence=f"path={_display_path(path)}; markdown_files=0",
            next_action=build_action,
            notes="Generate current research cards before relying on the card directory.",
        )

    if not index_path.exists():
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="ready",
            evidence=(
                f"path={_display_path(path)}; markdown_files={count}; "
                f"card_index={_display_path(index_path)}; index_exists=false"
            ),
            next_action="review_generated_cards",
            notes=(
                "Cards exist, but stale-card auditing requires card_index.csv; "
                "the separate report_card_index row tracks that artifact."
            ),
        )

    try:
        indexed_card_names = _indexed_card_names(index_path)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return MvpReadinessRow(
            area=area,
            requirement=requirement,
            status="degraded",
            evidence=(
                f"path={_display_path(path)}; markdown_files={count}; "
                f"card_index={_display_path(index_path)}; read_error={type(exc).__name__}"
            ),
            next_action="rebuild_card_index",
            notes=str(exc),
        )

    stale_files = sorted(card_names - indexed_card_names)
    missing_files = sorted(indexed_card_names - card_names)
    status = "degraded" if stale_files or missing_files else "ready"
    next_action = (
        "regenerate_cards_and_review_stale_generated_cards"
        if status == "degraded"
        else "review_generated_cards"
    )
    notes = "Cards match the current card index; still verify original sources before company-level use."
    if stale_files or missing_files:
        parts: list[str] = []
        if stale_files:
            parts.append(f"stale_not_in_index={_sample_names(stale_files)}")
        if missing_files:
            parts.append(f"indexed_missing_file={_sample_names(missing_files)}")
        notes = (
            "; ".join(parts)
            + ". No files were deleted; stale generated cards should not be used without regeneration."
        )

    return MvpReadinessRow(
        area=area,
        requirement=requirement,
        status=status,
        evidence=(
            f"path={_display_path(path)}; markdown_files={count}; "
            f"indexed_cards={len(indexed_card_names)}; "
            f"stale_card_files={len(stale_files)}; "
            f"missing_indexed_files={len(missing_files)}"
        ),
        next_action=next_action,
        notes=notes,
    )


def _missing_report_row(
    area: str,
    requirement: str,
    path: Path,
    build_action: str,
) -> MvpReadinessRow:
    return MvpReadinessRow(
        area=area,
        requirement=requirement,
        status="needs_data",
        evidence=f"path={_display_path(path)}; exists=false",
        next_action=build_action,
        notes="Generate this artifact before relying on the full MVP operating loop.",
    )


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _indexed_card_names(index_path: Path) -> set[str]:
    names: set[str] = set()
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            card_path = str(row.get("card_path") or "").strip()
            ticker = str(row.get("ticker") or "").strip().upper()
            if card_path:
                names.add(Path(card_path).name.upper())
            elif ticker:
                names.add(f"{ticker}.MD")
    return names


def _source_rows(conn: sqlite3.Connection) -> dict[str, Mapping[str, object]]:
    rows = conn.execute(
        """
        SELECT source_name, status, reason, last_checked_at
        FROM data_source_status
        """
    ).fetchall()
    source_rows: dict[str, Mapping[str, object]] = {
        str(row["source_name"]): {
            "source_name": row["source_name"],
            "status": row["status"],
            "reason": row["reason"],
            "last_checked_at": row["last_checked_at"],
        }
        for row in rows
    }
    _add_local_source_evidence(conn, source_rows)
    return source_rows


def _add_local_source_evidence(
    conn: sqlite3.Connection,
    source_rows: dict[str, Mapping[str, object]],
) -> None:
    _add_or_preserve_local_source(
        source_rows,
        source_name="SEC company_tickers",
        row_count=_scalar_count(
            conn,
            """
            SELECT COUNT(*) AS count
            FROM company_profile
            WHERE source = 'SEC company_tickers'
              AND COALESCE(ticker, '') != ''
              AND COALESCE(cik, '') != ''
            """,
        ),
        reason_prefix="Local source-backed SEC company_tickers",
        last_checked_at=_scalar_value(
            conn,
            """
            SELECT MAX(updated_at) AS value
            FROM company_profile
            WHERE source = 'SEC company_tickers'
              AND COALESCE(ticker, '') != ''
              AND COALESCE(cik, '') != ''
            """,
        ),
    )
    _add_or_preserve_local_source(
        source_rows,
        source_name="SEC companyfacts",
        row_count=_scalar_count(
            conn,
            """
            SELECT COUNT(*) AS count
            FROM financial_facts
            WHERE source LIKE 'https://data.sec.gov/api/xbrl/companyfacts/%'
            """,
        ),
        reason_prefix="Local source-backed SEC companyfacts financial_facts",
        last_checked_at=_scalar_value(
            conn,
            """
            SELECT MAX(updated_at) AS value
            FROM financial_facts
            WHERE source LIKE 'https://data.sec.gov/api/xbrl/companyfacts/%'
            """,
        ),
    )
    _add_or_preserve_local_source(
        source_rows,
        source_name="Financial Modeling Prep financial statements API",
        row_count=_scalar_count(
            conn,
            """
            SELECT COUNT(*) AS count
            FROM financial_facts
            WHERE source LIKE '%financialmodelingprep.com/stable/income-statement%'
            """,
        ),
        reason_prefix="Local source-backed FMP financial_facts",
        last_checked_at=_scalar_value(
            conn,
            """
            SELECT MAX(updated_at) AS value
            FROM financial_facts
            WHERE source LIKE '%financialmodelingprep.com/stable/income-statement%'
            """,
        ),
    )
    _add_or_preserve_local_source(
        source_rows,
        source_name="SEC latest filings Atom feed",
        row_count=_scalar_count(
            conn,
            """
            SELECT COUNT(*) AS count
            FROM rss_filing_events
            WHERE COALESCE(filing_url, '') != ''
            """,
        ),
        reason_prefix="Local source-backed SEC latest filings Atom feed",
        last_checked_at=_scalar_value(
            conn,
            """
            SELECT MAX(fetched_at) AS value
            FROM rss_filing_events
            WHERE COALESCE(filing_url, '') != ''
            """,
        ),
    )


def _add_or_preserve_local_source(
    source_rows: dict[str, Mapping[str, object]],
    *,
    source_name: str,
    row_count: int,
    reason_prefix: str,
    last_checked_at: object,
) -> None:
    if row_count <= 0:
        return
    existing = source_rows.get(source_name)
    existing_status = str(existing["status"]) if existing else "missing"
    existing_reason = str(existing["reason"] or "") if existing else ""
    if existing_status == "ok":
        return
    reason = f"{reason_prefix} rows={row_count}."
    if existing_status != "missing":
        reason += f" Latest network/source status was {existing_status}"
        if existing_reason:
            reason += f": {_shorten(existing_reason, 120)}"
        reason += "."
    source_rows[source_name] = {
        "source_name": source_name,
        "status": "ok",
        "reason": reason,
        "last_checked_at": last_checked_at,
    }


def _scalar_count(conn: sqlite3.Connection, query: str) -> int:
    row = conn.execute(query).fetchone()
    return int(row["count"] or 0)


def _scalar_value(conn: sqlite3.Connection, query: str) -> object:
    row = conn.execute(query).fetchone()
    return row["value"] if row else None


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _sample_names(names: list[str], limit: int = 5) -> str:
    sample = names[:limit]
    suffix = f";+{len(names) - limit}" if len(names) > limit else ""
    return ",".join(sample) + suffix
