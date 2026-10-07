from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3

from ai_stock_discovery.review_queue import ReviewQueueRow, build_review_queue
from ai_stock_discovery.source_failures import SourceFailureRow, build_source_failure_report
from ai_stock_discovery.sources.local_research import (
    LOCAL_RESEARCH_RELATED_MODULE,
    LOCAL_RESEARCH_SOURCE_NAME,
)


@dataclass(frozen=True)
class RefreshPlanRow:
    ticker: str
    company_name: str | None
    review_bucket: str
    review_priority: str
    score_total: float
    tracking_frequency: str
    action_type: str
    source_name: str
    due_status: str
    last_observed_at: str | None
    days_since_observed: int | None
    due_after_days: int
    recommended_command: str
    source_status: str
    reason: str
    plan_notes: str


def build_refresh_plan(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 200,
    min_score: float | None = None,
    as_of: str | datetime | date | None = None,
    include_scheduled: bool = False,
) -> list[RefreshPlanRow]:
    as_of_date = _as_of_date(as_of)
    queue_rows = build_review_queue(conn, tickers=tickers, limit=limit, min_score=min_score)
    source_failure_replacements = _source_failure_replacements(conn)
    planned: dict[tuple[str, str], RefreshPlanRow] = {}
    for queue_row in queue_rows:
        context = _company_context(conn, queue_row.ticker)
        _add_missing_source_actions(conn, planned, queue_row, context, as_of_date, source_failure_replacements)
        _add_external_research_verification_action(conn, planned, queue_row, context, as_of_date)
        _add_ir_monitor_action(conn, planned, queue_row, context, as_of_date, include_scheduled)
        financial_replacement = _best_replacement(
            source_failure_replacements,
            queue_row.ticker,
            actions=("configure_sec_user_agent_then_backfill", "backfill_sec_companyfacts"),
        )
        _add_recency_action(
            conn,
            planned,
            queue_row,
            context,
            as_of_date,
            include_scheduled,
            action_type="refresh_financial_facts",
            source_name="SEC companyfacts",
            last_observed_at=_latest_financial_at(conn, queue_row.ticker),
            due_after_days=max(30, queue_row.next_review_due_days),
            recommended_command=(
                financial_replacement.replacement_command
                if financial_replacement
                else f"python -m ai_stock_discovery.cli fetch-companyfacts --ticker {queue_row.ticker}"
            ),
            reason=_reason_with_replacement(
                "Refresh SEC XBRL financial facts when core financial evidence is missing or stale.",
                financial_replacement,
            ),
        )
        valuation_replacement = _best_replacement(
            source_failure_replacements,
            queue_row.ticker,
            actions=("fetch_quote_fallback_or_import_manual_snapshot",),
        )
        _add_recency_action(
            conn,
            planned,
            queue_row,
            context,
            as_of_date,
            include_scheduled,
            action_type="refresh_valuation_context",
            source_name="Financial Modeling Prep quote API",
            last_observed_at=_latest_valuation_at(conn, queue_row.ticker),
            due_after_days=queue_row.next_review_due_days,
            recommended_command=(
                valuation_replacement.replacement_command
                if valuation_replacement
                else (
                    "python -m ai_stock_discovery.cli refresh-ai-seed-data "
                    f"--ticker {queue_row.ticker} --skip-sec-facts --skip-fmp-financials --skip-analyst --skip-finra"
                )
            ),
            reason=_reason_with_replacement(
                (
                    "Refresh valuation context from FMP quote/profile; profile market cap is a source-backed "
                    "fallback when quote is unavailable."
                ),
                valuation_replacement,
            ),
        )
        market_replacement = _best_replacement(
            source_failure_replacements,
            queue_row.ticker,
            actions=("fetch_market_bars_from_fallback_or_local_source",),
        )
        _add_recency_action(
            conn,
            planned,
            queue_row,
            context,
            as_of_date,
            include_scheduled,
            action_type="refresh_market_confirmation",
            source_name="Yahoo Finance chart endpoint prototype",
            last_observed_at=_latest_market_bar_at(conn, queue_row.ticker),
            due_after_days=max(7, queue_row.next_review_due_days),
            recommended_command=(
                market_replacement.replacement_command
                if market_replacement
                else (
                    "python -m ai_stock_discovery.cli fetch-yahoo-market-bars "
                    f"--ticker {queue_row.ticker} --benchmark QQQ --lookback-days 90"
                )
            ),
            reason=_reason_with_replacement(
                (
                    "Refresh daily price/volume bars for market-confirmation context. "
                    "Yahoo is a prototype fallback while FMP historical EOD entitlement gaps remain open."
                ),
                market_replacement,
            ),
        )
        if queue_row.review_bucket in {"priority_research", "watch_observe", "light_monitor"}:
            _add_recency_action(
                conn,
                planned,
                queue_row,
                context,
                as_of_date,
                include_scheduled,
                action_type="refresh_news_and_catalysts",
                source_name="News and catalyst monitor",
                last_observed_at=_latest_news_or_catalyst_at(conn, queue_row.ticker),
                due_after_days=queue_row.next_review_due_days,
                recommended_command="python -m ai_stock_discovery.cli fetch-news-rss --limit 25",
                reason=(
                    "Refresh announcement monitoring for active queue items; run extract-catalysts after new "
                    "source-backed news or filings are stored."
                ),
            )
    rows = sorted(
        planned.values(),
        key=lambda row: (
            _due_status_rank(row.due_status),
            _bucket_rank(row.review_bucket),
            -row.score_total,
            row.ticker,
            row.action_type,
        ),
    )
    return rows[:limit]


def write_refresh_plan_csv(rows: list[RefreshPlanRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RefreshPlanRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _add_missing_source_actions(
    conn: sqlite3.Connection,
    planned: dict[tuple[str, str], RefreshPlanRow],
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
    source_failure_replacements: dict[str, list[SourceFailureRow]],
) -> None:
    checks = {check for check in queue_row.required_review_checks.split(";") if check}
    financial_replacement = _best_replacement(
        source_failure_replacements,
        queue_row.ticker,
        actions=("configure_sec_user_agent_then_backfill", "backfill_sec_companyfacts"),
    )
    valuation_replacement = _best_replacement(
        source_failure_replacements,
        queue_row.ticker,
        actions=("fetch_quote_fallback_or_import_manual_snapshot",),
    )
    market_replacement = _best_replacement(
        source_failure_replacements,
        queue_row.ticker,
        actions=("fetch_market_bars_from_fallback_or_local_source",),
    )
    expectation_replacement = _best_replacement(
        source_failure_replacements,
        queue_row.ticker,
        actions=("use_source_backed_analyst_or_expectation_gap_template",),
    )
    missing_actions = {
        "verify_ai_relevance_maps_to_business_impact": (
            "refresh_ai_source_evidence",
            "SEC filing documents",
            (
                "python -m ai_stock_discovery.cli run-pipeline "
                f"--sec-submissions-ticker {queue_row.ticker} --sec-submissions-limit 5 "
                "--process-filing-limit 2 --forms 10-K,10-Q,8-K"
            ),
            "Missing AI relevance evidence; refresh source-backed SEC/filing context before upgrading review status.",
        ),
        "load_or_refresh_sec_companyfacts_financials": (
            "refresh_financial_facts",
            "SEC companyfacts",
            (
                financial_replacement.replacement_command
                if financial_replacement
                else f"python -m ai_stock_discovery.cli fetch-companyfacts --ticker {queue_row.ticker}"
            ),
            _reason_with_replacement(
                "Missing SEC financial facts; load XBRL facts before using fundamental improvement scores.",
                financial_replacement,
            ),
        ),
        "load_valuation_snapshot_or_peer_comparison": (
            "refresh_valuation_context",
            "Financial Modeling Prep quote API",
            (
                valuation_replacement.replacement_command
                if valuation_replacement
                else (
                    "python -m ai_stock_discovery.cli refresh-ai-seed-data "
                    f"--ticker {queue_row.ticker} --skip-sec-facts --skip-fmp-financials --skip-analyst --skip-finra"
                )
            ),
            _reason_with_replacement(
                "Missing valuation context; refresh FMP quote/profile or import a source-backed peer comparison.",
                valuation_replacement,
            ),
        ),
        "load_expectation_gap_or_analyst_coverage_evidence": (
            "collect_expectation_gap_evidence",
            "Expectation gap evidence",
            (
                expectation_replacement.replacement_command
                if expectation_replacement
                else "python -m ai_stock_discovery.cli import-expectation-gap-csv --csv expectation_gap_signals.csv"
            ),
            _reason_with_replacement(
                "Missing expectation-gap evidence; import only source-backed low-coverage, estimate, or market-label evidence.",
                expectation_replacement,
            ),
        ),
        "add_dated_source_backed_catalyst": (
            "collect_catalyst_evidence",
            "Financial Modeling Prep earnings calendar API",
            f"python -m ai_stock_discovery.cli fetch-fmp-earnings-calendar --ticker {queue_row.ticker}",
            "Missing dated catalyst evidence; fetch source-backed earnings dates or import catalyst calendar events with original URLs.",
        ),
        "complete_risk_review": (
            "refresh_risk_review",
            "Risk review evidence",
            f"python -m ai_stock_discovery.cli extract-risk-flags --ticker {queue_row.ticker} --limit 50",
            "Missing risk review evidence; run conservative extraction or import source-backed risk flags.",
        ),
    }
    if "load_market_confirmation" in checks:
        missing_actions["load_market_confirmation"] = (
            "refresh_market_confirmation",
            "Yahoo Finance chart endpoint prototype",
            (
                market_replacement.replacement_command
                if market_replacement
                else (
                    "python -m ai_stock_discovery.cli fetch-yahoo-market-bars "
                    f"--ticker {queue_row.ticker} --benchmark QQQ --lookback-days 90"
                )
            ),
            _reason_with_replacement(
                "Missing market confirmation bars/signals; refresh source-backed daily bars before anomaly detection.",
                market_replacement,
            ),
        )
    for check, (action_type, source_name, command, reason) in missing_actions.items():
        if check not in checks:
            continue
        _put_row(
            planned,
            _row(
                conn,
                queue_row,
                context,
                as_of_date,
                action_type=action_type,
                source_name=source_name,
                due_status="missing_source_evidence",
                last_observed_at=_last_observed_for_action(conn, queue_row.ticker, action_type),
                due_after_days=0,
                recommended_command=command,
                reason=reason,
            ),
        )
    if "add_dated_source_backed_catalyst" in checks:
        _add_free_news_catalyst_search_action(conn, planned, queue_row, context, as_of_date)
    if "complete_risk_review" in checks:
        _add_short_sale_risk_context_action(conn, planned, queue_row, context, as_of_date)


def _add_free_news_catalyst_search_action(
    conn: sqlite3.Connection,
    planned: dict[tuple[str, str], RefreshPlanRow],
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
) -> None:
    query = _gdelt_catalyst_query(queue_row, context)
    command = (
        "python -m ai_stock_discovery.cli fetch-news-rss --all-defaults --limit 50; "
        f"python -m ai_stock_discovery.cli fetch-gdelt-doc-news --query {_ps_single_quote(query)} "
        "--limit 10 --timespan 30d; "
        f"python -m ai_stock_discovery.cli extract-catalysts --ticker {queue_row.ticker} --limit 100"
    )
    _put_row(
        planned,
        _row(
            conn,
            queue_row,
            context,
            as_of_date,
            action_type="search_news_for_catalysts",
            source_name="GDELT DOC API",
            due_status="missing_source_evidence",
            last_observed_at=_latest_news_or_catalyst_at(conn, queue_row.ticker),
            due_after_days=0,
            recommended_command=command,
            reason=(
                "Missing catalyst evidence; refresh free announcement RSS and run a focused GDELT article-metadata "
                "search before extracting or importing dated catalyst records. News search metadata is not a "
                "company-level conclusion."
            ),
        ),
    )


def _add_short_sale_risk_context_action(
    conn: sqlite3.Connection,
    planned: dict[tuple[str, str], RefreshPlanRow],
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
) -> None:
    _put_row(
        planned,
        _row(
            conn,
            queue_row,
            context,
            as_of_date,
            action_type="refresh_short_sale_context",
            source_name="FINRA Daily Short Sale Volume",
            due_status="missing_source_evidence",
            last_observed_at=_latest_short_sale_at(conn, queue_row.ticker),
            due_after_days=0,
            recommended_command=(
                "python -m ai_stock_discovery.cli fetch-finra-short-sale-volume "
                f"--ticker {queue_row.ticker} --lookback-days 10"
            ),
            reason=(
                "Missing risk review evidence; add FINRA daily short-sale-volume context as a review input only. "
                "FINRA daily short sale volume is not short interest and is not an automatic bearish signal."
            ),
        ),
    )


def _add_ir_monitor_action(
    conn: sqlite3.Connection,
    planned: dict[tuple[str, str], RefreshPlanRow],
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
    include_scheduled: bool,
) -> None:
    has_ir_config = bool(context.get("ir_url")) or _active_ir_page_count(conn, queue_row.ticker) > 0
    if not has_ir_config:
        has_website = bool(context.get("website"))
        command = (
            f"python -m ai_stock_discovery.cli discover-ir-pages --ticker {queue_row.ticker} --limit 1"
            if has_website
            else "python -m ai_stock_discovery.cli import-ir-urls --csv ir_urls.csv"
        )
        reason = (
            "No active IR URL is configured, but a source-backed company website is available; "
            "try conservative IR page discovery from the official website."
            if has_website
            else "No active IR URL is configured, so the IR monitor cannot check page hash changes."
        )
        _put_row(
            planned,
            _row(
                conn,
                queue_row,
                context,
                as_of_date,
                action_type="configure_ir_monitor",
                source_name="Company IR page",
                due_status="missing_local_configuration",
                last_observed_at=None,
                due_after_days=0,
                recommended_command=command,
                reason=reason,
            ),
        )
        return

    last_checked_at = _latest_ir_checked_at(conn, queue_row.ticker)
    status = _due_status(last_checked_at, queue_row.next_review_due_days, as_of_date)
    if status == "scheduled" and not include_scheduled:
        return
    command = (
        f"python -m ai_stock_discovery.cli check-ir-url --ticker {queue_row.ticker}"
        if context.get("ir_url")
        else "python -m ai_stock_discovery.cli check-ir-pages --limit 20"
    )
    _put_row(
        planned,
        _row(
            conn,
            queue_row,
            context,
            as_of_date,
            action_type="check_ir_monitor",
            source_name="Company IR page",
            due_status=status,
            last_observed_at=last_checked_at,
            due_after_days=queue_row.next_review_due_days,
            recommended_command=command,
            reason="IR page hash/title checks help detect company announcements without storing full page content.",
        ),
    )


def _add_external_research_verification_action(
    conn: sqlite3.Connection,
    planned: dict[tuple[str, str], RefreshPlanRow],
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
) -> None:
    last_observed_at = _latest_external_research_at(conn, queue_row.ticker)
    if not last_observed_at:
        return
    if _has_source_backed_profile(conn, queue_row.ticker):
        return
    if queue_row.review_priority == "ready_for_human_review":
        return
    _put_row(
        planned,
        _row(
            conn,
            queue_row,
            context,
            as_of_date,
            action_type="verify_external_research_lead",
            source_name=LOCAL_RESEARCH_SOURCE_NAME,
            due_status="missing_primary_source_verification",
            last_observed_at=last_observed_at,
            due_after_days=0,
            recommended_command=(
                "python -m ai_stock_discovery.cli enrich-fmp-profiles "
                f"--ticker {queue_row.ticker} --include-existing"
            ),
            reason=(
                "A local external research lead exists for this ticker. Verify it with company profile, SEC filings, "
                "company IR, original announcements, or API data before using it as company-level evidence."
            ),
        ),
    )


def _add_recency_action(
    conn: sqlite3.Connection,
    planned: dict[tuple[str, str], RefreshPlanRow],
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
    include_scheduled: bool,
    *,
    action_type: str,
    source_name: str,
    last_observed_at: str | None,
    due_after_days: int,
    recommended_command: str,
    reason: str,
) -> None:
    status = _due_status(last_observed_at, due_after_days, as_of_date)
    if status == "scheduled" and not include_scheduled:
        return
    _put_row(
        planned,
        _row(
            conn,
            queue_row,
            context,
            as_of_date,
            action_type=action_type,
            source_name=source_name,
            due_status=status,
            last_observed_at=last_observed_at,
            due_after_days=due_after_days,
            recommended_command=recommended_command,
            reason=reason,
        ),
    )


def _row(
    conn: sqlite3.Connection,
    queue_row: ReviewQueueRow,
    context: dict[str, str | None],
    as_of_date: date,
    *,
    action_type: str,
    source_name: str,
    due_status: str,
    last_observed_at: str | None,
    due_after_days: int,
    recommended_command: str,
    reason: str,
) -> RefreshPlanRow:
    return RefreshPlanRow(
        ticker=queue_row.ticker,
        company_name=queue_row.company_name or context.get("company_name"),
        review_bucket=queue_row.review_bucket,
        review_priority=queue_row.review_priority,
        score_total=queue_row.score_total,
        tracking_frequency=queue_row.tracking_frequency,
        action_type=action_type,
        source_name=source_name,
        due_status=due_status,
        last_observed_at=last_observed_at,
        days_since_observed=_days_since(last_observed_at, as_of_date),
        due_after_days=due_after_days,
        recommended_command=recommended_command,
        source_status=_source_status(conn, source_name),
        reason=reason,
        plan_notes="Refresh planning only; verify original sources and do not treat this as investment advice.",
    )


def _put_row(planned: dict[tuple[str, str], RefreshPlanRow], row: RefreshPlanRow) -> None:
    key = (row.ticker, row.action_type)
    existing = planned.get(key)
    if existing is None or _due_status_rank(row.due_status) < _due_status_rank(existing.due_status):
        planned[key] = row


def _source_failure_replacements(conn: sqlite3.Connection) -> dict[str, list[SourceFailureRow]]:
    rows = build_source_failure_report(conn, status="open", limit=500)
    by_ticker: dict[str, list[SourceFailureRow]] = {}
    for row in rows:
        if not row.ticker:
            continue
        by_ticker.setdefault(row.ticker.upper(), []).append(row)
    return by_ticker


def _best_replacement(
    replacements: dict[str, list[SourceFailureRow]],
    ticker: str,
    *,
    actions: tuple[str, ...],
) -> SourceFailureRow | None:
    action_rank = {action: index for index, action in enumerate(actions)}
    matches = [
        row
        for row in replacements.get(ticker.upper(), [])
        if row.replacement_action in action_rank and row.replacement_command
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda row: (_priority_rank(row.replacement_priority), action_rank[row.replacement_action]))[0]


def _reason_with_replacement(base_reason: str, replacement: SourceFailureRow | None) -> str:
    if replacement is None:
        return base_reason
    return f"{base_reason} Replacement from source_failures: {replacement.replacement_notes}"


def _priority_rank(priority: str) -> int:
    ranks = {"P0": 0, "P1": 1, "P2": 2}
    return ranks.get(priority.upper(), 9)


def _company_context(conn: sqlite3.Connection, ticker: str) -> dict[str, str | None]:
    row = conn.execute(
        """
        SELECT company_name, cik, ir_url, website
        FROM company_profile
        WHERE ticker = ?
        """,
        (ticker,),
    ).fetchone()
    if row:
        return {
            "company_name": row["company_name"],
            "cik": row["cik"],
            "ir_url": row["ir_url"],
            "website": row["website"],
        }
    universe_row = conn.execute(
        """
        SELECT company_name
        FROM universe
        WHERE ticker = ?
        """,
        (ticker,),
    ).fetchone()
    return {
        "company_name": universe_row["company_name"] if universe_row else None,
        "cik": None,
        "ir_url": None,
        "website": None,
    }


def _last_observed_for_action(conn: sqlite3.Connection, ticker: str, action_type: str) -> str | None:
    if action_type == "refresh_financial_facts":
        return _latest_financial_at(conn, ticker)
    if action_type == "refresh_valuation_context":
        return _latest_valuation_at(conn, ticker)
    if action_type == "refresh_market_confirmation":
        return _latest_market_bar_at(conn, ticker)
    if action_type == "collect_catalyst_evidence":
        return _latest_news_or_catalyst_at(conn, ticker)
    if action_type == "refresh_risk_review":
        return _latest_table_value(conn, "risk_flags", "ticker", ticker, "updated_at")
    if action_type == "collect_expectation_gap_evidence":
        return _latest_expectation_gap_at(conn, ticker)
    if action_type == "refresh_ai_source_evidence":
        return _latest_ai_source_at(conn, ticker)
    if action_type == "verify_external_research_lead":
        return _latest_external_research_at(conn, ticker)
    if action_type == "search_news_for_catalysts":
        return _latest_news_or_catalyst_at(conn, ticker)
    if action_type == "refresh_short_sale_context":
        return _latest_short_sale_at(conn, ticker)
    return None


def _latest_financial_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT period AS value FROM financial_facts WHERE ticker = ?
        UNION ALL
        SELECT updated_at FROM financial_facts WHERE ticker = ?
        """,
        (ticker, ticker),
    )


def _latest_valuation_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT date AS value FROM valuation_snapshots WHERE ticker = ?
        UNION ALL
        SELECT updated_at FROM valuation_snapshots WHERE ticker = ?
        UNION ALL
        SELECT COALESCE(comparison_date, updated_at) FROM peer_valuation_comparisons WHERE ticker = ?
        """,
        (ticker, ticker, ticker),
    )


def _latest_market_bar_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT bar_date AS value FROM market_price_bars WHERE ticker = ?
        UNION ALL
        SELECT updated_at FROM market_price_bars WHERE ticker = ?
        UNION ALL
        SELECT COALESCE(signal_date, updated_at) FROM market_confirmation_signals WHERE ticker = ?
        """,
        (ticker, ticker, ticker),
    )


def _latest_news_or_catalyst_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT COALESCE(published_at, fetched_at) AS value FROM news_events WHERE related_ticker = ?
        UNION ALL
        SELECT COALESCE(catalyst_date, updated_at) FROM catalysts WHERE ticker = ?
        UNION ALL
        SELECT fetched_at FROM evidence_items
        WHERE related_ticker = ? AND related_module IN ('news_monitor', 'catalyst')
        """,
        (ticker, ticker, ticker),
    )


def _latest_short_sale_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT trade_date AS value FROM short_sale_volume WHERE ticker = ?
        UNION ALL
        SELECT updated_at FROM short_sale_volume WHERE ticker = ?
        """,
        (ticker, ticker),
    )


def _latest_expectation_gap_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT COALESCE(signal_date, updated_at) AS value FROM expectation_gap_signals WHERE ticker = ?
        UNION ALL
        SELECT COALESCE(event_date, updated_at) FROM analyst_estimate_events WHERE ticker = ?
        """,
        (ticker, ticker),
    )


def _latest_ai_source_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT COALESCE(signal_date, updated_at) AS value FROM ai_relevance_signals WHERE ticker = ?
        UNION ALL
        SELECT updated_at FROM ai_industry_tags WHERE ticker = ?
        """,
        (ticker, ticker),
    )


def _latest_external_research_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_union_value(
        conn,
        """
        SELECT COALESCE(published_at, fetched_at) AS value
        FROM evidence_items
        WHERE related_ticker = ? AND related_module = ?
        """,
        (ticker, LOCAL_RESEARCH_RELATED_MODULE),
    )


def _has_source_backed_profile(conn: sqlite3.Connection, ticker: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM company_profile
        WHERE ticker = ?
          AND COALESCE(source, '') != ''
          AND COALESCE(source, '') != 'SEC company_tickers'
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    return row is not None


def _gdelt_catalyst_query(queue_row: ReviewQueueRow, context: dict[str, str | None]) -> str:
    company_name = (queue_row.company_name or context.get("company_name") or "").strip()
    subject = company_name or queue_row.ticker
    return (
        f'"{subject}" '
        '(AI OR "data center" OR earnings OR guidance OR backlog OR order OR customer OR partnership)'
    )


def _ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _latest_ir_checked_at(conn: sqlite3.Connection, ticker: str) -> str | None:
    return _latest_table_value(
        conn,
        "ir_pages",
        "ticker",
        ticker,
        "last_checked_at",
        extra="AND status = 'active'",
    )


def _active_ir_page_count(conn: sqlite3.Connection, ticker: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM ir_pages
        WHERE ticker = ? AND status = 'active'
        """,
        (ticker,),
    ).fetchone()
    return int(row["count"] or 0)


def _latest_table_value(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    value: str,
    target_column: str,
    *,
    extra: str = "",
) -> str | None:
    row = conn.execute(
        f"""
        SELECT MAX({target_column}) AS value
        FROM {table}
        WHERE {column} = ? {extra}
        """,
        (value,),
    ).fetchone()
    return row["value"] if row and row["value"] else None


def _latest_union_value(conn: sqlite3.Connection, query: str, params: tuple[object, ...]) -> str | None:
    rows = conn.execute(
        f"""
        SELECT value
        FROM ({query})
        WHERE value IS NOT NULL AND value != ''
        """,
        params,
    ).fetchall()
    parsed_values: list[tuple[date, str]] = []
    fallback: str | None = None
    for row in rows:
        raw = str(row["value"])
        fallback = raw
        parsed = _parse_date(raw)
        if parsed is not None:
            parsed_values.append((parsed, raw))
    if parsed_values:
        return max(parsed_values, key=lambda item: item[0])[1]
    return fallback


def _source_status(conn: sqlite3.Connection, source_name: str) -> str:
    row = conn.execute(
        """
        SELECT status, reason, last_checked_at
        FROM data_source_status
        WHERE source_name = ?
        """,
        (source_name,),
    ).fetchone()
    if not row:
        return "unknown"
    parts = [row["status"]]
    if row["last_checked_at"]:
        parts.append(f"last_checked_at={row['last_checked_at']}")
    if row["reason"]:
        parts.append(str(row["reason"]))
    return " | ".join(parts)


def _due_status(last_observed_at: str | None, due_after_days: int, as_of_date: date) -> str:
    if not last_observed_at:
        return "never_checked"
    days = _days_since(last_observed_at, as_of_date)
    if days is None:
        return "timestamp_unparseable"
    if days >= due_after_days:
        return "stale"
    return "scheduled"


def _days_since(value: str | None, as_of_date: date) -> int | None:
    observed = _parse_date(value)
    if observed is None:
        return None
    return max(0, (as_of_date - observed).days)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    if not text or text.lower() == "now":
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _as_of_date(value: str | datetime | date | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _parse_date(value)
    if parsed is None:
        raise ValueError(f"Could not parse as_of date: {value}")
    return parsed


def _due_status_rank(status: str) -> int:
    order = {
        "missing_primary_source_verification": 0,
        "missing_source_evidence": 1,
        "missing_local_configuration": 2,
        "never_checked": 3,
        "timestamp_unparseable": 4,
        "stale": 5,
        "scheduled": 6,
    }
    return order.get(status, 9)


def _bucket_rank(bucket: str) -> int:
    order = {
        "priority_research": 0,
        "watch_observe": 1,
        "light_monitor": 2,
        "evidence_gap_review": 3,
        "archive_or_event_watch": 4,
    }
    return order.get(bucket, 9)
