from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import sqlite3

from ai_stock_discovery.review_queue import build_review_queue
from ai_stock_discovery.source_failures import build_source_failure_report
from ai_stock_discovery.timeutils import utc_now_iso


def build_ops_report(
    conn: sqlite3.Connection,
    *,
    review_limit: int = 20,
    source_limit: int = 80,
    evidence_limit: int = 30,
    source_input_import_path: Path = Path("reports/source_input_import.csv"),
    source_input_plan_path: Path = Path("reports/source_input_plan.csv"),
    source_input_limit: int = 20,
    source_input_plan_limit: int = 20,
) -> str:
    latest_run = _latest_pipeline_run(conn)
    source_rows = _source_status_rows(conn, limit=source_limit)
    source_failure_rows = build_source_failure_report(conn, status="open", limit=source_limit)
    evidence_rows = _evidence_module_rows(conn, limit=evidence_limit)
    review_rows = build_review_queue(conn, limit=review_limit)
    source_input_rows = _source_input_import_rows(source_input_import_path, limit=source_input_limit)
    source_input_plan_rows = _source_input_plan_rows(source_input_plan_path, limit=source_input_plan_limit)

    lines = [
        "# AI Stock Discovery Operations Report",
        "",
        f"Generated at: {utc_now_iso()}",
        "",
        (
            "Purpose: local operating health for the MVP pipeline, data-source status, "
            "evidence inventory, and human review queue."
        ),
        "",
        "Boundary: this report is for research operations only and is not investment advice.",
        "",
    ]
    lines.extend(_pipeline_section(conn, latest_run))
    lines.extend(_source_status_section(source_rows))
    lines.extend(_source_failures_section(source_failure_rows))
    lines.extend(_source_input_import_section(source_input_rows, source_input_import_path))
    lines.extend(_source_input_action_plan_section(source_input_plan_rows, source_input_plan_path))
    lines.extend(_evidence_section(evidence_rows))
    lines.extend(_review_queue_section(review_rows))
    lines.extend(_operating_notes(source_rows, review_rows))
    return "\n".join(lines).rstrip() + "\n"


def write_ops_report(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _pipeline_section(conn: sqlite3.Connection, latest_run: sqlite3.Row | None) -> list[str]:
    lines = ["## Latest Pipeline Run", ""]
    if latest_run is None:
        return lines + ["No pipeline run has been recorded yet.", ""]

    steps = _pipeline_steps(conn, run_id=int(latest_run["id"]))
    status_counts = Counter(str(step["status"]) for step in steps)
    lines.extend(
        [
            f"- Run id: {latest_run['id']}",
            f"- Status: {latest_run['status']}",
            f"- Started at: {latest_run['started_at']}",
            f"- Completed at: {latest_run['completed_at'] or 'NA'}",
            f"- Step status counts: {_format_counts(status_counts)}",
            "",
        ]
    )
    if steps:
        lines.extend(
            _markdown_table(
                ["step", "status", "records", "message"],
                [
                    [
                        step["step_name"],
                        step["status"],
                        _format_optional(step["records_changed"]),
                        step["error"] or step["message"] or "",
                    ]
                    for step in steps
                ],
            )
        )
    else:
        lines.append("No pipeline steps have been recorded for this run.")
    lines.append("")
    return lines


def _source_status_section(rows: list[sqlite3.Row]) -> list[str]:
    lines = ["## Data Source Status", ""]
    if not rows:
        return lines + ["No data source status has been recorded yet.", ""]

    status_counts = Counter(str(row["status"]) for row in rows)
    lines.extend([f"- Status counts: {_format_counts(status_counts)}", ""])
    lines.extend(
        _markdown_table(
            ["source", "status", "last_checked_at", "reason"],
            [
                [
                    row["source_name"],
                    row["status"],
                    row["last_checked_at"],
                    row["reason"] or "",
                ]
                for row in rows
            ],
        )
    )
    lines.append("")
    return lines


def _source_input_import_section(rows: list[dict[str, str]], path: Path) -> list[str]:
    lines = ["## Source Input Import Plan", ""]
    if not path.exists():
        return lines + [
            f"No source input import dry-run report found at {path}.",
            "",
        ]
    if not rows:
        return lines + [
            f"Source input import dry-run report at {path} has no rows.",
            "",
        ]

    status_counts = Counter(row.get("import_status", "") or "unknown" for row in rows)
    ready_count = status_counts.get("dry_run_ready", 0)
    lines.extend(
        [
            f"- Report path: {path}",
            f"- Import status counts: {_format_counts(status_counts)}",
            f"- Rows ready for explicit --apply: {ready_count}",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["template", "audit_status", "import_status", "data_rows", "rows_imported", "recommended_action"],
            [
                [
                    row.get("template_name", ""),
                    row.get("audit_status", ""),
                    row.get("import_status", ""),
                    row.get("data_rows", ""),
                    row.get("rows_imported", ""),
                    row.get("recommended_action", ""),
                ]
                for row in rows
            ],
        )
    )
    lines.append("")
    return lines


def _source_input_action_plan_section(rows: list[dict[str, str]], path: Path) -> list[str]:
    lines = ["## Source Input Action Plan", ""]
    if not path.exists():
        return lines + [
            f"No source input action plan found at {path}.",
            "",
        ]
    if not rows:
        return lines + [
            f"Source input action plan at {path} has no rows.",
            "",
        ]

    audit_counts = Counter(row.get("audit_status", "") or "unknown" for row in rows)
    import_counts = Counter(row.get("import_status", "") or "unknown" for row in rows)
    ready_count = import_counts.get("dry_run_ready", 0)
    lines.extend(
        [
            f"- Report path: {path}",
            f"- Audit status counts: {_format_counts(audit_counts)}",
            f"- Import status counts: {_format_counts(import_counts)}",
            f"- Rows ready for explicit --apply after review: {ready_count}",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["template", "readiness", "audit_status", "import_status", "next_action", "audit_command"],
            [
                [
                    row.get("template_name", ""),
                    row.get("readiness_statuses", ""),
                    row.get("audit_status", ""),
                    row.get("import_status", ""),
                    row.get("recommended_next_action", ""),
                    row.get("audit_command", ""),
                ]
                for row in rows
            ],
        )
    )
    lines.append("")
    return lines


def _evidence_section(rows: list[sqlite3.Row]) -> list[str]:
    lines = ["## Evidence Inventory", ""]
    total = sum(int(row["evidence_count"] or 0) for row in rows)
    lines.append(f"- Evidence item count in listed modules: {total}")
    lines.append("")
    if rows:
        lines.extend(
            _markdown_table(
                ["related_module", "evidence_count", "latest_fetched_at"],
                [
                    [
                        row["related_module"],
                        row["evidence_count"],
                        row["latest_fetched_at"] or "NA",
                    ]
                    for row in rows
                ],
            )
        )
    else:
        lines.append("No evidence_items rows have been recorded yet.")
    lines.append("")
    return lines


def _review_queue_section(rows: list) -> list[str]:
    lines = ["## Human Review Queue", ""]
    if not rows:
        return lines + ["No review queue rows are available from current local evidence.", ""]

    bucket_counts = Counter(row.review_bucket for row in rows)
    priority_counts = Counter(row.review_priority for row in rows)
    lines.extend(
        [
            f"- Review bucket counts: {_format_counts(bucket_counts)}",
            f"- Review priority counts: {_format_counts(priority_counts)}",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["ticker", "score", "bucket", "frequency", "required_checks"],
            [
                [
                    row.ticker,
                    f"{row.score_total:.2f}",
                    row.review_bucket,
                    row.tracking_frequency,
                    row.required_review_checks,
                ]
                for row in rows
            ],
        )
    )
    lines.append("")
    return lines


def _operating_notes(source_rows: list[sqlite3.Row], review_rows: list) -> list[str]:
    source_issues = [
        row for row in source_rows if row["status"] in {"unavailable", "degraded"}
    ]
    evidence_gap_count = sum(1 for row in review_rows if row.review_bucket == "evidence_gap_review")
    lines = ["## Operating Notes", ""]
    if source_issues:
        lines.append(
            f"- {len(source_issues)} data source(s) need attention before affected evidence can be refreshed."
        )
    else:
        lines.append("- No degraded or unavailable data source status is currently recorded.")
    if evidence_gap_count:
        lines.append(
            f"- {evidence_gap_count} review queue row(s) are evidence-gap items; do not upgrade conviction until missing source categories are addressed."
        )
    else:
        lines.append("- Current listed review queue rows are not blocked by core evidence gaps.")
    lines.append("- Verify original source links before using any company-level conclusion.")
    lines.append("")
    return lines


def _source_failures_section(rows: list) -> list[str]:
    lines = ["## Open Source Failures", ""]
    if not rows:
        return lines + ["No ticker-level open source failures are currently recorded.", ""]

    failure_counts = Counter(row.failure_type for row in rows)
    lines.extend(
        [
            f"- Open failure count: {len(rows)}",
            f"- Failure type counts: {_format_counts(failure_counts)}",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "ticker",
                "source",
                "endpoint",
                "failure_type",
                "replacement",
                "recommended_command",
                "occurrences",
                "last_seen_at",
                "reason",
            ],
            [
                [
                    row.ticker or "NA",
                    row.source_name,
                    row.endpoint or "NA",
                    row.failure_type,
                    row.replacement_action,
                    row.replacement_command,
                    row.occurrences,
                    row.last_seen_at,
                    row.reason,
                ]
                for row in rows[:20]
            ],
        )
    )
    lines.append("")
    return lines


def _latest_pipeline_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, run_type, status, started_at, completed_at, summary
        FROM pipeline_runs
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()


def _pipeline_steps(conn: sqlite3.Connection, *, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT step_name, status, records_changed, message, error
        FROM pipeline_steps
        WHERE run_id = ?
        ORDER BY id
        """,
        (run_id,),
    ).fetchall()


def _source_status_rows(conn: sqlite3.Connection, *, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT source_name, status, reason, last_checked_at
        FROM data_source_status
        ORDER BY
            CASE status
                WHEN 'unavailable' THEN 0
                WHEN 'degraded' THEN 1
                WHEN 'ok' THEN 2
                ELSE 3
            END,
            source_name
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _evidence_module_rows(conn: sqlite3.Connection, *, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            related_module,
            COUNT(*) AS evidence_count,
            MAX(fetched_at) AS latest_fetched_at
        FROM evidence_items
        GROUP BY related_module
        ORDER BY evidence_count DESC, related_module
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _source_input_import_rows(path: Path, *, limit: int) -> list[dict[str, str]]:
    return _csv_rows(path, limit=limit)


def _source_input_plan_rows(path: Path, *, limit: int) -> list[dict[str, str]]:
    return _csv_rows(path, limit=limit)


def _csv_rows(path: Path, *, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for _, row in zip(range(limit), reader)]


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _format_optional(value: object) -> str:
    return "NA" if value is None else str(value)
