from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ai_stock_discovery.analysis.industry_tags import import_ai_tags_csv
from ai_stock_discovery.source_input_templates import (
    SourceInputTemplate,
    build_source_input_audit,
)
from ai_stock_discovery.sources import (
    analysts,
    catalyst_calendar,
    expectations,
    finra,
    guidance,
    macro,
    market,
    peer_valuation,
    sec,
    transcripts,
)


@dataclass(frozen=True)
class SourceInputImportRow:
    template_name: str
    priority: str
    readiness_areas: str
    template_path: str
    audit_status: str
    import_status: str
    data_rows: int
    rows_imported: int
    source_status_name: str
    recommended_action: str
    notes: str


SOURCE_STATUS_NAMES: dict[str, str] = {
    "ai_industry_tags": "Manual AI industry tags CSV",
    "catalyst_calendar": "Manual catalyst calendar CSV",
    "market_confirmation_signals": "Local market confirmation CSV",
    "market_price_bars": "Local market price bars CSV",
    "finra_short_sale_volume": finra.DEFAULT_SHORT_SALE_SOURCE,
    "macro_indicators": "Manual macro indicator CSV",
    "expectation_gap_signals": "Expectation gap CSV",
    "analyst_estimate_events": "Analyst estimate events CSV",
    "guidance_events": "Manual guidance events CSV",
    "peer_valuation": "Manual peer valuation CSV",
    "transcript_snippets": "Manual transcript snippets CSV",
}


def import_source_input_pack(
    conn: sqlite3.Connection,
    templates: list[SourceInputTemplate],
    *,
    input_dir: Path,
    apply: bool = False,
    finra_source_url: str | None = None,
    transcript_scan_ai: bool = True,
    transcript_scan_risks: bool = True,
) -> list[SourceInputImportRow]:
    audit_rows = build_source_input_audit(templates, input_dir=input_dir)
    rows: list[SourceInputImportRow] = []
    for index, (template, audit) in enumerate(zip(templates, audit_rows, strict=True), start=1):
        source_status_name = SOURCE_STATUS_NAMES.get(template.template_name, "")
        if audit.audit_status != "ready_to_import":
            rows.append(
                _import_row(
                    template,
                    audit_status=audit.audit_status,
                    template_path=audit.template_path,
                    import_status="not_ready",
                    data_rows=audit.data_rows,
                    rows_imported=0,
                    source_status_name=source_status_name,
                    recommended_action=audit.recommended_action,
                    notes=audit.notes,
                )
            )
            continue
        if not apply:
            rows.append(
                _import_row(
                    template,
                    audit_status=audit.audit_status,
                    template_path=audit.template_path,
                    import_status="dry_run_ready",
                    data_rows=audit.data_rows,
                    rows_imported=0,
                    source_status_name=source_status_name,
                    recommended_action="Rerun import-source-input-pack with --apply after reviewing the audit.",
                    notes="Dry run only; database was not modified.",
                )
            )
            continue
        template_path = input_dir / template.filename
        savepoint = f"source_input_{index}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            imported = _import_template(
                conn,
                template.template_name,
                template_path,
                finra_source_url=finra_source_url,
                transcript_scan_ai=transcript_scan_ai,
                transcript_scan_risks=transcript_scan_risks,
            )
            _mark_source_status(conn, template.template_name, template_path, imported, finra_source_url)
        except Exception as exc:  # noqa: BLE001 - keep other ready templates reportable.
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            rows.append(
                _import_row(
                    template,
                    audit_status=audit.audit_status,
                    template_path=audit.template_path,
                    import_status="import_error",
                    data_rows=audit.data_rows,
                    rows_imported=0,
                    source_status_name=source_status_name,
                    recommended_action="Fix the source input file and rerun with --apply.",
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        rows.append(
            _import_row(
                template,
                audit_status=audit.audit_status,
                template_path=audit.template_path,
                import_status="imported",
                data_rows=audit.data_rows,
                rows_imported=imported,
                source_status_name=source_status_name,
                recommended_action="Review downstream watchlist, evidence audit, and research cards.",
                notes="Imported only source-backed rows that passed the local audit.",
            )
        )
    return rows


def write_source_input_import_csv(rows: list[SourceInputImportRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceInputImportRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _import_template(
    conn: sqlite3.Connection,
    template_name: str,
    template_path: Path,
    *,
    finra_source_url: str | None,
    transcript_scan_ai: bool,
    transcript_scan_risks: bool,
) -> int:
    if template_name == "ai_industry_tags":
        return import_ai_tags_csv(conn, template_path)
    if template_name == "catalyst_calendar":
        return catalyst_calendar.import_catalyst_calendar_csv(conn, template_path)
    if template_name == "market_confirmation_signals":
        return market.import_market_confirmation_csv(conn, template_path)
    if template_name == "market_price_bars":
        return market.import_market_price_bars_csv(conn, template_path)
    if template_name == "finra_short_sale_volume":
        return finra.import_short_sale_volume_file(
            conn,
            template_path,
            source_url=finra_source_url,
        )
    if template_name == "macro_indicators":
        return macro.import_macro_indicators_csv(conn, template_path)
    if template_name == "expectation_gap_signals":
        return expectations.import_expectation_gap_csv(conn, template_path)
    if template_name == "analyst_estimate_events":
        return analysts.import_analyst_estimate_events_csv(conn, template_path)
    if template_name == "guidance_events":
        return guidance.import_guidance_events_csv(conn, template_path)
    if template_name == "peer_valuation":
        return peer_valuation.import_peer_valuation_csv(conn, template_path)
    if template_name == "transcript_snippets":
        result = transcripts.import_transcript_snippets_csv(
            conn,
            template_path,
            scan_ai=transcript_scan_ai,
            scan_risks=transcript_scan_risks,
        )
        return result.snippets_imported
    raise ValueError(f"Unsupported source input template: {template_name}")


def _mark_source_status(
    conn: sqlite3.Connection,
    template_name: str,
    template_path: Path,
    rows_imported: int,
    finra_source_url: str | None,
) -> None:
    source_name = SOURCE_STATUS_NAMES.get(template_name)
    if not source_name:
        return
    reason = f"Imported {rows_imported} row(s) from {template_path} via import-source-input-pack."
    if template_name == "finra_short_sale_volume":
        source = finra_source_url or str(template_path)
        reason = (
            f"Imported {rows_imported} FINRA daily short sale volume row(s) from {source}; "
            "this is not short interest."
        )
    elif template_name == "transcript_snippets":
        reason = (
            f"Imported {rows_imported} source-backed transcript snippet row(s) from {template_path}; "
            "snippets require original-source review."
        )
    elif template_name == "analyst_estimate_events":
        reason = (
            f"Imported {rows_imported} row(s) from {template_path}; no analyst estimate data was inferred."
        )
    sec.mark_source_status(conn, source_name=source_name, status="ok", reason=reason)


def _import_row(
    template: SourceInputTemplate,
    *,
    audit_status: str,
    template_path: str,
    import_status: str,
    data_rows: int,
    rows_imported: int,
    source_status_name: str,
    recommended_action: str,
    notes: str,
) -> SourceInputImportRow:
    return SourceInputImportRow(
        template_name=template.template_name,
        priority=template.priority,
        readiness_areas=";".join(template.readiness_areas),
        template_path=template_path,
        audit_status=audit_status,
        import_status=import_status,
        data_rows=data_rows,
        rows_imported=rows_imported,
        source_status_name=source_status_name,
        recommended_action=recommended_action,
        notes=notes,
    )
