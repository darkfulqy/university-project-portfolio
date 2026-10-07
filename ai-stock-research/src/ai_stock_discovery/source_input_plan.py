from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ai_stock_discovery.source_input_import import SOURCE_STATUS_NAMES
from ai_stock_discovery.source_input_templates import (
    SourceInputTemplate,
    build_source_input_audit,
)


@dataclass(frozen=True)
class SourceInputPlanRow:
    template_name: str
    priority: str
    readiness_areas: str
    readiness_statuses: str
    template_path: str
    audit_status: str
    import_status: str
    data_rows: int
    source_reference_columns: str
    required_columns: str
    source_status_name: str
    recommended_next_action: str
    template_command: str
    audit_command: str
    dry_run_command: str
    apply_command: str
    notes: str


def build_source_input_plan(
    templates: list[SourceInputTemplate],
    *,
    input_dir: Path,
    readiness_statuses: Mapping[str, str] | None = None,
    audit_output: Path = Path("reports/source_input_audit.csv"),
    import_output: Path = Path("reports/source_input_import.csv"),
) -> list[SourceInputPlanRow]:
    audit_rows = build_source_input_audit(templates, input_dir=input_dir)
    statuses = readiness_statuses or {}
    rows: list[SourceInputPlanRow] = []
    for template, audit in zip(templates, audit_rows, strict=True):
        import_status = "dry_run_ready" if audit.audit_status == "ready_to_import" else "not_ready"
        rows.append(
            SourceInputPlanRow(
                template_name=template.template_name,
                priority=template.priority,
                readiness_areas=";".join(template.readiness_areas),
                readiness_statuses=_readiness_statuses(template, statuses),
                template_path=audit.template_path,
                audit_status=audit.audit_status,
                import_status=import_status,
                data_rows=audit.data_rows,
                source_reference_columns=audit.source_reference_columns,
                required_columns=";".join(template.required_columns),
                source_status_name=SOURCE_STATUS_NAMES.get(template.template_name, ""),
                recommended_next_action=_recommended_next_action(
                    template,
                    audit_status=audit.audit_status,
                    template_path=audit.template_path,
                ),
                template_command=_template_command(template, input_dir),
                audit_command=_audit_command(template, input_dir, audit_output),
                dry_run_command=_dry_run_command(template, input_dir, import_output),
                apply_command=_apply_command(template, input_dir, import_output),
                notes=_notes(template, audit.notes),
            )
        )
    return rows


def write_source_input_plan_csv(rows: list[SourceInputPlanRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceInputPlanRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _readiness_statuses(
    template: SourceInputTemplate,
    statuses: Mapping[str, str],
) -> str:
    return ";".join(f"{area}={statuses.get(area, 'unknown')}" for area in template.readiness_areas)


def _recommended_next_action(
    template: SourceInputTemplate,
    *,
    audit_status: str,
    template_path: str,
) -> str:
    if audit_status == "missing_template":
        return "Generate the blank template, then fill only source-backed rows."
    if audit_status == "missing_header":
        return "Restore the CSV header before adding or importing data."
    if audit_status == "missing_required_columns":
        return "Add the missing required columns before any import attempt."
    if audit_status == "ready_to_fill":
        return f"Fill {template_path} with source-backed rows, then rerun the audit command."
    if audit_status == "missing_required_values":
        return "Complete every required value in populated rows, then rerun the audit command."
    if audit_status == "source_reference_gap":
        return "Add original source URLs or local source paths for every populated row."
    if audit_status == "placeholder_source_reference":
        return "Replace placeholder source references with original URLs or local source paths."
    if audit_status == "ready_to_import":
        return "Review the dry-run report, then run the apply command only if every row is source-backed."
    if audit_status == "blocked":
        return "Fix the local template read error, then rerun the audit command."
    return "Review the audit status before taking action."


def _template_command(template: SourceInputTemplate, input_dir: Path) -> str:
    return (
        "python -m ai_stock_discovery.cli build-source-input-templates "
        f"--template {template.template_name} --output-dir {_display_path(input_dir)}"
    )


def _audit_command(template: SourceInputTemplate, input_dir: Path, audit_output: Path) -> str:
    return (
        "python -m ai_stock_discovery.cli build-source-input-audit "
        f"--template {template.template_name} --input-dir {_display_path(input_dir)} "
        f"--output {_display_path(audit_output)}"
    )


def _dry_run_command(template: SourceInputTemplate, input_dir: Path, import_output: Path) -> str:
    return (
        "python -m ai_stock_discovery.cli import-source-input-pack "
        f"--template {template.template_name} --input-dir {_display_path(input_dir)} "
        f"--output {_display_path(import_output)}"
    )


def _apply_command(template: SourceInputTemplate, input_dir: Path, import_output: Path) -> str:
    return _dry_run_command(template, input_dir, import_output) + " --apply"


def _notes(template: SourceInputTemplate, audit_notes: str) -> str:
    return f"{template.source_notes} Audit notes: {audit_notes}"


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")
