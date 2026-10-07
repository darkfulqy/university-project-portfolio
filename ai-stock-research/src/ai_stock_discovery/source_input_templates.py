from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class SourceInputTemplate:
    template_name: str
    filename: str
    priority: str
    readiness_areas: tuple[str, ...]
    import_command: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    source_notes: str

    @property
    def columns(self) -> tuple[str, ...]:
        return self.required_columns + self.optional_columns


@dataclass(frozen=True)
class SourceInputTemplateRow:
    template_name: str
    priority: str
    readiness_areas: str
    template_path: str
    import_command: str
    required_columns: str
    optional_columns: str
    write_status: str
    notes: str


@dataclass(frozen=True)
class SourceInputAuditRow:
    template_name: str
    priority: str
    readiness_areas: str
    template_path: str
    file_status: str
    audit_status: str
    data_rows: int
    missing_required_columns: str
    rows_missing_required_values: int
    source_reference_columns: str
    rows_with_source_reference: int
    rows_with_placeholder_source_reference: int
    import_command: str
    recommended_action: str
    notes: str


SOURCE_INPUT_TEMPLATES: tuple[SourceInputTemplate, ...] = (
    SourceInputTemplate(
        template_name="ai_industry_tags",
        filename="ai_industry_tags.csv",
        priority="P0",
        readiness_areas=("table_ai_chain_tags",),
        import_command="python -m ai_stock_discovery.cli import-ai-tags --csv {path}",
        required_columns=("ticker", "tag"),
        optional_columns=("confidence", "source_type", "source_url", "evidence_snippet"),
        source_notes=(
            "Manual AI chain tags must cite source-backed evidence; do not tag a ticker from theme exposure alone."
        ),
    ),
    SourceInputTemplate(
        template_name="catalyst_calendar",
        filename="catalyst_calendar.csv",
        priority="P0",
        readiness_areas=("table_catalysts",),
        import_command="python -m ai_stock_discovery.cli import-catalyst-calendar-csv --csv {path}",
        required_columns=("ticker", "catalyst_type", "catalyst_date", "description", "source_url", "confidence"),
        optional_columns=("status",),
        source_notes="Only enter dated, source-backed events; this is not a prediction that the catalyst will occur.",
    ),
    SourceInputTemplate(
        template_name="market_confirmation_signals",
        filename="market_confirmation_signals.csv",
        priority="P0",
        readiness_areas=("source_market_confirmation", "table_market_confirmation"),
        import_command="python -m ai_stock_discovery.cli import-market-confirmation-csv --csv {path}",
        required_columns=("ticker", "signal_type", "direction", "description", "confidence"),
        optional_columns=("signal_date", "source_type", "source_name", "source_path", "magnitude"),
        source_notes="Use only traceable local/API market evidence; confirmation signals are small review inputs.",
    ),
    SourceInputTemplate(
        template_name="market_price_bars",
        filename="market_price_bars.csv",
        priority="P0",
        readiness_areas=("source_market_confirmation", "table_market_confirmation"),
        import_command="python -m ai_stock_discovery.cli import-market-price-bars-csv --csv {path}",
        required_columns=("ticker", "date", "close", "volume"),
        optional_columns=(
            "open",
            "high",
            "low",
            "benchmark_close",
            "premarket_price",
            "source_type",
            "source_name",
            "source_path",
        ),
        source_notes=(
            "After importing local bars, run detect-market-anomalies; price action alone is not a thesis."
        ),
    ),
    SourceInputTemplate(
        template_name="finra_short_sale_volume",
        filename="finra_short_sale_volume.csv",
        priority="P0",
        readiness_areas=("source_short_sale",),
        import_command=(
            "python -m ai_stock_discovery.cli import-finra-short-sale-csv --csv {path} "
            "--source-url SOURCE_URL_OR_LOCAL_PATH"
        ),
        required_columns=("Date", "Symbol"),
        optional_columns=("ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"),
        source_notes=(
            "FINRA daily short sale volume is not short interest and must not be treated as short positioning."
        ),
    ),
    SourceInputTemplate(
        template_name="macro_indicators",
        filename="macro_indicators.csv",
        priority="P0",
        readiness_areas=("source_macro",),
        import_command="python -m ai_stock_discovery.cli import-macro-csv --csv {path}",
        required_columns=("series_id", "source_name", "metric_name", "category", "period", "value", "source_url"),
        optional_columns=("geography", "frequency", "unit"),
        source_notes=(
            "Macro/electricity rows are background context only; they are not company orders, customers, or revenue."
        ),
    ),
    SourceInputTemplate(
        template_name="expectation_gap_signals",
        filename="expectation_gap_signals.csv",
        priority="P0",
        readiness_areas=("source_expectation_gap", "table_expectation_gap"),
        import_command="python -m ai_stock_discovery.cli import-expectation-gap-csv --csv {path}",
        required_columns=("ticker", "source_url", "signal_type", "direction", "description", "confidence"),
        optional_columns=("signal_date", "source_type", "source_name", "magnitude"),
        source_notes="Expectation-gap rows require original source links and should not infer analyst consensus.",
    ),
    SourceInputTemplate(
        template_name="analyst_estimate_events",
        filename="analyst_estimate_events.csv",
        priority="P1",
        readiness_areas=("source_expectation_gap", "table_expectation_gap"),
        import_command="python -m ai_stock_discovery.cli import-analyst-estimate-events-csv --csv {path}",
        required_columns=("ticker", "source_url", "event_type", "direction", "description", "confidence"),
        optional_columns=(
            "event_date",
            "fiscal_period",
            "metric",
            "previous_value",
            "current_value",
            "unit",
            "analyst_firm",
            "source_type",
            "source_name",
        ),
        source_notes="Analyst events must come from source-backed revisions/coverage records; do not infer consensus.",
    ),
    SourceInputTemplate(
        template_name="guidance_events",
        filename="guidance_events.csv",
        priority="P1",
        readiness_areas=("source_expectation_gap",),
        import_command="python -m ai_stock_discovery.cli import-guidance-events-csv --csv {path}",
        required_columns=("ticker", "source_url", "metric", "direction", "description", "confidence"),
        optional_columns=(
            "guidance_date",
            "fiscal_period",
            "previous_value",
            "current_value",
            "unit",
            "source_type",
            "source_name",
        ),
        source_notes="Guidance rows are review context; they do not replace SEC financial facts.",
    ),
    SourceInputTemplate(
        template_name="peer_valuation",
        filename="peer_valuation.csv",
        priority="P1",
        readiness_areas=("source_expectation_gap",),
        import_command="python -m ai_stock_discovery.cli import-peer-valuation-csv --csv {path}",
        required_columns=(
            "ticker",
            "source_url",
            "peer_tickers",
            "metric",
            "target_value",
            "peer_median",
            "description",
            "confidence",
        ),
        optional_columns=(
            "comparison_date",
            "peer_group",
            "peer_mean",
            "discount_premium_pct",
            "direction",
            "source_type",
            "source_name",
        ),
        source_notes="Peer valuation rows are small review inputs and must keep the peer set/source traceable.",
    ),
    SourceInputTemplate(
        template_name="transcript_snippets",
        filename="transcript_snippets.csv",
        priority="P1",
        readiness_areas=("source_expectation_gap",),
        import_command="python -m ai_stock_discovery.cli import-transcript-snippets-csv --csv {path}",
        required_columns=("ticker", "source_url", "transcript_excerpt", "confidence"),
        optional_columns=(
            "transcript_date",
            "fiscal_period",
            "event_type",
            "speaker",
            "source_type",
            "source_name",
            "title",
        ),
        source_notes="Transcript snippets must be short source-backed excerpts; verify against the original transcript.",
    ),
)


def select_source_input_templates(
    *,
    readiness_statuses: Mapping[str, str] | None = None,
    only_gaps: bool = False,
) -> list[SourceInputTemplate]:
    if not only_gaps:
        return list(SOURCE_INPUT_TEMPLATES)
    statuses = readiness_statuses or {}
    return [
        template
        for template in SOURCE_INPUT_TEMPLATES
        if any(statuses.get(area, "needs_data") != "ready" for area in template.readiness_areas)
    ]


def filter_source_input_templates(
    templates: list[SourceInputTemplate],
    names: list[str] | tuple[str, ...],
) -> list[SourceInputTemplate]:
    requested = _expand_template_name_args(names)
    if not requested:
        return list(templates)

    canonical_by_alias: dict[str, str] = {}
    for template in templates:
        for alias in _template_aliases(template):
            canonical_by_alias[alias] = template.template_name

    selected: set[str] = set()
    unknown: list[str] = []
    for name in requested:
        canonical = _resolve_template_name(name, canonical_by_alias)
        if canonical is None:
            unknown.append(name)
        else:
            selected.add(canonical)

    if unknown:
        available = ", ".join(template.template_name for template in templates)
        raise ValueError(
            f"Unknown source input template(s): {', '.join(unknown)}. "
            f"Available templates in this selection: {available}."
        )

    return [template for template in templates if template.template_name in selected]


def write_source_input_template_pack(
    templates: list[SourceInputTemplate],
    *,
    output_dir: Path,
    manifest_path: Path,
    overwrite: bool = False,
) -> list[SourceInputTemplateRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SourceInputTemplateRow] = []
    for template in templates:
        template_path = output_dir / template.filename
        write_status = "exists"
        if overwrite or not template_path.exists():
            with template_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(template.columns)
            write_status = "written"
        display_path = _display_path(template_path)
        rows.append(
            SourceInputTemplateRow(
                template_name=template.template_name,
                priority=template.priority,
                readiness_areas=";".join(template.readiness_areas),
                template_path=display_path,
                import_command=template.import_command.format(path=display_path),
                required_columns=";".join(template.required_columns),
                optional_columns=";".join(template.optional_columns),
                write_status=write_status,
                notes=template.source_notes,
            )
        )
    write_source_input_template_manifest(rows, manifest_path)
    return rows


def build_source_input_audit(
    templates: list[SourceInputTemplate],
    *,
    input_dir: Path,
) -> list[SourceInputAuditRow]:
    return [_audit_template(template, input_dir=input_dir) for template in templates]


def write_source_input_audit_csv(rows: list[SourceInputAuditRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceInputAuditRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_source_input_template_manifest(rows: list[SourceInputTemplateRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceInputTemplateRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _expand_template_name_args(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for raw in names:
        expanded.extend(part.strip() for part in raw.split(",") if part.strip())
    return tuple(expanded)


def _template_aliases(template: SourceInputTemplate) -> tuple[str, ...]:
    aliases = {
        template.template_name,
        template.filename,
        Path(template.filename).stem,
    }
    return tuple(alias.lower() for alias in aliases)


def _resolve_template_name(name: str, canonical_by_alias: Mapping[str, str]) -> str | None:
    path_name = Path(name).name
    aliases = {
        name,
        path_name,
        Path(path_name).stem,
    }
    for alias in aliases:
        canonical = canonical_by_alias.get(alias.lower())
        if canonical:
            return canonical
    return None


def _audit_template(template: SourceInputTemplate, *, input_dir: Path) -> SourceInputAuditRow:
    template_path = input_dir / template.filename
    display_path = _display_path(template_path)
    import_command = template.import_command.format(path=display_path)
    source_columns = _source_reference_columns(template)
    if not template_path.exists():
        return _audit_row(
            template,
            template_path=display_path,
            file_status="missing_file",
            audit_status="missing_template",
            data_rows=0,
            missing_required_columns=";".join(template.required_columns),
            rows_missing_required_values=0,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=0,
            rows_with_placeholder_source_reference=0,
            import_command=import_command,
            recommended_action=f"python -m ai_stock_discovery.cli build-source-input-templates --output-dir {input_dir}",
            notes="Template file is missing; no data was imported or inferred.",
        )
    try:
        with template_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return _audit_row(
                    template,
                    template_path=display_path,
                    file_status="empty_file",
                    audit_status="missing_header",
                    data_rows=0,
                    missing_required_columns=";".join(template.required_columns),
                    rows_missing_required_values=0,
                    source_reference_columns=";".join(source_columns),
                    rows_with_source_reference=0,
                    rows_with_placeholder_source_reference=0,
                    import_command=import_command,
                    recommended_action=f"Rewrite the template header with build-source-input-templates --overwrite.",
                    notes="Template file has no CSV header.",
                )
            fieldnames = tuple(name.strip() for name in reader.fieldnames if name)
            rows = [
                stripped_row
                for row in reader
                if _has_data(row)
                for stripped_row in [_strip_row(row)]
            ]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return _audit_row(
            template,
            template_path=display_path,
            file_status="read_error",
            audit_status="blocked",
            data_rows=0,
            missing_required_columns="unknown",
            rows_missing_required_values=0,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=0,
            rows_with_placeholder_source_reference=0,
            import_command=import_command,
            recommended_action="Fix the local template file before import.",
            notes=f"{type(exc).__name__}: {exc}",
        )

    missing_required = [column for column in template.required_columns if column not in fieldnames]
    rows_with_source_reference = _rows_with_source_reference(rows, source_columns)
    rows_missing_required_values = _rows_missing_required_values(rows, template.required_columns)
    rows_with_placeholder_source_reference = _rows_with_placeholder_source_reference(rows, source_columns)
    if missing_required:
        return _audit_row(
            template,
            template_path=display_path,
            file_status="has_rows" if rows else "header_only",
            audit_status="missing_required_columns",
            data_rows=len(rows),
            missing_required_columns=";".join(missing_required),
            rows_missing_required_values=0,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=rows_with_source_reference,
            rows_with_placeholder_source_reference=0,
            import_command=import_command,
            recommended_action=f"Add missing required columns: {';'.join(missing_required)}",
            notes="Required columns are checked before any import command is run.",
        )
    if not rows:
        return _audit_row(
            template,
            template_path=display_path,
            file_status="header_only",
            audit_status="ready_to_fill",
            data_rows=0,
            missing_required_columns="",
            rows_missing_required_values=0,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=0,
            rows_with_placeholder_source_reference=0,
            import_command=import_command,
            recommended_action="Fill rows only with source-backed data, then rerun this audit before importing.",
            notes=template.source_notes,
        )
    if rows_missing_required_values:
        return _audit_row(
            template,
            template_path=display_path,
            file_status="has_rows",
            audit_status="missing_required_values",
            data_rows=len(rows),
            missing_required_columns="",
            rows_missing_required_values=rows_missing_required_values,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=rows_with_source_reference,
            rows_with_placeholder_source_reference=0,
            import_command=import_command,
            recommended_action="Fill every required field in populated rows before import.",
            notes="Rows with data must have non-empty values for all required columns.",
        )
    if source_columns and rows_with_source_reference < len(rows):
        return _audit_row(
            template,
            template_path=display_path,
            file_status="has_rows",
            audit_status="source_reference_gap",
            data_rows=len(rows),
            missing_required_columns="",
            rows_missing_required_values=0,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=rows_with_source_reference,
            rows_with_placeholder_source_reference=0,
            import_command=import_command,
            recommended_action="Add source_url/source_path/source evidence for every populated row before import.",
            notes="Rows with data must remain traceable to original sources.",
        )
    if rows_with_placeholder_source_reference:
        return _audit_row(
            template,
            template_path=display_path,
            file_status="has_rows",
            audit_status="placeholder_source_reference",
            data_rows=len(rows),
            missing_required_columns="",
            rows_missing_required_values=0,
            source_reference_columns=";".join(source_columns),
            rows_with_source_reference=rows_with_source_reference,
            rows_with_placeholder_source_reference=rows_with_placeholder_source_reference,
            import_command=import_command,
            recommended_action="Replace placeholder source references with original URLs or local source paths.",
            notes="Source references must be concrete original URLs or local source paths, not placeholders.",
        )
    source_note = (
        "Source reference is supplied by the import command/source file path."
        if not source_columns
        else "Every populated row has at least one source reference column populated."
    )
    return _audit_row(
        template,
        template_path=display_path,
        file_status="has_rows",
        audit_status="ready_to_import",
        data_rows=len(rows),
        missing_required_columns="",
        rows_missing_required_values=0,
        source_reference_columns=";".join(source_columns),
        rows_with_source_reference=rows_with_source_reference,
        rows_with_placeholder_source_reference=0,
        import_command=import_command,
        recommended_action=import_command,
        notes=f"{source_note} {template.source_notes}",
    )


def _audit_row(
    template: SourceInputTemplate,
    *,
    template_path: str,
    file_status: str,
    audit_status: str,
    data_rows: int,
    missing_required_columns: str,
    rows_missing_required_values: int,
    source_reference_columns: str,
    rows_with_source_reference: int,
    rows_with_placeholder_source_reference: int,
    import_command: str,
    recommended_action: str,
    notes: str,
) -> SourceInputAuditRow:
    return SourceInputAuditRow(
        template_name=template.template_name,
        priority=template.priority,
        readiness_areas=";".join(template.readiness_areas),
        template_path=template_path,
        file_status=file_status,
        audit_status=audit_status,
        data_rows=data_rows,
        missing_required_columns=missing_required_columns,
        rows_missing_required_values=rows_missing_required_values,
        source_reference_columns=source_reference_columns,
        rows_with_source_reference=rows_with_source_reference,
        rows_with_placeholder_source_reference=rows_with_placeholder_source_reference,
        import_command=import_command,
        recommended_action=recommended_action,
        notes=notes,
    )


def _source_reference_columns(template: SourceInputTemplate) -> tuple[str, ...]:
    source_names = {"source_url", "source_path", "source", "url"}
    return tuple(column for column in template.columns if column in source_names)


def _has_data(row: dict[str, str | None]) -> bool:
    return any((value or "").strip() for value in row.values())


def _strip_row(row: dict[str, str | None]) -> dict[str, str | None]:
    return {key.strip(): value for key, value in row.items() if key}


def _rows_with_source_reference(rows: list[dict[str, str | None]], columns: tuple[str, ...]) -> int:
    if not columns:
        return 0
    return sum(1 for row in rows if any((row.get(column) or "").strip() for column in columns))


def _rows_with_placeholder_source_reference(rows: list[dict[str, str | None]], columns: tuple[str, ...]) -> int:
    if not columns:
        return 0
    return sum(
        1
        for row in rows
        if any(_is_placeholder_source_reference(row.get(column) or "") for column in columns)
    )


def _is_placeholder_source_reference(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    placeholder_values = {
        "source_url",
        "source_url_or_local_path",
        "source_path",
        "url",
        "todo",
        "tbd",
        "na",
        "n/a",
        "none",
        "unknown",
        "placeholder",
    }
    if normalized in placeholder_values:
        return True
    return "example." in normalized or normalized.startswith("example/")


def _rows_missing_required_values(rows: list[dict[str, str | None]], columns: tuple[str, ...]) -> int:
    return sum(
        1
        for row in rows
        if any(not (row.get(column) or "").strip() for column in columns)
    )
