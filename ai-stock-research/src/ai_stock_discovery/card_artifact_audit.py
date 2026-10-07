from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3

from ai_stock_discovery.review_queue import build_review_queue
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class CardArtifactAuditRow:
    ticker: str
    card_path: str
    file_exists: bool
    indexed: bool
    in_current_review_queue: bool
    active_exclusion: bool
    exclusion_reason_types: str
    exclusion_sources: str
    artifact_status: str
    recommended_action: str
    notes: str


@dataclass(frozen=True)
class _IndexedCard:
    ticker: str
    card_path: str
    file_name: str


@dataclass(frozen=True)
class CardArtifactArchiveRow:
    ticker: str
    artifact_status: str
    source_path: str
    archive_path: str
    applied: bool
    action_status: str
    notes: str


def build_card_artifact_audit(
    conn: sqlite3.Connection,
    *,
    cards_dir: Path = Path("reports/cards"),
    card_index_path: Path = Path("reports/card_index.csv"),
    limit: int = 200,
    min_score: float | None = None,
) -> list[CardArtifactAuditRow]:
    indexed_cards = _read_card_index(card_index_path)
    indexed_by_name = {card.file_name.upper(): card for card in indexed_cards}
    existing_files = sorted(cards_dir.glob("*.md"))
    existing_by_name = {path.name.upper(): path for path in existing_files}
    review_tickers = {
        row.ticker
        for row in build_review_queue(
            conn,
            limit=limit,
            min_score=min_score,
            cards_dir=cards_dir,
        )
    }
    names = sorted(set(indexed_by_name) | set(existing_by_name))
    rows: list[CardArtifactAuditRow] = []
    for file_name in names:
        indexed_card = indexed_by_name.get(file_name)
        existing_file = existing_by_name.get(file_name)
        ticker = _ticker_from_artifact(indexed_card, existing_file)
        exclusions = _active_exclusions(conn, ticker)
        rows.append(
            _audit_row(
                ticker=ticker,
                card_path=_display_card_path(indexed_card, existing_file),
                file_exists=existing_file is not None,
                indexed=indexed_card is not None,
                in_current_review_queue=ticker in review_tickers,
                exclusions=exclusions,
            )
        )
    rows.sort(key=lambda row: (_status_rank(row.artifact_status), row.ticker, row.card_path))
    return rows


def write_card_artifact_audit_csv(rows: list[CardArtifactAuditRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CardArtifactAuditRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def archive_card_artifacts(
    conn: sqlite3.Connection,
    *,
    cards_dir: Path = Path("reports/cards"),
    card_index_path: Path = Path("reports/card_index.csv"),
    archive_dir: Path = Path("reports/cards/_archive"),
    statuses: set[str] | None = None,
    apply: bool = False,
    limit: int = 200,
    min_score: float | None = None,
) -> list[CardArtifactArchiveRow]:
    selected_statuses = statuses or {"stale_excluded_card", "stale_unindexed_card"}
    audit_rows = build_card_artifact_audit(
        conn,
        cards_dir=cards_dir,
        card_index_path=card_index_path,
        limit=limit,
        min_score=min_score,
    )
    batch_dir = archive_dir / _archive_batch_name()
    rows: list[CardArtifactArchiveRow] = []
    for audit_row in audit_rows:
        if audit_row.artifact_status not in selected_statuses:
            continue
        rows.append(
            _archive_candidate(
                audit_row=audit_row,
                cards_dir=cards_dir,
                batch_dir=batch_dir,
                apply=apply,
            )
        )
    return rows


def write_card_artifact_archive_csv(rows: list[CardArtifactArchiveRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CardArtifactArchiveRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _audit_row(
    *,
    ticker: str,
    card_path: str,
    file_exists: bool,
    indexed: bool,
    in_current_review_queue: bool,
    exclusions: list[sqlite3.Row],
) -> CardArtifactAuditRow:
    active_exclusion = bool(exclusions)
    reason_types = ";".join(sorted({str(row["reason_type"]) for row in exclusions}))
    sources = ";".join(sorted({str(row["source_name"]) for row in exclusions}))
    status, action, notes = _artifact_status(
        file_exists=file_exists,
        indexed=indexed,
        in_current_review_queue=in_current_review_queue,
        active_exclusion=active_exclusion,
    )
    return CardArtifactAuditRow(
        ticker=ticker,
        card_path=card_path,
        file_exists=file_exists,
        indexed=indexed,
        in_current_review_queue=in_current_review_queue,
        active_exclusion=active_exclusion,
        exclusion_reason_types=reason_types,
        exclusion_sources=sources,
        artifact_status=status,
        recommended_action=action,
        notes=notes,
    )


def _archive_candidate(
    *,
    audit_row: CardArtifactAuditRow,
    cards_dir: Path,
    batch_dir: Path,
    apply: bool,
) -> CardArtifactArchiveRow:
    source = Path(audit_row.card_path)
    destination = batch_dir / source.name
    safety_error = _archive_safety_error(source=source, cards_dir=cards_dir, destination=destination)
    if safety_error:
        return CardArtifactArchiveRow(
            ticker=audit_row.ticker,
            artifact_status=audit_row.artifact_status,
            source_path=str(source),
            archive_path=str(destination),
            applied=False,
            action_status="blocked_unsafe_path",
            notes=safety_error,
        )
    if not source.exists():
        return CardArtifactArchiveRow(
            ticker=audit_row.ticker,
            artifact_status=audit_row.artifact_status,
            source_path=str(source),
            archive_path=str(destination),
            applied=False,
            action_status="skipped_missing_file",
            notes="Source generated card file no longer exists.",
        )
    if not apply:
        return CardArtifactArchiveRow(
            ticker=audit_row.ticker,
            artifact_status=audit_row.artifact_status,
            source_path=str(source),
            archive_path=str(destination),
            applied=False,
            action_status="planned_dry_run",
            notes="Dry run only; pass --apply to archive this generated card.",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return CardArtifactArchiveRow(
        ticker=audit_row.ticker,
        artifact_status=audit_row.artifact_status,
        source_path=str(source),
        archive_path=str(destination),
        applied=True,
        action_status="archived",
        notes="Generated card moved to archive; no raw data, database row, or source evidence was modified.",
    )


def _archive_safety_error(*, source: Path, cards_dir: Path, destination: Path) -> str | None:
    try:
        source_resolved = source.resolve()
        cards_dir_resolved = cards_dir.resolve()
        destination_resolved = destination.resolve()
        archive_parent_resolved = destination.parent.resolve()
    except OSError as exc:
        return f"Could not resolve archive path safely: {type(exc).__name__}: {exc}"
    if source_resolved.suffix.lower() != ".md":
        return "Only generated Markdown card files can be archived."
    if source_resolved.parent != cards_dir_resolved:
        return "Only top-level files directly inside cards_dir can be archived."
    if not _is_relative_to(archive_parent_resolved, cards_dir_resolved):
        return "Archive destination must remain inside cards_dir."
    if destination_resolved.exists():
        return "Archive destination already exists."
    return None


def _artifact_status(
    *,
    file_exists: bool,
    indexed: bool,
    in_current_review_queue: bool,
    active_exclusion: bool,
) -> tuple[str, str, str]:
    if indexed and not file_exists:
        return (
            "indexed_missing_file",
            "regenerate_current_research_cards",
            "The current card index points to a missing generated card file.",
        )
    if active_exclusion and not indexed:
        return (
            "stale_excluded_card",
            "review_stale_generated_card_before_use",
            "Generated card file is outside the current index and ticker has an active exclusion; no file was deleted.",
        )
    if active_exclusion:
        return (
            "indexed_excluded_card",
            "review_exclusion_before_using_card",
            "Ticker has an active universe exclusion; verify before using this generated card.",
        )
    if not indexed:
        return (
            "stale_unindexed_card",
            "review_stale_generated_card_before_use",
            "Generated card file is outside the current card index; no file was deleted.",
        )
    if not in_current_review_queue:
        return (
            "indexed_not_in_current_review_queue",
            "regenerate_card_index_or_review_filters",
            "Card is indexed but absent from the current review queue under the selected filters.",
        )
    return (
        "current_indexed_card",
        "review_original_sources_before_company_level_use",
        "Card is in the current index and review queue; still verify original sources before any conclusion.",
    )


def _read_card_index(card_index_path: Path) -> list[_IndexedCard]:
    if not card_index_path.exists():
        return []
    rows: list[_IndexedCard] = []
    with card_index_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_path = str(row.get("card_path") or "").strip()
            ticker = str(row.get("ticker") or "").strip().upper()
            if not raw_path and not ticker:
                continue
            file_name = Path(raw_path).name.upper() if raw_path else f"{ticker}.MD"
            if not ticker:
                ticker = Path(file_name).stem.upper()
            rows.append(_IndexedCard(ticker=ticker, card_path=raw_path, file_name=file_name))
    return rows


def _active_exclusions(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT reason_type, source_name
        FROM universe_exclusions
        WHERE ticker = ?
          AND is_active = 1
          AND severity = 'exclude'
        ORDER BY reason_type, source_name
        """,
        (ticker.upper(),),
    ).fetchall()


def _ticker_from_artifact(indexed_card: _IndexedCard | None, existing_file: Path | None) -> str:
    if indexed_card is not None:
        return indexed_card.ticker.upper()
    if existing_file is not None:
        return existing_file.stem.upper()
    return ""


def _display_card_path(indexed_card: _IndexedCard | None, existing_file: Path | None) -> str:
    if indexed_card is not None and indexed_card.card_path:
        return indexed_card.card_path
    if existing_file is not None:
        return str(existing_file)
    if indexed_card is not None:
        return indexed_card.file_name
    return ""


def _status_rank(status: str) -> int:
    order = {
        "stale_excluded_card": 0,
        "indexed_excluded_card": 1,
        "indexed_missing_file": 2,
        "stale_unindexed_card": 3,
        "indexed_not_in_current_review_queue": 4,
        "current_indexed_card": 5,
    }
    return order.get(status, 9)


def _archive_batch_name() -> str:
    return utc_now_iso().replace(":", "").replace("+", "Z").replace("-", "").replace(".", "")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
