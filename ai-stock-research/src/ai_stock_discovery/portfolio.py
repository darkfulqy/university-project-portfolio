from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from ai_stock_discovery.review_queue import build_review_queue
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class PortfolioPosition:
    portfolio_name: str
    ticker: str
    position_date: str | None
    quantity: float | None
    market_value: float | None
    weight_pct: float | None
    source_name: str
    source_path: str
    notes: str | None
    content_hash: str


@dataclass(frozen=True)
class PortfolioRiskRow:
    portfolio_name: str
    ticker: str
    position_date: str | None
    weight_pct: float | None
    market_value: float | None
    sector: str | None
    industry: str | None
    score_total: float
    review_bucket: str
    review_priority: str
    evidence_coverage_score: float
    active_risk_flag_count: int
    source_link_count: int
    concentration_note: str
    evidence_note: str
    risk_note: str
    source_path: str
    report_notes: str


def import_portfolio_positions_csv(
    conn: sqlite3.Connection,
    csv_path: Path,
    *,
    default_portfolio: str = "default",
) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        if "ticker" not in set(reader.fieldnames):
            raise ValueError("Portfolio positions CSV missing required column: ticker")
        positions = [
            _position_from_row(row, csv_path, default_portfolio=default_portfolio)
            for row in reader
        ]
    return upsert_portfolio_positions(conn, positions)


def upsert_portfolio_positions(conn: sqlite3.Connection, positions: list[PortfolioPosition]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO portfolio_positions (
            portfolio_name, ticker, position_date, quantity, market_value,
            weight_pct, source_name, source_path, notes, content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            quantity=excluded.quantity,
            market_value=excluded.market_value,
            weight_pct=excluded.weight_pct,
            source_name=excluded.source_name,
            source_path=excluded.source_path,
            notes=excluded.notes,
            updated_at=excluded.updated_at
        """,
        [
            (
                position.portfolio_name,
                position.ticker,
                position.position_date,
                position.quantity,
                position.market_value,
                position.weight_pct,
                position.source_name,
                position.source_path,
                position.notes,
                position.content_hash,
                now,
            )
            for position in positions
        ],
    )
    return len(positions)


def build_portfolio_risk_report(
    conn: sqlite3.Connection,
    *,
    portfolio_name: str | None = None,
    limit: int = 500,
) -> list[PortfolioRiskRow]:
    positions = _latest_positions(conn, portfolio_name=portfolio_name, limit=limit)
    inferred_weights = _infer_weights(positions)
    rows: list[PortfolioRiskRow] = []
    for position in positions:
        ticker = position["ticker"]
        review_row = build_review_queue(conn, tickers=[ticker], limit=1)[0]
        profile = _profile(conn, ticker)
        weight_pct = _row_float(position, "weight_pct")
        if weight_pct is None:
            weight_pct = inferred_weights.get((position["portfolio_name"], ticker))
        active_risks = _active_risk_flag_count(conn, ticker)
        rows.append(
            PortfolioRiskRow(
                portfolio_name=position["portfolio_name"],
                ticker=ticker,
                position_date=position["position_date"],
                weight_pct=weight_pct,
                market_value=_row_float(position, "market_value"),
                sector=profile["sector"] if profile else None,
                industry=profile["industry"] if profile else None,
                score_total=review_row.score_total,
                review_bucket=review_row.review_bucket,
                review_priority=review_row.review_priority,
                evidence_coverage_score=review_row.evidence_coverage_score,
                active_risk_flag_count=active_risks,
                source_link_count=review_row.source_link_count,
                concentration_note=_concentration_note(weight_pct),
                evidence_note=_evidence_note(review_row.review_bucket, review_row.review_priority),
                risk_note=_risk_note(active_risks),
                source_path=position["source_path"],
                report_notes="Portfolio risk review only; not allocation advice or investment advice.",
            )
        )
    rows.sort(
        key=lambda row: (
            row.portfolio_name,
            -(row.weight_pct or 0.0),
            -row.score_total,
            row.ticker,
        )
    )
    return rows


def write_portfolio_risk_csv(rows: list[PortfolioRiskRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PortfolioRiskRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _position_from_row(
    row: dict[str, str],
    csv_path: Path,
    *,
    default_portfolio: str,
) -> PortfolioPosition:
    ticker = row.get("ticker", "").strip().upper()
    if not ticker:
        raise ValueError("Portfolio positions CSV contains a row without ticker.")
    portfolio_name = row.get("portfolio_name", "").strip() or default_portfolio
    position_date = row.get("position_date", "").strip() or row.get("date", "").strip() or None
    source_name = row.get("source_name", "").strip() or csv_path.name
    source_path = row.get("source_path", "").strip() or str(csv_path)
    quantity = _optional_float(row.get("quantity"))
    market_value = _optional_float(row.get("market_value"))
    weight_pct = _normalize_weight_pct(row.get("weight_pct") or row.get("weight"))
    notes = row.get("notes", "").strip() or None
    content_hash = hashlib.sha256(
        "|".join(
            [
                portfolio_name,
                ticker,
                position_date or "",
                source_name,
                source_path,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return PortfolioPosition(
        portfolio_name=portfolio_name,
        ticker=ticker,
        position_date=position_date,
        quantity=quantity,
        market_value=market_value,
        weight_pct=weight_pct,
        source_name=source_name,
        source_path=source_path,
        notes=notes,
        content_hash=content_hash,
    )


def _latest_positions(
    conn: sqlite3.Connection,
    *,
    portfolio_name: str | None,
    limit: int,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = ""
    if portfolio_name:
        where = "WHERE portfolio_name = ?"
        params.append(portfolio_name)
    rows = conn.execute(
        f"""
        SELECT *
        FROM portfolio_positions
        {where}
        ORDER BY portfolio_name, ticker, COALESCE(position_date, '') DESC, updated_at DESC, id DESC
        """,
        params,
    ).fetchall()
    latest: list[sqlite3.Row] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["portfolio_name"], row["ticker"])
        if key in seen:
            continue
        latest.append(row)
        seen.add(key)
        if len(latest) >= limit:
            break
    return latest


def _infer_weights(rows: list[sqlite3.Row]) -> dict[tuple[str, str], float]:
    totals: dict[str, float] = {}
    for row in rows:
        value = _row_float(row, "market_value")
        if value is not None:
            totals[row["portfolio_name"]] = totals.get(row["portfolio_name"], 0.0) + value
    weights: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["weight_pct"] is not None:
            continue
        value = _row_float(row, "market_value")
        total = totals.get(row["portfolio_name"], 0.0)
        if value is not None and total > 0:
            weights[(row["portfolio_name"], row["ticker"])] = value / total * 100.0
    return weights


def _profile(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT sector, industry
        FROM company_profile
        WHERE ticker = ?
        """,
        (ticker,),
    ).fetchone()


def _active_risk_flag_count(conn: sqlite3.Connection, ticker: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM risk_flags
        WHERE ticker = ? AND status IN ('active', 'watch')
        """,
        (ticker,),
    ).fetchone()
    return int(row["count"] or 0)


def _concentration_note(weight_pct: float | None) -> str:
    if weight_pct is None:
        return "missing_position_weight_or_market_value"
    if weight_pct >= 25.0:
        return "high_concentration_review_required"
    if weight_pct >= 10.0:
        return "elevated_position_weight_review"
    return "position_weight_within_basic_review_threshold"


def _evidence_note(review_bucket: str, review_priority: str) -> str:
    if review_bucket == "evidence_gap_review":
        return f"evidence_gap:{review_priority}"
    return "core_evidence_present_verify_original_sources"


def _risk_note(active_risk_flag_count: int) -> str:
    if active_risk_flag_count:
        return f"{active_risk_flag_count}_active_or_watch_risk_flag(s)_require_review"
    return "no_active_or_watch_risk_flag_recorded"


def _normalize_weight_pct(raw: str | None) -> float | None:
    value = _optional_float(raw)
    if value is None:
        return None
    if 0 <= value <= 1:
        return value * 100.0
    return value


def _optional_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _row_float(row: sqlite3.Row, key: str) -> float | None:
    value = row[key]
    if value is None:
        return None
    return float(value)
