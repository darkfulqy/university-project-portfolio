from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class BacktestRow:
    ticker: str
    snapshot_at: str
    run_id: int | None
    score_total: float
    research_status: str
    evidence_coverage_score: float
    review_bucket: str
    horizon_bars: int
    entry_date: str | None
    exit_date: str | None
    entry_close: float | None
    exit_close: float | None
    forward_return_pct: float | None
    benchmark_return_pct: float | None
    excess_return_pct: float | None
    source_path: str | None
    outcome_status: str
    notes: str


def build_snapshot_backtest(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 500,
    horizon_bars: int = 5,
    min_score: float | None = None,
    buckets: list[str] | None = None,
    include_incomplete: bool = True,
) -> list[BacktestRow]:
    horizon = max(1, horizon_bars)
    rows: list[BacktestRow] = []
    for snapshot in _snapshot_rows(conn, tickers=tickers, limit=limit, min_score=min_score, buckets=buckets):
        row = _backtest_snapshot(conn, snapshot, horizon_bars=horizon)
        if include_incomplete or row.outcome_status == "complete":
            rows.append(row)
    return rows


def write_backtest_csv(rows: list[BacktestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BacktestRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _snapshot_rows(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None,
    limit: int,
    min_score: float | None,
    buckets: list[str] | None,
) -> list[sqlite3.Row]:
    conditions: list[str] = []
    params: list[object] = []
    if tickers:
        placeholders = ", ".join("?" for _ in tickers)
        conditions.append(f"ticker IN ({placeholders})")
        params.extend(ticker.upper() for ticker in tickers)
    if min_score is not None:
        conditions.append("score_total >= ?")
        params.append(min_score)
    if buckets:
        placeholders = ", ".join("?" for _ in buckets)
        conditions.append(f"review_bucket IN ({placeholders})")
        params.extend(buckets)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)
    return conn.execute(
        f"""
        SELECT ticker, snapshot_at, run_id, score_total, research_status,
               evidence_coverage_score, review_bucket
        FROM score_snapshots
        {where}
        ORDER BY snapshot_at DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _backtest_snapshot(
    conn: sqlite3.Connection,
    snapshot: sqlite3.Row,
    *,
    horizon_bars: int,
) -> BacktestRow:
    ticker = snapshot["ticker"]
    snapshot_date = str(snapshot["snapshot_at"])[:10]
    bars = _future_bars(conn, ticker=ticker, from_date=snapshot_date, limit=horizon_bars + 1)
    base = {
        "ticker": ticker,
        "snapshot_at": snapshot["snapshot_at"],
        "run_id": snapshot["run_id"],
        "score_total": snapshot["score_total"],
        "research_status": snapshot["research_status"],
        "evidence_coverage_score": snapshot["evidence_coverage_score"],
        "review_bucket": snapshot["review_bucket"],
        "horizon_bars": horizon_bars,
    }
    if not bars:
        return BacktestRow(
            **base,
            entry_date=None,
            exit_date=None,
            entry_close=None,
            exit_close=None,
            forward_return_pct=None,
            benchmark_return_pct=None,
            excess_return_pct=None,
            source_path=None,
            outcome_status="no_price_bar",
            notes="No local price bar is available on or after the snapshot date; no outcome is inferred.",
        )
    entry = bars[0]
    if len(bars) <= horizon_bars:
        return BacktestRow(
            **base,
            entry_date=entry["bar_date"],
            exit_date=None,
            entry_close=float(entry["close"]),
            exit_close=None,
            forward_return_pct=None,
            benchmark_return_pct=None,
            excess_return_pct=None,
            source_path=entry["source_path"],
            outcome_status="insufficient_forward_bars",
            notes="Local price bars are insufficient for the requested horizon; no return is inferred.",
        )
    exit_bar = bars[horizon_bars]
    entry_close = float(entry["close"])
    exit_close = float(exit_bar["close"])
    forward_return = (exit_close - entry_close) / entry_close if entry_close else None
    benchmark_return = _benchmark_return(entry, exit_bar)
    excess_return = (
        forward_return - benchmark_return
        if forward_return is not None and benchmark_return is not None
        else None
    )
    return BacktestRow(
        **base,
        entry_date=entry["bar_date"],
        exit_date=exit_bar["bar_date"],
        entry_close=entry_close,
        exit_close=exit_close,
        forward_return_pct=_to_pct(forward_return),
        benchmark_return_pct=_to_pct(benchmark_return),
        excess_return_pct=_to_pct(excess_return),
        source_path=_source_paths(entry, exit_bar),
        outcome_status="complete",
        notes="Retrospective local price-bar calculation only; not predictive and not investment advice.",
    )


def _future_bars(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    from_date: str,
    limit: int,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT ticker, bar_date, close, benchmark_close, source_path
        FROM market_price_bars
        WHERE ticker = ? AND bar_date >= ?
        ORDER BY bar_date, updated_at
        LIMIT ?
        """,
        (ticker, from_date, limit),
    ).fetchall()


def _benchmark_return(entry: sqlite3.Row, exit_bar: sqlite3.Row) -> float | None:
    entry_value = entry["benchmark_close"]
    exit_value = exit_bar["benchmark_close"]
    if entry_value is None or exit_value is None:
        return None
    entry_float = float(entry_value)
    if entry_float == 0:
        return None
    return (float(exit_value) - entry_float) / entry_float


def _to_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100.0, 6)


def _source_paths(entry: sqlite3.Row, exit_bar: sqlite3.Row) -> str:
    paths = [str(entry["source_path"])]
    exit_path = str(exit_bar["source_path"])
    if exit_path not in paths:
        paths.append(exit_path)
    return ";".join(paths)
