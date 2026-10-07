from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def open_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with open_db(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        migrate_db(conn)


def migrate_db(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "rss_filing_events", "ticker", "TEXT")
    _add_column_if_missing(conn, "rss_filing_events", "accession_number", "TEXT")
    _add_column_if_missing(conn, "ir_pages", "last_error", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rss_filing_events_ticker ON rss_filing_events (ticker)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_price_bars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            bar_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            benchmark_close REAL,
            premarket_price REAL,
            source_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (content_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_market_price_bars_ticker_date ON market_price_bars (ticker, bar_date)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS score_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            run_id INTEGER,
            score_total REAL NOT NULL,
            research_status TEXT NOT NULL,
            evidence_coverage_score REAL NOT NULL,
            review_priority TEXT NOT NULL,
            review_bucket TEXT NOT NULL,
            tracking_frequency TEXT NOT NULL,
            source_link_count INTEGER NOT NULL,
            latest_evidence_at TEXT,
            required_review_checks TEXT NOT NULL,
            invalidating_conditions TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            UNIQUE (content_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_score_snapshots_ticker_time ON score_snapshots (ticker, snapshot_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            position_date TEXT,
            quantity REAL,
            market_value REAL,
            weight_pct REAL,
            source_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            notes TEXT,
            content_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (content_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_positions_portfolio_ticker ON portfolio_positions (portfolio_name, ticker, position_date)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            ticker TEXT,
            endpoint TEXT,
            failure_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            reason TEXT NOT NULL,
            source_url TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            last_success_at TEXT,
            content_hash TEXT NOT NULL,
            UNIQUE (content_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_failures_status_source ON source_failures (status, source_name, ticker)"
    )


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
