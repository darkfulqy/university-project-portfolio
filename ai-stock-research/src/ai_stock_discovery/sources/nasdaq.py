from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import sqlite3

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass(frozen=True)
class UniverseRecord:
    ticker: str
    company_name: str
    exchange: str
    security_type: str
    is_etf: bool
    is_preferred: bool
    is_unit: bool
    is_active: bool


def fetch_universe(client: HttpClient, *, limit: int | None = None) -> list[UniverseRecord]:
    records: list[UniverseRecord] = []
    records.extend(_parse_nasdaq_listed(client.get_text(NASDAQ_LISTED_URL)))
    records.extend(_parse_other_listed(client.get_text(OTHER_LISTED_URL)))
    filtered = [record for record in records if _is_common_stock_candidate(record)]
    filtered.sort(key=lambda record: record.ticker)
    return filtered[:limit] if limit else filtered


def upsert_universe(conn: sqlite3.Connection, records: list[UniverseRecord]) -> int:
    checked_at = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO universe (
            ticker, company_name, exchange, security_type, is_etf, is_preferred,
            is_unit, is_active, last_checked_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            company_name=excluded.company_name,
            exchange=excluded.exchange,
            security_type=excluded.security_type,
            is_etf=excluded.is_etf,
            is_preferred=excluded.is_preferred,
            is_unit=excluded.is_unit,
            is_active=excluded.is_active,
            last_checked_at=excluded.last_checked_at
        """,
        [
            (
                record.ticker,
                record.company_name,
                record.exchange,
                record.security_type,
                int(record.is_etf),
                int(record.is_preferred),
                int(record.is_unit),
                int(record.is_active),
                checked_at,
            )
            for record in records
        ],
    )
    return len(records)


def _parse_nasdaq_listed(text: str) -> list[UniverseRecord]:
    rows = _pipe_rows(text)
    records: list[UniverseRecord] = []
    for row in rows:
        ticker = row.get("Symbol", "").strip()
        name = row.get("Security Name", "").strip()
        if not ticker or ticker == "File Creation Time":
            continue
        records.append(
            UniverseRecord(
                ticker=ticker,
                company_name=name,
                exchange="NASDAQ",
                security_type="common_stock_candidate",
                is_etf=_yes(row.get("ETF")),
                is_preferred=_looks_preferred(ticker, name),
                is_unit=_looks_unit(ticker, name),
                is_active=not _yes(row.get("Test Issue")),
            )
        )
    return records


def _parse_other_listed(text: str) -> list[UniverseRecord]:
    rows = _pipe_rows(text)
    records: list[UniverseRecord] = []
    for row in rows:
        ticker = row.get("ACT Symbol", "").strip()
        name = row.get("Security Name", "").strip()
        if not ticker or ticker == "File Creation Time":
            continue
        records.append(
            UniverseRecord(
                ticker=ticker,
                company_name=name,
                exchange=_exchange_name(row.get("Exchange", "").strip()),
                security_type="common_stock_candidate",
                is_etf=_yes(row.get("ETF")),
                is_preferred=_looks_preferred(ticker, name),
                is_unit=_looks_unit(ticker, name),
                is_active=not _yes(row.get("Test Issue")),
            )
        )
    return records


def _pipe_rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    return list(csv.DictReader(StringIO("\n".join(lines)), delimiter="|"))


def _is_common_stock_candidate(record: UniverseRecord) -> bool:
    if record.is_etf or record.is_preferred or record.is_unit or not record.is_active:
        return False
    ticker = record.ticker.upper()
    name = record.company_name.upper()
    if any(mark in ticker for mark in ["$", "^", "+", "*"]):
        return False
    excluded_name_terms = [
        " WARRANT",
        " RIGHT",
        " PREFERRED",
        " PFD",
        " DEPOSITARY",
        " UNIT",
        " ETN",
        " ETF",
        " FUND",
        " TRUST",
    ]
    return not any(term in name for term in excluded_name_terms)


def _looks_preferred(ticker: str, name: str) -> bool:
    ticker_upper = ticker.upper()
    name_upper = name.upper()
    return "$" in ticker_upper or "PREFERRED" in name_upper or " PFD" in name_upper


def _looks_unit(ticker: str, name: str) -> bool:
    ticker_upper = ticker.upper()
    name_upper = name.upper()
    return ticker_upper.endswith("U") and (" UNIT" in name_upper or "UNITS" in name_upper)


def _yes(value: str | None) -> bool:
    return (value or "").strip().upper() == "Y"


def _exchange_name(code: str) -> str:
    return {
        "A": "NYSE American",
        "N": "NYSE",
        "P": "NYSE Arca",
        "Z": "Cboe BZX",
        "V": "IEXG",
    }.get(code, code or "UNKNOWN")
