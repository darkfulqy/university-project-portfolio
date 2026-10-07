from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class InsiderTransaction:
    ticker: str
    owner_name: str
    relationship: str | None
    transaction_date: str
    transaction_code: str | None
    acquired_disposed_code: str | None
    transaction_shares: float | None
    transaction_price: float | None
    shares_owned_following: float | None
    source_type: str
    source_name: str
    source_url: str
    content_hash: str


def import_insider_transactions_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {"ticker", "owner_name", "transaction_date", "source_url"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Insider transaction CSV missing required columns: {', '.join(sorted(missing))}")
        transactions = [_transaction_from_row(row, csv_path) for row in reader]
    return upsert_insider_transactions(conn, transactions)


def parse_ownership_xml(
    *,
    ticker: str | None,
    xml_text: str,
    source_url: str,
    source_type: str = "SEC Form 4",
    source_name: str = "SEC EDGAR ownership filing",
) -> list[InsiderTransaction]:
    root = ET.fromstring(xml_text)
    issuer_ticker = _text(root, "issuer", "issuerTradingSymbol") or _text(root, "issuerTradingSymbol")
    resolved_ticker = (ticker or issuer_ticker or "").strip().upper()
    if not resolved_ticker:
        raise ValueError("SEC ownership XML is missing ticker and issuerTradingSymbol.")
    owners = _reporting_owners(root)
    owner_name = owners[0][0] if owners else "Unknown reporting owner"
    relationship = owners[0][1] if owners else None
    transactions: list[InsiderTransaction] = []
    for node in _children(root, "nonDerivativeTransaction"):
        transaction_date = _text(node, "transactionDate", "value")
        if not transaction_date:
            continue
        tx = InsiderTransaction(
            ticker=resolved_ticker,
            owner_name=owner_name,
            relationship=relationship,
            transaction_date=transaction_date,
            transaction_code=_text(node, "transactionCoding", "transactionCode") or None,
            acquired_disposed_code=_text(
                node,
                "transactionAmounts",
                "transactionAcquiredDisposedCode",
                "value",
            )
            or None,
            transaction_shares=_xml_number(_text(node, "transactionAmounts", "transactionShares", "value")),
            transaction_price=_xml_number(_text(node, "transactionAmounts", "transactionPricePerShare", "value")),
            shares_owned_following=_xml_number(
                _text(node, "postTransactionAmounts", "sharesOwnedFollowingTransaction", "value")
            ),
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            content_hash="",
        )
        transactions.append(_with_hash(tx))
    return transactions


def upsert_insider_transactions(
    conn: sqlite3.Connection,
    transactions: list[InsiderTransaction],
) -> int:
    if not transactions:
        return 0
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO insider_transactions (
            ticker, owner_name, relationship, transaction_date, transaction_code,
            acquired_disposed_code, transaction_shares, transaction_price,
            shares_owned_following, source_type, source_name, source_url,
            content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            relationship=excluded.relationship,
            transaction_code=excluded.transaction_code,
            acquired_disposed_code=excluded.acquired_disposed_code,
            transaction_shares=excluded.transaction_shares,
            transaction_price=excluded.transaction_price,
            shares_owned_following=excluded.shares_owned_following,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            updated_at=excluded.updated_at
        """,
        [
            (
                tx.ticker,
                tx.owner_name,
                tx.relationship,
                tx.transaction_date,
                tx.transaction_code,
                tx.acquired_disposed_code,
                tx.transaction_shares,
                tx.transaction_price,
                tx.shares_owned_following,
                tx.source_type,
                tx.source_name,
                tx.source_url,
                tx.content_hash,
                now,
            )
            for tx in transactions
        ],
    )
    conn.executemany(
        """
        INSERT INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at, raw_title, summary,
            evidence_snippet, confidence, related_ticker, related_module, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            fetched_at=excluded.fetched_at,
            raw_title=excluded.raw_title,
            summary=excluded.summary,
            evidence_snippet=excluded.evidence_snippet,
            confidence=excluded.confidence,
            related_ticker=excluded.related_ticker
        """,
        [
            (
                tx.source_type,
                tx.source_name,
                tx.source_url,
                tx.transaction_date,
                now,
                f"{tx.owner_name} insider transaction",
                "Source-backed insider transaction event. A single event is not automatically a risk conclusion.",
                _snippet(tx),
                _confidence(tx.source_type, tx.source_name),
                tx.ticker,
                "insider_transactions",
                f"insider-transaction:{tx.content_hash}",
            )
            for tx in transactions
        ],
    )
    return len(transactions)


def list_insider_transactions(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    limit: int = 20,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = ""
    if ticker:
        where = "WHERE ticker = ?"
        params.append(ticker.upper())
    params.append(limit)
    return conn.execute(
        f"""
        SELECT ticker, owner_name, relationship, transaction_date, transaction_code,
               acquired_disposed_code, transaction_shares, transaction_price,
               shares_owned_following, source_type, source_name, source_url, updated_at
        FROM insider_transactions
        {where}
        ORDER BY transaction_date DESC, updated_at DESC, ticker, owner_name
        LIMIT ?
        """,
        params,
    ).fetchall()


def _transaction_from_row(row: dict[str, str], csv_path: Path) -> InsiderTransaction:
    ticker = _field(row, "ticker").upper()
    owner_name = _field(row, "owner_name", "reporting_owner", "owner")
    transaction_date = _field(row, "transaction_date", "date")
    source_url = _field(row, "source_url", "url")
    if not ticker:
        raise ValueError("Insider transaction row is missing ticker.")
    if not owner_name:
        raise ValueError(f"Insider transaction row for {ticker} is missing owner_name.")
    if not transaction_date:
        raise ValueError(f"Insider transaction row for {ticker} is missing transaction_date.")
    if not source_url:
        raise ValueError(f"Insider transaction row for {ticker} is missing source_url.")
    tx = InsiderTransaction(
        ticker=ticker,
        owner_name=owner_name,
        relationship=_field(row, "relationship", "owner_relationship") or None,
        transaction_date=transaction_date,
        transaction_code=_field(row, "transaction_code", "code") or None,
        acquired_disposed_code=_field(row, "acquired_disposed_code", "acquired_disposed", "a_d") or None,
        transaction_shares=_optional_number(_field(row, "transaction_shares", "shares")),
        transaction_price=_optional_number(_field(row, "transaction_price", "price")),
        shares_owned_following=_optional_number(_field(row, "shares_owned_following", "owned_following")),
        source_type=_field(row, "source_type") or "SEC Form 4",
        source_name=_field(row, "source_name") or csv_path.name,
        source_url=source_url,
        content_hash="",
    )
    return _with_hash(tx)


def _reporting_owners(root: ET.Element) -> list[tuple[str, str | None]]:
    owners: list[tuple[str, str | None]] = []
    for owner in _children(root, "reportingOwner"):
        name = _text(owner, "reportingOwnerId", "rptOwnerName")
        if not name:
            continue
        relationship_parts: list[str] = []
        rel = _first_child(owner, "reportingOwnerRelationship")
        if rel is not None:
            if _text(rel, "isDirector") == "1":
                relationship_parts.append("Director")
            if _text(rel, "isOfficer") == "1":
                title = _text(rel, "officerTitle")
                relationship_parts.append(f"Officer: {title}" if title else "Officer")
            if _text(rel, "isTenPercentOwner") == "1":
                relationship_parts.append("10% Owner")
            if _text(rel, "isOther") == "1":
                other = _text(rel, "otherText")
                relationship_parts.append(f"Other: {other}" if other else "Other")
        owners.append((name, "; ".join(relationship_parts) or None))
    return owners


def _text(root: ET.Element, *path: str) -> str:
    current: ET.Element | None = root
    for name in path:
        if current is None:
            return ""
        current = _first_child(current, name)
    if current is None or current.text is None:
        return ""
    return current.text.strip()


def _first_child(root: ET.Element, name: str) -> ET.Element | None:
    for child in list(root):
        if _local_name(child.tag) == name:
            return child
    return None


def _children(root: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in root.iter() if _local_name(child.tag) == name]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _with_hash(tx: InsiderTransaction) -> InsiderTransaction:
    content_hash = hashlib.sha256(
        "|".join(
            [
                tx.ticker,
                tx.owner_name,
                tx.relationship or "",
                tx.transaction_date,
                tx.transaction_code or "",
                tx.acquired_disposed_code or "",
                _number_text(tx.transaction_shares),
                _number_text(tx.transaction_price),
                _number_text(tx.shares_owned_following),
                tx.source_type,
                tx.source_name,
                tx.source_url,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return InsiderTransaction(
        ticker=tx.ticker,
        owner_name=tx.owner_name,
        relationship=tx.relationship,
        transaction_date=tx.transaction_date,
        transaction_code=tx.transaction_code,
        acquired_disposed_code=tx.acquired_disposed_code,
        transaction_shares=tx.transaction_shares,
        transaction_price=tx.transaction_price,
        shares_owned_following=tx.shares_owned_following,
        source_type=tx.source_type,
        source_name=tx.source_name,
        source_url=tx.source_url,
        content_hash=content_hash,
    )


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _optional_number(raw: str) -> float | None:
    value = raw.strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric insider transaction value, got {raw!r}.") from exc


def _xml_number(raw: str) -> float | None:
    value = raw.strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _number_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _snippet(tx: InsiderTransaction) -> str:
    parts = [
        f"{tx.owner_name}",
        f"date={tx.transaction_date}",
    ]
    if tx.relationship:
        parts.append(f"relationship={tx.relationship}")
    if tx.transaction_code:
        parts.append(f"code={tx.transaction_code}")
    if tx.acquired_disposed_code:
        parts.append(f"acquired_disposed={tx.acquired_disposed_code}")
    if tx.transaction_shares is not None:
        parts.append(f"shares={tx.transaction_shares:g}")
    if tx.transaction_price is not None:
        parts.append(f"price={tx.transaction_price:g}")
    if tx.shares_owned_following is not None:
        parts.append(f"owned_following={tx.shares_owned_following:g}")
    parts.append("not automatically a risk conclusion")
    return "; ".join(parts)


def _confidence(source_type: str, source_name: str) -> float:
    combined = f"{source_type} {source_name}".lower()
    if "sec" in combined or "form 4" in combined:
        return 0.85
    return 0.65
