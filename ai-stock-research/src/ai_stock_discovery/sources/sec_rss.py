from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3
import xml.etree.ElementTree as ET

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


SEC_CURRENT_ATOM_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&owner=include&count={count}&output=atom"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class SecRssEvent:
    title: str
    form: str | None
    cik: str | None
    ticker: str | None
    accession_number: str | None
    company_name: str | None
    filing_url: str
    published_at: str | None
    summary: str | None
    content_hash: str


@dataclass(frozen=True)
class StoreRssResult:
    events_seen: int
    queued: int


def fetch_current_filings(client: HttpClient, *, count: int = 40) -> list[SecRssEvent]:
    xml = client.get_text(SEC_CURRENT_ATOM_URL.format(count=count), accept="application/atom+xml,application/xml,text/xml")
    return parse_current_filings(xml)


def parse_current_filings(xml: str) -> list[SecRssEvent]:
    root = ET.fromstring(xml)
    events: list[SecRssEvent] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _text(entry, "atom:title") or ""
        updated = _text(entry, "atom:updated")
        summary = _text(entry, "atom:summary")
        form = _category_term(entry) or _form_from_title(title)
        link = _link_href(entry)
        cik = _cik_from_text(" ".join(part for part in [title, summary or "", link or ""] if part))
        accession_number = _accession_from_url(link or "")
        company_name = _company_from_title(title)
        if not title or not link:
            continue
        content_hash = hashlib.sha256(f"{title}|{updated}|{link}".encode("utf-8")).hexdigest()
        events.append(
            SecRssEvent(
                title=title,
                form=form,
                cik=cik,
                ticker=None,
                accession_number=accession_number,
                company_name=company_name,
                filing_url=link,
                published_at=updated,
                summary=summary,
                content_hash=content_hash,
            )
        )
    return events


def store_rss_events(
    conn: sqlite3.Connection,
    events: list[SecRssEvent],
    *,
    forms: set[str] | None = None,
) -> StoreRssResult:
    filtered = [event for event in events if not forms or (event.form or "").upper() in forms]
    now = utc_now_iso()
    resolved = [_resolve_event(conn, event) for event in filtered]
    conn.executemany(
        """
        INSERT INTO rss_filing_events (
            title, form, ticker, cik, accession_number, company_name, filing_url,
            published_at, fetched_at, summary, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            ticker=COALESCE(excluded.ticker, rss_filing_events.ticker),
            cik=COALESCE(excluded.cik, rss_filing_events.cik),
            accession_number=COALESCE(excluded.accession_number, rss_filing_events.accession_number),
            form=COALESCE(excluded.form, rss_filing_events.form),
            company_name=COALESCE(excluded.company_name, rss_filing_events.company_name),
            fetched_at=excluded.fetched_at
        """,
        [
            (
                event.title,
                event.form,
                event.ticker,
                event.cik,
                event.accession_number,
                event.company_name,
                event.filing_url,
                event.published_at,
                now,
                event.summary,
                event.content_hash,
            )
            for event in resolved
        ],
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at, raw_title, summary,
            evidence_snippet, confidence, related_ticker, related_module, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "SEC RSS",
                "SEC latest filings Atom feed",
                event.filing_url,
                event.published_at,
                now,
                event.title,
                event.summary,
                event.summary or event.title,
                0.9,
                event.ticker,
                "filing_monitor",
                f"sec-rss:{event.content_hash}",
            )
            for event in resolved
        ],
    )
    queue_rows = _queue_rows(conn, resolved, now)
    conn.executemany(
        """
        INSERT OR IGNORE INTO filing_queue (
            event_id, ticker, cik, form, accession_number, filing_url, document_url,
            source, status, queued_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
        """,
        queue_rows,
    )
    return StoreRssResult(events_seen=len(filtered), queued=len(queue_rows))


def _resolve_event(conn: sqlite3.Connection, event: SecRssEvent) -> SecRssEvent:
    if event.ticker or not event.cik:
        return event
    row = conn.execute(
        """
        SELECT ticker FROM company_profile
        WHERE cik = ?
        ORDER BY ticker
        LIMIT 1
        """,
        (event.cik,),
    ).fetchone()
    ticker = row["ticker"] if row else None
    return SecRssEvent(
        title=event.title,
        form=event.form,
        cik=event.cik,
        ticker=ticker,
        accession_number=event.accession_number,
        company_name=event.company_name,
        filing_url=event.filing_url,
        published_at=event.published_at,
        summary=event.summary,
        content_hash=event.content_hash,
    )


def _queue_rows(
    conn: sqlite3.Connection,
    events: list[SecRssEvent],
    queued_at: str,
) -> list[tuple[int | None, str | None, str | None, str | None, str | None, str, None, str, str]]:
    rows = []
    for event in events:
        if not event.ticker or not event.cik or not event.form:
            continue
        event_row = conn.execute(
            "SELECT id FROM rss_filing_events WHERE content_hash = ?",
            (event.content_hash,),
        ).fetchone()
        event_id = int(event_row["id"]) if event_row else None
        rows.append(
            (
                event_id,
                event.ticker,
                event.cik,
                event.form,
                event.accession_number,
                event.filing_url,
                None,
                "SEC RSS",
                queued_at,
            )
        )
    return rows


def _text(entry: ET.Element, name: str) -> str | None:
    found = entry.find(name, ATOM_NS)
    return found.text.strip() if found is not None and found.text else None


def _link_href(entry: ET.Element) -> str | None:
    link = entry.find("atom:link", ATOM_NS)
    if link is None:
        return None
    href = link.attrib.get("href")
    return href.strip() if href else None


def _category_term(entry: ET.Element) -> str | None:
    category = entry.find("atom:category", ATOM_NS)
    if category is None:
        return None
    term = category.attrib.get("term")
    return term.strip().upper() if term else None


def _form_from_title(title: str) -> str | None:
    match = re.match(r"\s*([A-Z0-9-]+)\s*[-:]", title)
    return match.group(1).upper() if match else None


def _cik_from_text(text: str) -> str | None:
    match = re.search(r"\bCIK[:= ]+0*([0-9]{1,10})\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).zfill(10)
    match = re.search(r"\(([0-9]{10})\)", text)
    return match.group(1) if match else None


def _accession_from_url(url: str) -> str | None:
    match = re.search(r"/Archives/edgar/data/\d+/([0-9]{18})/", url)
    if match:
        raw = match.group(1)
        return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"
    match = re.search(r"\b([0-9]{10}-[0-9]{2}-[0-9]{6})\b", url)
    return match.group(1) if match else None


def _company_from_title(title: str) -> str | None:
    if " - " not in title:
        return None
    parts = title.split(" - ", 1)
    company = parts[1].strip()
    company = re.sub(r"\s*\([0-9]{10}\).*$", "", company).strip()
    return company or None
