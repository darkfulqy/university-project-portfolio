from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import re
import sqlite3
import xml.etree.ElementTree as ET
from urllib.parse import quote

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


GLOBENEWSWIRE_PRESS_RELEASES_RSS = (
    "https://www.globenewswire.com/RssFeed/subjectcode/72-Press%20Releases/"
    "feedTitle/GlobeNewswire%20-%20Press%20Releases"
)
PR_NEWSWIRE_NEWS_RELEASES_RSS = "https://www.prnewswire.com/rss/news-releases-list.rss"
YAHOO_FINANCE_TICKER_RSS_TEMPLATE = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
)
NASDAQ_TICKER_RSS_TEMPLATE = "https://www.nasdaq.com/feed/rssoutbound?symbol={ticker}"


@dataclass(frozen=True)
class NewsRssSource:
    source_name: str
    url: str
    related_ticker: str | None = None


DEFAULT_NEWS_RSS_SOURCES: tuple[NewsRssSource, ...] = (
    NewsRssSource("GlobeNewswire Press Releases", GLOBENEWSWIRE_PRESS_RELEASES_RSS),
    NewsRssSource("PR Newswire News Releases", PR_NEWSWIRE_NEWS_RELEASES_RSS),
)


@dataclass(frozen=True)
class NewsEvent:
    source_name: str
    title: str
    url: str
    published_at: str | None
    summary: str
    related_ticker: str | None
    content_hash: str


@dataclass(frozen=True)
class StoreNewsResult:
    events_seen: int
    mapped: int


def fetch_news_rss(
    client: HttpClient,
    *,
    url: str = GLOBENEWSWIRE_PRESS_RELEASES_RSS,
    source_name: str = "GlobeNewswire Press Releases",
    limit: int = 50,
    related_ticker: str | None = None,
) -> list[NewsEvent]:
    xml = client.get_text(url, accept="application/rss+xml,application/xml,text/xml")
    return parse_news_rss(xml, source_name=source_name, limit=limit, related_ticker=related_ticker)


def parse_news_rss(
    xml: str,
    *,
    source_name: str,
    limit: int = 50,
    related_ticker: str | None = None,
) -> list[NewsEvent]:
    root = ET.fromstring(xml)
    raw_items = _rss_items(root) or _atom_items(root)
    events: list[NewsEvent] = []
    for item in raw_items[:limit]:
        title = item.get("title", "").strip()
        url = item.get("link", "").strip()
        if not title or not url:
            continue
        summary = _clean_html(item.get("summary", ""))
        published_at = item.get("published_at") or None
        content_hash = hashlib.sha256(
            f"{source_name}|{title}|{url}|{published_at}".encode("utf-8")
        ).hexdigest()
        events.append(
            NewsEvent(
                source_name=source_name,
                title=title,
                url=url,
                published_at=published_at,
                summary=summary,
                related_ticker=related_ticker.upper() if related_ticker else None,
                content_hash=content_hash,
            )
        )
    return events


def ticker_news_rss_sources(ticker: str) -> tuple[NewsRssSource, ...]:
    normalized = ticker.strip().upper()
    encoded = quote(normalized, safe="")
    return (
        NewsRssSource(
            f"Yahoo Finance RSS ({normalized})",
            YAHOO_FINANCE_TICKER_RSS_TEMPLATE.format(ticker=encoded),
            normalized,
        ),
        NewsRssSource(
            f"Nasdaq RSS ({normalized})",
            NASDAQ_TICKER_RSS_TEMPLATE.format(ticker=encoded),
            normalized,
        ),
    )


def store_news_events(
    conn: sqlite3.Connection,
    events: list[NewsEvent],
    *,
    source_type: str = "News RSS",
    content_hash_prefix: str = "news-rss",
    mapped_confidence: float = 0.65,
    unmapped_confidence: float = 0.4,
) -> StoreNewsResult:
    now = utc_now_iso()
    resolved = [_resolve_news_event(conn, event) for event in events]
    conn.executemany(
        """
        INSERT INTO news_events (
            source_name, title, url, published_at, fetched_at,
            summary, related_ticker, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            related_ticker=excluded.related_ticker,
            fetched_at=excluded.fetched_at,
            summary=excluded.summary
        """,
        [
            (
                event.source_name,
                event.title,
                event.url,
                event.published_at,
                now,
                event.summary,
                event.related_ticker,
                event.content_hash,
            )
            for event in resolved
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
                source_type,
                event.source_name,
                event.url,
                event.published_at,
                now,
                event.title,
                event.summary,
                event.summary or event.title,
                mapped_confidence if event.related_ticker else unmapped_confidence,
                event.related_ticker,
                "news_monitor",
                f"{content_hash_prefix}:{event.content_hash}",
            )
            for event in resolved
        ],
    )
    return StoreNewsResult(
        events_seen=len(resolved),
        mapped=sum(1 for event in resolved if event.related_ticker),
    )


def _resolve_news_event(conn: sqlite3.Connection, event: NewsEvent) -> NewsEvent:
    text = f"{event.title} {event.summary}"
    ticker = _preset_ticker(conn, event.related_ticker)
    ticker = ticker or _ticker_from_exchange_pattern(text)
    if ticker and not _known_ticker(conn, ticker):
        ticker = None
    ticker = ticker or _ticker_from_company_names(conn, text)
    return NewsEvent(
        source_name=event.source_name,
        title=event.title,
        url=event.url,
        published_at=event.published_at,
        summary=event.summary,
        related_ticker=ticker,
        content_hash=event.content_hash,
    )


def _preset_ticker(conn: sqlite3.Connection, ticker: str | None) -> str | None:
    if not ticker:
        return None
    normalized = ticker.strip().upper()
    return normalized if normalized and _known_ticker(conn, normalized) else None


def _rss_items(root: ET.Element) -> list[dict[str, str]]:
    items = root.findall(".//item")
    parsed: list[dict[str, str]] = []
    for item in items:
        parsed.append(
            {
                "title": _child_text(item, "title"),
                "link": _child_text(item, "link"),
                "summary": _child_text(item, "description"),
                "published_at": _child_text(item, "pubDate"),
            }
        )
    return parsed


def _atom_items(root: ET.Element) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        link = ""
        link_element = entry.find("{http://www.w3.org/2005/Atom}link")
        if link_element is not None:
            link = link_element.attrib.get("href", "")
        parsed.append(
            {
                "title": _child_text(entry, "{http://www.w3.org/2005/Atom}title"),
                "link": link,
                "summary": _child_text(entry, "{http://www.w3.org/2005/Atom}summary"),
                "published_at": _child_text(entry, "{http://www.w3.org/2005/Atom}updated"),
            }
        )
    return parsed


def _child_text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    return child.text.strip() if child is not None and child.text else ""


def _ticker_from_exchange_pattern(text: str) -> str | None:
    match = re.search(
        r"\b(?:NASDAQ|Nasdaq|NYSE|NYSE American|NYSEAMERICAN|OTC|OTCQB|OTCQX|TSX|CSE)\s*:\s*([A-Z][A-Z0-9.\-]{0,9})\b",
        text,
    )
    if match:
        return match.group(1).replace(".", "-").upper()
    return None


def _ticker_from_company_names(conn: sqlite3.Connection, text: str) -> str | None:
    normalized_text = _normalize_name(text)
    rows = _company_name_candidates(conn)
    matches: list[tuple[int, str]] = []
    for row in rows:
        name = _normalize_name(row["company_name"])
        if len(name) < 8 and len(name.split()) < 2:
            continue
        if _normalized_name_in_text(name, normalized_text):
            matches.append((len(name), row["ticker"]))
    if not matches:
        return None
    best_length = max(length for length, _ticker in matches)
    best_tickers = {ticker.upper() for length, ticker in matches if length == best_length}
    return next(iter(best_tickers)) if len(best_tickers) == 1 else None


def _known_ticker(conn: sqlite3.Connection, ticker: str) -> bool:
    normalized = ticker.upper()
    row = conn.execute(
        """
        SELECT 1 FROM universe
        WHERE ticker = ? AND is_etf = 0 AND is_preferred = 0 AND is_unit = 0 AND is_active = 1
        UNION
        SELECT 1 FROM company_profile
        WHERE ticker = ?
          AND (? = 1 OR COALESCE(source, '') != 'SEC company_tickers')
        LIMIT 1
        """,
        (normalized, normalized, 1 if _clean_universe_count(conn) == 0 else 0),
    ).fetchone()
    return row is not None


def _company_name_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    profile_where = "company_name IS NOT NULL AND company_name != ''"
    if _clean_universe_count(conn) > 0:
        profile_where += " AND COALESCE(source, '') != 'SEC company_tickers'"
    return conn.execute(
        f"""
        SELECT ticker, company_name FROM universe
        WHERE company_name IS NOT NULL AND company_name != ''
          AND is_etf = 0 AND is_preferred = 0 AND is_unit = 0 AND is_active = 1
        UNION
        SELECT ticker, company_name FROM company_profile
        WHERE {profile_where}
        """
    ).fetchall()


def _clean_universe_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count FROM universe
        WHERE is_etf = 0 AND is_preferred = 0 AND is_unit = 0 AND is_active = 1
        """
    ).fetchone()
    return int(row["count"] or 0)


def _normalize_name(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(
        r"\b(incorporated|inc|corp|corporation|company|co|ltd|plc|class a|common stock)\b",
        " ",
        lowered,
    )
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _normalized_name_in_text(name: str, normalized_text: str) -> bool:
    if not name:
        return False
    pattern = r"(?:^|\s)" + re.escape(name) + r"(?:\s|$)"
    return re.search(pattern, normalized_text) is not None


def _clean_html(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html or "")
    return re.sub(r"\s+", " ", parser.text()).strip()


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def text(self) -> str:
        return " ".join(self._parts)
