from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import sqlite3
from urllib.parse import urljoin, urlparse

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class IrPageCheck:
    ticker: str
    url: str
    title: str
    content_hash: str
    changed: bool


@dataclass(frozen=True)
class IrPageCheckError:
    ticker: str
    url: str
    error: str


@dataclass(frozen=True)
class IrBatchCheckResult:
    checked: int
    changed: int
    errors: list[IrPageCheckError]


@dataclass(frozen=True)
class IrDiscovery:
    ticker: str
    website: str
    discovered_url: str | None
    title: str | None
    status: str
    reason: str


@dataclass(frozen=True)
class IrDiscoveryResult:
    candidates: int
    discovered: int
    checked: int
    errors: list[IrDiscovery]


def upsert_ir_url(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    url: str,
    page_type: str = "ir_home",
    source: str = "manual",
    notes: str | None = None,
) -> None:
    ticker = ticker.upper()
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO company_profile (ticker, ir_url, source, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            ir_url=excluded.ir_url,
            source=CASE
                WHEN company_profile.source IS NULL OR company_profile.source = ''
                THEN excluded.source
                ELSE company_profile.source
            END,
            updated_at=excluded.updated_at
        """,
        (ticker, url, source, now),
    )
    conn.execute(
        """
        INSERT INTO ir_pages (ticker, url, page_type, source, status, notes)
        VALUES (?, ?, ?, ?, 'active', ?)
        ON CONFLICT(ticker, url) DO UPDATE SET
            page_type=excluded.page_type,
            source=excluded.source,
            status='active',
            notes=excluded.notes
        """,
        (ticker, url, page_type, source, notes),
    )


def import_ir_urls(conn: sqlite3.Connection, csv_path: Path) -> int:
    count = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = (row.get("ticker") or row.get("Ticker") or "").strip()
            url = (row.get("ir_url") or row.get("url") or row.get("IR URL") or "").strip()
            if not ticker or not url:
                continue
            upsert_ir_url(
                conn,
                ticker=ticker,
                url=url,
                page_type=(row.get("page_type") or "ir_home").strip() or "ir_home",
                source=(row.get("source") or "csv").strip() or "csv",
                notes=(row.get("notes") or "").strip() or None,
            )
            count += 1
    return count


def get_ir_url(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        """
        SELECT ir_url FROM company_profile
        WHERE ticker = ? AND ir_url IS NOT NULL AND ir_url != ''
        """,
        (ticker.upper(),),
    ).fetchone()
    return row["ir_url"] if row else None


def list_active_ir_pages(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    params: list[object] = []
    ticker_filter = ""
    if ticker:
        ticker_filter = "AND ticker = ?"
        params.append(ticker.upper())
    params.append(limit)
    return conn.execute(
        f"""
        SELECT ticker, url, page_type, last_checked_at
        FROM ir_pages
        WHERE status = 'active'
          AND url IS NOT NULL
          AND url != ''
          {ticker_filter}
        ORDER BY
          CASE WHEN last_checked_at IS NULL THEN 0 ELSE 1 END,
          last_checked_at,
          ticker,
          url
        LIMIT ?
        """,
        params,
    ).fetchall()


def check_ir_pages(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    ticker: str | None = None,
    limit: int = 50,
) -> IrBatchCheckResult:
    rows = list_active_ir_pages(conn, ticker=ticker, limit=limit)
    checked = 0
    changed = 0
    errors: list[IrPageCheckError] = []
    for row in rows:
        page_ticker = row["ticker"]
        page_url = row["url"]
        try:
            result = check_ir_page(conn, client, ticker=page_ticker, url=page_url)
            checked += 1
            if result.changed:
                changed += 1
        except Exception as exc:
            message = str(exc)
            errors.append(IrPageCheckError(ticker=page_ticker, url=page_url, error=message))
            conn.execute(
                """
                UPDATE ir_pages
                SET last_checked_at = ?, last_error = ?
                WHERE ticker = ? AND url = ?
                """,
                (utc_now_iso(), message, page_ticker, page_url),
            )
    return IrBatchCheckResult(checked=checked, changed=changed, errors=errors)


def check_ir_page(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    ticker: str,
    url: str,
) -> IrPageCheck:
    ticker = ticker.upper()
    text = client.get_text(url, accept="text/html,application/xhtml+xml,text/plain")
    return _store_checked_ir_page(conn, ticker=ticker, url=url, text=text, source="manual")


def discover_ir_pages_from_profiles(
    conn: sqlite3.Connection,
    client: HttpClient,
    *,
    tickers: list[str] | None = None,
    limit: int = 25,
    include_existing: bool = False,
    max_candidate_urls: int = 8,
) -> IrDiscoveryResult:
    rows = _ir_discovery_profile_candidates(
        conn,
        tickers=tickers,
        limit=limit,
        include_existing=include_existing,
    )
    discoveries: list[IrDiscovery] = []
    checked = 0
    for row in rows:
        ticker = row["ticker"]
        website = row["website"]
        try:
            urls = _candidate_ir_urls(client, website, max_candidate_urls=max_candidate_urls)
        except Exception as exc:
            discoveries.append(
                IrDiscovery(
                    ticker=ticker,
                    website=website,
                    discovered_url=None,
                    title=None,
                    status="error",
                    reason=f"Could not inspect company website: {exc}",
                )
            )
            continue
        if not urls:
            discoveries.append(
                IrDiscovery(
                    ticker=ticker,
                    website=website,
                    discovered_url=None,
                    title=None,
                    status="not_found",
                    reason="No investor-relations link candidate found on company website or common IR paths.",
                )
            )
            continue
        discovered = False
        for candidate_url in urls:
            try:
                text = client.get_text(candidate_url, accept="text/html,application/xhtml+xml,text/plain")
                checked += 1
                title = _extract_title(text) or candidate_url
                if not _looks_like_ir_page(candidate_url, title, text):
                    continue
                upsert_ir_url(
                    conn,
                    ticker=ticker,
                    url=candidate_url,
                    page_type="ir_home",
                    source="Company website IR discovery",
                    notes=f"Discovered from source-backed company website {website}.",
                )
                _store_checked_ir_page(
                    conn,
                    ticker=ticker,
                    url=candidate_url,
                    text=text,
                    source="Company website IR discovery",
                )
                discoveries.append(
                    IrDiscovery(
                        ticker=ticker,
                        website=website,
                        discovered_url=candidate_url,
                        title=title,
                        status="discovered",
                        reason="Matched investor-relations URL/title/content on source-backed company website.",
                    )
                )
                discovered = True
                break
            except Exception as exc:
                discoveries.append(
                    IrDiscovery(
                        ticker=ticker,
                        website=website,
                        discovered_url=candidate_url,
                        title=None,
                        status="error",
                        reason=str(exc),
                    )
                )
        if not discovered and not any(item.ticker == ticker and item.status == "error" for item in discoveries):
            discoveries.append(
                IrDiscovery(
                    ticker=ticker,
                    website=website,
                    discovered_url=None,
                    title=None,
                    status="not_found",
                    reason="Candidate URLs were reachable or inspectable but did not pass conservative IR checks.",
                )
            )
    return IrDiscoveryResult(
        candidates=len(rows),
        discovered=sum(1 for item in discoveries if item.status == "discovered"),
        checked=checked,
        errors=[item for item in discoveries if item.status == "error"],
    )


def _store_checked_ir_page(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    url: str,
    text: str,
    source: str,
) -> IrPageCheck:
    title = _extract_title(text) or url
    content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    old = conn.execute(
        "SELECT content_hash FROM ir_pages WHERE ticker = ? AND url = ?",
        (ticker, url),
    ).fetchone()
    changed = bool(old and old["content_hash"] and old["content_hash"] != content_hash)
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO ir_pages (
            ticker, url, page_type, source, status, last_title, content_hash,
            last_checked_at, last_changed_at
        )
        VALUES (?, ?, 'ir_home', ?, 'active', ?, ?, ?, ?)
        ON CONFLICT(ticker, url) DO UPDATE SET
            status='active',
            last_title=excluded.last_title,
            content_hash=excluded.content_hash,
            last_checked_at=excluded.last_checked_at,
            last_error=NULL,
            last_changed_at=CASE
                WHEN ir_pages.content_hash IS NULL OR ir_pages.content_hash != excluded.content_hash
                THEN excluded.last_changed_at
                ELSE ir_pages.last_changed_at
            END
        """,
        (ticker, url, source, title, content_hash, now, now),
    )
    _store_ir_evidence(conn, ticker=ticker, url=url, title=title, content_hash=content_hash)
    return IrPageCheck(
        ticker=ticker,
        url=url,
        title=title,
        content_hash=content_hash,
        changed=changed,
    )


def _ir_discovery_profile_candidates(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None,
    limit: int,
    include_existing: bool,
) -> list[sqlite3.Row]:
    params: list[object] = []
    filters = [
        "website IS NOT NULL",
        "website != ''",
        "COALESCE(source, '') != ''",
        "COALESCE(source, '') != 'SEC company_tickers'",
        """
        NOT EXISTS (
            SELECT 1 FROM universe_exclusions AS exclusion
            WHERE exclusion.ticker = company_profile.ticker
              AND exclusion.is_active = 1
              AND exclusion.severity = 'exclude'
        )
        """,
    ]
    if tickers:
        normalized = [ticker.upper() for ticker in tickers]
        placeholders = ",".join("?" for _ in normalized)
        filters.append(f"ticker IN ({placeholders})")
        params.extend(normalized)
    if not include_existing:
        filters.append(
            """
            NOT EXISTS (
                SELECT 1 FROM ir_pages
                WHERE ir_pages.ticker = company_profile.ticker
                  AND ir_pages.status = 'active'
            )
            """
        )
    params.append(limit)
    return conn.execute(
        f"""
        SELECT ticker, company_name, website
        FROM company_profile
        WHERE {' AND '.join(filters)}
        ORDER BY
          CASE WHEN ticker IN (
            SELECT related_ticker
            FROM evidence_items
            WHERE related_module = 'external_research_monitor'
          ) THEN 0 ELSE 1 END,
          ticker
        LIMIT ?
        """,
        params,
    ).fetchall()


def _candidate_ir_urls(
    client: HttpClient,
    website: str,
    *,
    max_candidate_urls: int,
) -> list[str]:
    homepage_url = _normalize_website_url(website)
    html = client.get_text(homepage_url, accept="text/html,application/xhtml+xml,text/plain")
    parser = _LinkParser()
    parser.feed(html)
    candidates: list[str] = []
    for href, text in parser.links:
        resolved = _normalize_candidate_url(homepage_url, href)
        if not resolved:
            continue
        if _looks_like_ir_link(resolved, text):
            _append_unique(candidates, resolved)
    for common_url in _common_ir_urls(homepage_url):
        _append_unique(candidates, common_url)
    return candidates[:max_candidate_urls]


def _normalize_website_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        return stripped
    if not stripped.startswith(("http://", "https://")):
        stripped = "https://" + stripped
    return stripped.rstrip("/")


def _normalize_candidate_url(homepage_url: str, href: str) -> str | None:
    raw = (href or "").strip()
    if not raw or raw.startswith(("#", "mailto:", "tel:", "javascript:")):
        return None
    resolved = urljoin(homepage_url + "/", raw)
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if _is_social_or_document_url(resolved):
        return None
    return resolved.rstrip("/")


def _looks_like_ir_link(url: str, text: str) -> bool:
    haystack = f"{url} {text}".lower()
    negative_terms = ("career", "privacy", "cookie", "supplier", "support", "community")
    if any(term in haystack for term in negative_terms):
        return False
    patterns = (
        "investor relations",
        "investors",
        "/investor",
        "investor-relations",
        "shareholder",
        "stockholder",
        "financial information",
        "sec filings",
        "annual report",
    )
    return any(pattern in haystack for pattern in patterns)


def _looks_like_ir_page(url: str, title: str, html: str) -> bool:
    snippet = html[:5000].lower()
    haystack = f"{url} {title} {snippet}".lower()
    strong_terms = (
        "investor relations",
        "investors",
        "shareholder",
        "stockholder",
        "sec filings",
        "annual report",
        "quarterly results",
        "financial results",
    )
    if any(term in haystack for term in strong_terms):
        return True
    parsed = urlparse(url)
    return parsed.netloc.lower().startswith(("ir.", "investor.", "investors.")) and "invest" in haystack


def _common_ir_urls(homepage_url: str) -> list[str]:
    parsed = urlparse(homepage_url)
    root = _root_domain(parsed.netloc)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls = [
        f"https://investors.{root}",
        f"https://investor.{root}",
        f"https://ir.{root}",
        urljoin(base + "/", "investors"),
        urljoin(base + "/", "investor-relations"),
        urljoin(base + "/", "investor"),
        urljoin(base + "/", "company/investor-relations"),
    ]
    unique: list[str] = []
    for url in urls:
        _append_unique(unique, url.rstrip("/"))
    return unique


def _root_domain(netloc: str) -> str:
    hostname = netloc.split("@")[-1].split(":")[0].lower()
    parts = [part for part in hostname.split(".") if part]
    if len(parts) <= 2:
        return hostname
    if parts[-2] in {"co", "com", "org", "net"} and len(parts[-1]) == 2 and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _is_social_or_document_url(url: str) -> bool:
    lowered = url.lower()
    blocked_hosts = ("x.com", "twitter.com", "linkedin.com", "facebook.com", "youtube.com")
    if any(host in lowered for host in blocked_hosts):
        return True
    return lowered.endswith((".pdf", ".jpg", ".jpeg", ".png", ".svg", ".zip"))


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _store_ir_evidence(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    url: str,
    title: str,
    content_hash: str,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT OR IGNORE INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at, raw_title, summary,
            evidence_snippet, confidence, related_ticker, related_module, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Company IR",
            "Company IR page",
            url,
            None,
            now,
            title,
            "IR page hash check. Full page content is not stored.",
            f"IR page observed with title: {title}",
            0.7,
            ticker,
            "ir_monitor",
            f"ir:{content_hash}",
        ),
    )


def _extract_title(html: str) -> str:
    parser = _TitleParser()
    parser.feed(html)
    title = parser.title.strip()
    if title:
        return re.sub(r"\s+", " ", title)
    return ""


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(self._parts)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        self._current_href = attrs_dict.get("href")
        self._current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._current_href:
            return
        text = re.sub(r"\s+", " ", " ".join(self._current_text)).strip()
        self.links.append((self._current_href, text))
        self._current_href = None
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)
