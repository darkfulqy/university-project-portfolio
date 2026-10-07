from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


LOCAL_RESEARCH_SOURCE_TYPE = "local_external_research_markdown"
LOCAL_RESEARCH_SOURCE_NAME = "Local external research markdown"
LOCAL_RESEARCH_RELATED_MODULE = "external_research_monitor"

_URL_RE = re.compile(r"https?://\S+")
_TICKER_RE = re.compile(r"(?<![@A-Za-z0-9_$])\$?([A-Z][A-Z0-9]{1,5}(?:[.-][A-Z0-9]{1,3})?)(?![A-Za-z0-9_])")
_HEADING_RE = re.compile(r"^#{2,4}\s+(.+?)\s*$")
_LOCAL_FILE_SOURCE_NAME = "local_markdown_file"
_EXCLUDED_TOKENS = {
    "AI",
    "AGI",
    "API",
    "ARM64",
    "B200",
    "CDN",
    "CSV",
    "CPU",
    "CUDA",
    "DRAM",
    "EOD",
    "ETF",
    "FMP",
    "FOMO",
    "GPU",
    "HBM",
    "HK",
    "H100",
    "IR",
    "RSS",
    "SEC",
    "SFO",
    "SPX",
    "SPY",
    "TSX",
    "URL",
    "US",
    "USD",
    "X",
}


@dataclass(frozen=True)
class LocalResearchLead:
    source_name: str
    title: str
    source_url: str
    published_at: str | None
    summary: str
    evidence_snippet: str
    related_ticker: str
    confidence: float
    content_hash: str


@dataclass(frozen=True)
class StoreLocalResearchResult:
    leads_seen: int
    mapped_ticker_leads: int
    distinct_tickers: int


def import_local_research_markdown(
    conn: sqlite3.Connection,
    path: Path,
    *,
    source_name: str = LOCAL_RESEARCH_SOURCE_NAME,
    limit: int = 500,
) -> StoreLocalResearchResult:
    text = path.read_text(encoding="utf-8-sig")
    leads = parse_local_research_markdown(
        text,
        path=path,
        known_tickers=_known_tickers(conn),
        source_name=source_name,
        limit=limit,
    )
    return store_local_research_leads(conn, leads)


def parse_local_research_markdown(
    markdown: str,
    *,
    path: Path,
    known_tickers: set[str],
    source_name: str = LOCAL_RESEARCH_SOURCE_NAME,
    limit: int = 500,
) -> list[LocalResearchLead]:
    leads: list[LocalResearchLead] = []
    mode: str | None = None
    current_heading = "document"
    current_account: str | None = None
    published_at = _date_from_path_or_text(path, markdown)

    for line_number, raw_line in enumerate(markdown.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            current_account = current_heading if current_heading.startswith("@") else None
            mode = None
            continue
        lowered = line.lower()
        if lowered == "representative links:":
            mode = "representative_links"
            continue
        if lowered == "watchlist extracted:":
            mode = "watchlist"
            continue
        if line.endswith(":"):
            mode = None
            continue
        if lowered.startswith("suggested monitoring queries") or lowered.startswith("monitoring queries"):
            mode = None
            continue
        if not line.startswith("- "):
            continue

        if mode == "representative_links":
            leads.extend(
                _leads_from_representative_link(
                    line,
                    source_name=source_name,
                    heading=current_heading,
                    account=current_account,
                    published_at=published_at,
                    known_tickers=known_tickers,
                )
            )
        elif mode == "watchlist":
            leads.extend(
                _leads_from_watchlist_line(
                    line,
                    source_name=source_name,
                    path=path,
                    line_number=line_number,
                    heading=current_heading,
                    account=current_account,
                    published_at=published_at,
                    known_tickers=known_tickers,
                )
            )
        if len(leads) >= limit:
            return _dedupe_leads(leads)[:limit]
    return _dedupe_leads(leads)[:limit]


def store_local_research_leads(
    conn: sqlite3.Connection,
    leads: list[LocalResearchLead],
) -> StoreLocalResearchResult:
    now = utc_now_iso()
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
                LOCAL_RESEARCH_SOURCE_TYPE,
                lead.source_name,
                lead.source_url,
                lead.published_at,
                now,
                lead.title,
                lead.summary,
                lead.evidence_snippet,
                lead.confidence,
                lead.related_ticker,
                LOCAL_RESEARCH_RELATED_MODULE,
                lead.content_hash,
            )
            for lead in leads
        ],
    )
    return StoreLocalResearchResult(
        leads_seen=len(leads),
        mapped_ticker_leads=len(leads),
        distinct_tickers=len({lead.related_ticker for lead in leads}),
    )


def _leads_from_representative_link(
    line: str,
    *,
    source_name: str,
    heading: str,
    account: str | None,
    published_at: str | None,
    known_tickers: set[str],
) -> list[LocalResearchLead]:
    urls = _URL_RE.findall(line)
    if not urls:
        return []
    url = urls[0].rstrip(").,")
    label = line[2:].replace(url, "").strip(" :-")
    tickers = _extract_known_tickers(label, known_tickers)
    leads = []
    for ticker in tickers:
        title = f"External research link lead for {ticker}: {label or heading}"
        summary = _lead_summary(account, heading, line[2:], "representative_link")
        leads.append(
            _lead(
                source_name=source_name,
                title=title,
                source_url=url,
                published_at=published_at,
                summary=summary,
                related_ticker=ticker,
                confidence=0.35,
            )
        )
    return leads


def _leads_from_watchlist_line(
    line: str,
    *,
    source_name: str,
    path: Path,
    line_number: int,
    heading: str,
    account: str | None,
    published_at: str | None,
    known_tickers: set[str],
) -> list[LocalResearchLead]:
    text = line[2:].strip()
    tickers = _extract_known_tickers(text, known_tickers)
    source_url = _local_source_url(path, line_number)
    leads = []
    for ticker in tickers:
        title = f"External research watchlist lead for {ticker}: {heading}"
        summary = _lead_summary(account, heading, text, "watchlist")
        leads.append(
            _lead(
                source_name=source_name,
                title=title,
                source_url=source_url,
                published_at=published_at,
                summary=summary,
                related_ticker=ticker,
                confidence=0.25,
            )
        )
    return leads


def _lead(
    *,
    source_name: str,
    title: str,
    source_url: str,
    published_at: str | None,
    summary: str,
    related_ticker: str,
    confidence: float,
) -> LocalResearchLead:
    snippet = (
        summary
        + " This is an external research lead only; verify with SEC filings, company IR, "
        "API data, or original news before using it for any company-level conclusion."
    )
    content_hash = hashlib.sha256(
        f"{LOCAL_RESEARCH_SOURCE_TYPE}|{source_name}|{source_url}|{related_ticker}|{title}".encode("utf-8")
    ).hexdigest()
    return LocalResearchLead(
        source_name=source_name,
        title=title,
        source_url=source_url,
        published_at=published_at,
        summary=summary,
        evidence_snippet=snippet,
        related_ticker=related_ticker,
        confidence=confidence,
        content_hash=f"local-research:{content_hash}",
    )


def _lead_summary(account: str | None, heading: str, text: str, lead_type: str) -> str:
    account_text = f"{account} " if account else ""
    return (
        f"{account_text}{lead_type} line from local research section '{heading}': "
        f"{_compact_text(text, 260)}"
    )


def _extract_known_tickers(text: str, known_tickers: set[str]) -> list[str]:
    tickers: list[str] = []
    for match in _TICKER_RE.finditer(text):
        ticker = match.group(1).replace(".", "-").upper()
        if ticker in _EXCLUDED_TOKENS:
            continue
        if ticker not in known_tickers:
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _known_tickers(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT ticker, company_name, 1 AS universe_backed
        FROM universe
        WHERE is_etf = 0 AND is_preferred = 0 AND is_unit = 0 AND is_active = 1
        UNION
        SELECT ticker, company_name, 0 AS universe_backed
        FROM company_profile
        WHERE ticker IS NOT NULL AND ticker != ''
        """
    ).fetchall()
    tickers: set[str] = set()
    for row in rows:
        ticker = str(row["ticker"] or "").replace(".", "-").upper()
        if not ticker or ticker in _EXCLUDED_TOKENS:
            continue
        if _looks_like_non_operating_security(str(row["company_name"] or "")):
            continue
        tickers.add(ticker)
    return tickers


def _looks_like_non_operating_security(company_name: str) -> bool:
    lowered = company_name.lower()
    patterns = (
        " acquisition ",
        " spac",
        " etf",
        " exchange traded",
        " fund",
        " trust",
        " warrant",
        " unit",
        " rights",
        " notes due",
        " preferred",
    )
    padded = f" {lowered} "
    return any(pattern in padded for pattern in patterns)


def _date_from_path_or_text(path: Path, markdown: str) -> str | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name) or re.search(r"(20\d{2}-\d{2}-\d{2})", markdown[:300])
    return match.group(1) if match else None


def _local_source_url(path: Path, line_number: int) -> str:
    normalized = str(path.resolve()).replace("\\", "/")
    return f"{normalized}#line={line_number}"


def _compact_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _dedupe_leads(leads: list[LocalResearchLead]) -> list[LocalResearchLead]:
    seen: set[str] = set()
    deduped: list[LocalResearchLead] = []
    for lead in leads:
        if lead.content_hash in seen:
            continue
        seen.add(lead.content_hash)
        deduped.append(lead)
    return deduped
