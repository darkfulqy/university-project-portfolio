from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


CORE_AI_KEYWORDS = [
    "artificial intelligence",
    "generative ai",
    "genai",
    "machine learning",
    "large language model",
    "llm",
    "ai model",
    "ai workload",
    "accelerated computing",
]

AI_SUPPLY_CHAIN_KEYWORDS = [
    "gpu",
    "data center",
    "datacenter",
    "liquid cooling",
    "advanced packaging",
    "hbm",
    "infiniband",
    "ethernet fabric",
    "optical module",
    "transformer",
    "substation",
    "ups",
]

BUSINESS_IMPACT_TERMS = [
    "revenue",
    "sales",
    "order",
    "orders",
    "backlog",
    "rpo",
    "remaining performance obligation",
    "guidance",
    "margin",
    "gross margin",
    "contract",
    "capex",
    "cash flow",
    "profit",
    "earnings",
    "bookings",
    "customer demand",
    "customer adoption",
    "customer win",
    "customer wins",
    "large customer",
    "hyperscale customer",
]

KEYWORDS = CORE_AI_KEYWORDS + AI_SUPPLY_CHAIN_KEYWORDS

LOCAL_AI_CONTEXT_SOURCE_NAME = "Local company profile AI context scan"

TAG_SIGNAL_SOURCE_TYPE = "local_ai_industry_tag_context"

TAG_SIGNAL_KEYWORD_PREFIX = "ai_industry_tag:"

TAG_SIGNAL_LEVEL = 1

TAG_SIGNAL_MAX_CONFIDENCE = 0.55

RISK_CONTEXT_TERMS = [
    "risk",
    "risks",
    "cyberattack",
    "cyberattacks",
    "phishing",
    "vulnerabilities",
    "vulnerability",
    "threat",
    "threats",
    "harmful",
    "litigation",
    "regulatory",
    "security incident",
]


@dataclass(frozen=True)
class KeywordSignal:
    ticker: str
    signal_date: str | None
    source_type: str
    source_url: str
    keyword: str
    context_snippet: str
    ai_relevance_level: int
    confidence: float


@dataclass(frozen=True)
class LocalAiContextScanSummary:
    tickers_considered: int
    profiles_scanned: int
    profile_signals: int
    tag_rows_scanned: int
    tag_signals: int
    signals_written: int


def scan_text(
    *,
    ticker: str,
    text: str,
    source_type: str,
    source_url: str,
    signal_date: str | None = None,
    max_hits_per_keyword: int = 5,
) -> list[KeywordSignal]:
    normalized = re.sub(r"\s+", " ", text)
    signals: list[KeywordSignal] = []
    for keyword in KEYWORDS:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        hits = 0
        for match in pattern.finditer(normalized):
            snippet = _snippet(normalized, match.start(), match.end())
            if keyword in AI_SUPPLY_CHAIN_KEYWORDS and not _has_ai_context(snippet, keyword):
                continue
            level, confidence = _estimate_level(snippet)
            signals.append(
                KeywordSignal(
                    ticker=ticker.upper(),
                    signal_date=signal_date,
                    source_type=source_type,
                    source_url=source_url,
                    keyword=keyword,
                    context_snippet=snippet,
                    ai_relevance_level=level,
                    confidence=confidence,
                )
            )
            hits += 1
            if hits >= max_hits_per_keyword:
                break
    return signals


def scan_local_ai_context(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 500,
) -> LocalAiContextScanSummary:
    limit = max(0, limit)
    candidate_tickers = _candidate_tickers(conn, tickers=tickers, limit=limit)
    profile_signals: list[KeywordSignal] = []
    tag_signals: list[KeywordSignal] = []
    profiles_scanned = 0
    tag_rows_scanned = 0

    for ticker in candidate_tickers:
        profile = conn.execute(
            """
            SELECT sector, industry, description, source
            FROM company_profile
            WHERE ticker = ?
              AND COALESCE(source, '') <> ''
              AND COALESCE(description, '') <> ''
            """,
            (ticker,),
        ).fetchone()
        if profile:
            text = " ".join(
                part
                for part in (
                    profile["sector"] or "",
                    profile["industry"] or "",
                    profile["description"] or "",
                )
                if str(part).strip()
            )
            if text:
                profiles_scanned += 1
                profile_signals.extend(
                    scan_text(
                        ticker=ticker,
                        text=text,
                        source_type="company_profile",
                        source_url=profile["source"],
                        max_hits_per_keyword=2,
                    )
                )

        tag_rows = conn.execute(
            """
            SELECT tag, confidence, source_type, source_url, evidence_snippet
            FROM ai_industry_tags
            WHERE ticker = ?
              AND COALESCE(source_url, '') <> ''
              AND COALESCE(evidence_snippet, '') <> ''
            ORDER BY confidence DESC, tag
            LIMIT 20
            """,
            (ticker,),
        ).fetchall()
        tag_rows_scanned += len(tag_rows)
        for row in tag_rows:
            tag_signals.append(_signal_from_ai_tag(ticker, row))

    signals_written = store_keyword_signals(conn, profile_signals + tag_signals)
    return LocalAiContextScanSummary(
        tickers_considered=len(candidate_tickers),
        profiles_scanned=profiles_scanned,
        profile_signals=len(profile_signals),
        tag_rows_scanned=tag_rows_scanned,
        tag_signals=len(tag_signals),
        signals_written=signals_written,
    )


def store_keyword_signals(conn: sqlite3.Connection, signals: list[KeywordSignal]) -> int:
    now = utc_now_iso()
    ai_rows = []
    evidence_rows = []
    for signal in signals:
        ai_rows.append(
            (
                signal.ticker,
                signal.signal_date,
                signal.source_type,
                signal.source_url,
                signal.keyword,
                signal.context_snippet,
                signal.ai_relevance_level,
                signal.confidence,
                now,
            )
        )
        content_hash = _hash_signal(signal)
        evidence_rows.append(
            (
                signal.source_type,
                signal.source_type,
                signal.source_url,
                signal.signal_date,
                now,
                f"{signal.ticker} keyword: {signal.keyword}",
                "Automated keyword context candidate; requires human verification before strong conclusions.",
                signal.context_snippet,
                signal.confidence,
                signal.ticker,
                "ai_relevance",
                content_hash,
            )
        )
    conn.executemany(
        """
        INSERT INTO ai_relevance_signals (
            ticker, signal_date, source_type, source_url, keyword, context_snippet,
            ai_relevance_level, confidence, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, source_url, keyword, context_snippet) DO UPDATE SET
            ai_relevance_level=excluded.ai_relevance_level,
            confidence=excluded.confidence,
            updated_at=excluded.updated_at
        """,
        ai_rows,
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO evidence_items (
            source_type, source_name, url, published_at, fetched_at, raw_title, summary,
            evidence_snippet, confidence, related_ticker, related_module, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        evidence_rows,
    )
    return len(signals)


def _signal_from_ai_tag(ticker: str, row: sqlite3.Row) -> KeywordSignal:
    tag = str(row["tag"]).strip()
    confidence = min(
        TAG_SIGNAL_MAX_CONFIDENCE,
        max(0.25, float(row["confidence"] or 0.35) * 0.75),
    )
    source_type = str(row["source_type"] or "ai_industry_tag").strip()
    snippet = (
        f"AI industry-chain tag candidate '{tag}' from {source_type}: "
        f"{str(row['evidence_snippet']).strip()} "
        "Use as supply-chain context only; it does not prove revenue, orders, customers, or investment merit."
    )
    return KeywordSignal(
        ticker=ticker.upper(),
        signal_date=None,
        source_type=TAG_SIGNAL_SOURCE_TYPE,
        source_url=str(row["source_url"]).strip(),
        keyword=f"{TAG_SIGNAL_KEYWORD_PREFIX}{tag}",
        context_snippet=_trim(snippet, 520),
        ai_relevance_level=TAG_SIGNAL_LEVEL,
        confidence=confidence,
    )


def _candidate_tickers(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None,
    limit: int,
) -> list[str]:
    if tickers:
        return _unique_tickers(tickers)[: max(0, limit)]

    rows = conn.execute(
        """
        SELECT ticker
        FROM (
            SELECT ticker, 0 AS source_rank
            FROM ai_industry_tags
            UNION
            SELECT ticker, 1 AS source_rank
            FROM ai_relevance_signals
            UNION
            SELECT ticker, 2 AS source_rank
            FROM company_profile
            WHERE COALESCE(description, '') <> ''
        )
        GROUP BY ticker
        ORDER BY MIN(source_rank), ticker
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [str(row["ticker"]).upper() for row in rows]


def _unique_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for ticker in tickers:
        value = ticker.strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _estimate_level(snippet: str) -> tuple[int, float]:
    lower = snippet.lower()
    has_business_impact = any(term in lower for term in BUSINESS_IMPACT_TERMS)
    if has_business_impact:
        return 2, 0.6
    if any(term in lower for term in RISK_CONTEXT_TERMS):
        return 1, 0.35
    return 1, 0.45


def _has_ai_context(snippet: str, keyword: str) -> bool:
    lower = snippet.lower()
    if keyword in {"data center", "datacenter", "gpu", "hbm", "liquid cooling"}:
        return True
    return any(term in lower for term in CORE_AI_KEYWORDS) or "data center" in lower or "datacenter" in lower


def _snippet(text: str, start: int, end: int, window: int = 260) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "... " if left > 0 else ""
    suffix = " ..." if right < len(text) else ""
    return f"{prefix}{text[left:right].strip()}{suffix}"


def _trim(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 4].rstrip() + " ..."


def _hash_signal(signal: KeywordSignal) -> str:
    raw = "|".join(
        [
            signal.ticker,
            signal.source_url,
            signal.keyword,
            signal.context_snippet,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
