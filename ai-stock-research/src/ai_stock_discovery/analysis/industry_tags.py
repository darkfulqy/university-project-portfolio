from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


TAG_RULES: dict[str, list[str]] = {
    "ai_compute": [
        "gpu",
        "accelerator",
        "accelerated computing",
        "asic",
        "server",
        "ai workload",
    ],
    "semiconductors": [
        "semiconductor",
        "chip",
        "hbm",
        "advanced packaging",
        "foundry",
        "wafer",
        "eda",
        "memory",
    ],
    "semiconductor_equipment": [
        "semiconductor equipment",
        "deposition",
        "etch",
        "lithography",
        "metrology",
        "wafer inspection",
    ],
    "materials": [
        "substrate",
        "photoresist",
        "silicon carbide",
        "specialty materials",
        "electronic materials",
    ],
    "data_center_infrastructure": [
        "data center",
        "datacenter",
        "ups",
        "switchgear",
        "power distribution",
        "transformer",
        "substation",
        "backup power",
    ],
    "power": [
        "power",
        "electric",
        "grid",
        "generator",
        "turbine",
        "energy storage",
        "utility",
    ],
    "cooling": [
        "cooling",
        "liquid cooling",
        "thermal management",
        "hvac",
        "heat exchanger",
        "chiller",
    ],
    "networking_connectivity": [
        "networking",
        "ethernet",
        "infiniband",
        "optical module",
        "transceiver",
        "switch",
        "connector",
    ],
    "cloud": [
        "cloud",
        "iaas",
        "paas",
        "database",
        "data platform",
        "mlops",
    ],
    "software_enterprise_ai": [
        "software",
        "saas",
        "automation",
        "generative ai",
        "workflow",
        "enterprise ai",
    ],
    "cybersecurity": [
        "cybersecurity",
        "security",
        "identity",
        "endpoint",
        "data protection",
        "zero trust",
    ],
    "automation": [
        "automation",
        "robotics",
        "industrial automation",
        "factory automation",
    ],
    "vertical_applications": [
        "healthcare ai",
        "legal ai",
        "design software",
        "industrial software",
        "customer service",
        "marketing automation",
    ],
}


@dataclass(frozen=True)
class AiIndustryTag:
    ticker: str
    tag: str
    confidence: float
    source_type: str
    source_url: str
    evidence_snippet: str
    method: str = "rule"


@dataclass(frozen=True)
class AiTagInferenceSummary:
    tickers_considered: int
    tickers_tagged: int
    tags_written: int


def infer_ai_chain_tags(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    limit: int = 500,
) -> AiTagInferenceSummary:
    candidate_tickers = _candidate_tickers(conn, tickers=tickers, limit=limit)
    tickers_tagged = 0
    tags_written = 0
    for ticker in candidate_tickers:
        tags = infer_tags_for_ticker(conn, ticker)
        written = replace_inferred_tags_for_ticker(conn, ticker, tags)
        tags_written += written
        if written:
            tickers_tagged += 1
    return AiTagInferenceSummary(
        tickers_considered=len(candidate_tickers),
        tickers_tagged=tickers_tagged,
        tags_written=tags_written,
    )


def infer_tags_for_ticker(conn: sqlite3.Connection, ticker: str) -> list[AiIndustryTag]:
    ticker = ticker.upper()
    tags: list[AiIndustryTag] = []
    profile = conn.execute(
        """
        SELECT sector, industry, description, source
        FROM company_profile
        WHERE ticker = ?
        """,
        (ticker,),
    ).fetchone()
    if profile:
        text_parts = [
            profile["sector"] or "",
            profile["industry"] or "",
            profile["description"] or "",
        ]
        text = " ".join(text_parts).strip()
        if text:
            tags.extend(
                _infer_from_text(
                    ticker=ticker,
                    text=text,
                    source_type="company_profile",
                    source_url=profile["source"] or "local company_profile",
                    base_confidence=0.55,
                )
            )
    signal_rows = conn.execute(
        """
        SELECT source_type, source_url, keyword, context_snippet, ai_relevance_level, confidence
        FROM ai_relevance_signals
        WHERE ticker = ?
        ORDER BY confidence DESC
        LIMIT 40
        """,
        (ticker,),
    ).fetchall()
    for row in signal_rows:
        if int(row["ai_relevance_level"] or 0) < 2 or float(row["confidence"] or 0) < 0.45:
            continue
        text = f"{row['keyword']} {row['context_snippet']}"
        tags.extend(
            _infer_from_text(
                ticker=ticker,
                text=text,
                source_type=row["source_type"],
                source_url=row["source_url"],
                base_confidence=min(0.75, max(0.35, float(row["confidence"] or 0.35))),
            )
        )
    return _dedupe(tags)


def replace_inferred_tags_for_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    tags: list[AiIndustryTag],
) -> int:
    conn.execute(
        "DELETE FROM ai_industry_tags WHERE ticker = ? AND method = 'rule'",
        (ticker.upper(),),
    )
    return store_ai_industry_tags(conn, tags)


def import_ai_tags_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    tags: list[AiIndustryTag] = []
    default_source = f"manual_ai_tags_csv:{csv_path.name}"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = _field(row, "ticker", "Ticker")
            tag = _field(row, "tag", "ai_tag", "AI Tag")
            if not ticker or not tag:
                continue
            tags.append(
                AiIndustryTag(
                    ticker=ticker.upper(),
                    tag=tag,
                    confidence=_number(_field(row, "confidence")) or 0.7,
                    source_type=_field(row, "source_type") or "manual",
                    source_url=_field(row, "source_url", "source") or default_source,
                    evidence_snippet=_field(row, "evidence_snippet", "evidence")
                    or f"Manual AI industry-chain tag: {tag}",
                    method="manual",
                )
            )
    return store_ai_industry_tags(conn, tags)


def store_ai_industry_tags(conn: sqlite3.Connection, tags: list[AiIndustryTag]) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO ai_industry_tags (
            ticker, tag, confidence, source_type, source_url,
            evidence_snippet, method, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, tag, source_url, evidence_snippet) DO UPDATE SET
            confidence=excluded.confidence,
            source_type=excluded.source_type,
            method=excluded.method,
            updated_at=excluded.updated_at
        """,
        [
            (
                tag.ticker.upper(),
                tag.tag,
                tag.confidence,
                tag.source_type,
                tag.source_url,
                tag.evidence_snippet,
                tag.method,
                now,
            )
            for tag in tags
        ],
    )
    _store_tag_evidence(conn, tags, now)
    return len(tags)


def _infer_from_text(
    *,
    ticker: str,
    text: str,
    source_type: str,
    source_url: str,
    base_confidence: float,
) -> list[AiIndustryTag]:
    lower = text.lower()
    tags: list[AiIndustryTag] = []
    for tag, keywords in TAG_RULES.items():
        matched = [keyword for keyword in keywords if keyword in lower]
        if not matched:
            continue
        confidence = min(0.85, base_confidence + min(0.2, 0.05 * len(matched)))
        tags.append(
            AiIndustryTag(
                ticker=ticker.upper(),
                tag=tag,
                confidence=confidence,
                source_type=source_type,
                source_url=source_url,
                evidence_snippet=_snippet(text, matched[0]),
                method="rule",
            )
        )
    return tags


def _dedupe(tags: list[AiIndustryTag]) -> list[AiIndustryTag]:
    best: dict[tuple[str, str, str], AiIndustryTag] = {}
    for tag in tags:
        key = (tag.ticker.upper(), tag.tag, tag.source_url)
        current = best.get(key)
        if current is None or tag.confidence > current.confidence:
            best[key] = tag
    return sorted(best.values(), key=lambda item: (item.ticker, item.tag, -item.confidence))


def _candidate_tickers(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None,
    limit: int,
) -> list[str]:
    if tickers:
        return _unique_tickers(tickers)[:limit]

    active_universe_count = conn.execute(
        "SELECT COUNT(*) AS count FROM universe WHERE is_active = 1"
    ).fetchone()["count"]
    if int(active_universe_count or 0) > 0:
        rows = conn.execute(
            """
            SELECT ticker
            FROM (
                SELECT u.ticker, 2 AS source_rank
                FROM universe u
                WHERE u.is_active = 1
                UNION
                SELECT ticker, 0 AS source_rank
                FROM ai_relevance_signals
                UNION
                SELECT ticker, 1 AS source_rank
                FROM company_profile
                WHERE COALESCE(description, '') <> ''
                  AND COALESCE(source, '') <> 'SEC company_tickers'
            ) candidates
            WHERE NOT EXISTS (
                SELECT 1
                FROM universe_exclusions ex
                WHERE ex.ticker = candidates.ticker
                  AND ex.is_active = 1
                  AND ex.severity = 'exclude'
            )
            GROUP BY ticker
            ORDER BY MIN(source_rank), ticker
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [str(row["ticker"]).upper() for row in rows]

    rows = conn.execute(
        """
        SELECT ticker
        FROM (
            SELECT ticker, 0 AS source_rank
            FROM ai_relevance_signals
            WHERE ai_relevance_level >= 2
              AND confidence >= 0.45
            UNION
            SELECT ticker, 1 AS source_rank
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
        normalized.append(value)
        seen.add(value)
    return normalized


def _store_tag_evidence(
    conn: sqlite3.Connection,
    tags: list[AiIndustryTag],
    fetched_at: str,
) -> None:
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
                tag.source_type,
                "AI industry-chain tag",
                tag.source_url,
                None,
                fetched_at,
                f"{tag.ticker.upper()} AI industry tag: {tag.tag}",
                "Candidate AI industry-chain tag. Requires source review before high-conviction use.",
                tag.evidence_snippet,
                tag.confidence,
                tag.ticker.upper(),
                "ai_industry_tag",
                f"ai-tag:{_hash_tag(tag)}",
            )
            for tag in tags
        ],
    )


def _snippet(text: str, keyword: str, window: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    index = compact.lower().find(keyword.lower())
    if index < 0:
        return compact[: window * 2]
    left = max(0, index - window)
    right = min(len(compact), index + len(keyword) + window)
    prefix = "... " if left > 0 else ""
    suffix = " ..." if right < len(compact) else ""
    return f"{prefix}{compact[left:right].strip()}{suffix}"


def _hash_tag(tag: AiIndustryTag) -> str:
    raw = "|".join([tag.ticker.upper(), tag.tag, tag.source_url, tag.evidence_snippet])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
