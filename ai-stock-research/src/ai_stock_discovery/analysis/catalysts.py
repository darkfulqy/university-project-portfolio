from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


CATALYST_RULES: dict[str, list[str]] = {
    "earnings_or_filing": [
        "10-k",
        "10-q",
        "earnings",
        "quarterly results",
        "annual report",
    ],
    "guidance_change": [
        "guidance",
        "outlook",
        "forecast",
        "raise",
        "raised",
        "lower",
        "lowered",
    ],
    "order_contract": [
        "order",
        "orders",
        "contract",
        "backlog",
        "booking",
        "bookings",
        "award",
        "awarded",
    ],
    "product_launch": [
        "launch",
        "launched",
        "new product",
        "introduces",
    ],
    "data_center_project": [
        "data center",
        "datacenter",
        "substation",
        "power capacity",
        "liquid cooling",
    ],
    "capacity_expansion": [
        "capacity",
        "expansion",
        "ramp",
        "facility",
        "manufacturing",
    ],
    "management_change": [
        "appoint",
        "appointed",
        "resign",
        "resigned",
        "chief executive",
        "chief financial",
    ],
    "regulatory_policy": [
        "regulatory",
        "regulation",
        "policy",
        "approval",
        "investigation",
    ],
}

SOURCE_MODULES = {"filing_monitor", "ir_monitor", "news_monitor", "transcript_snippet"}
FILING_CATALYST_FORMS = {"10-K", "10-Q", "8-K"}


@dataclass(frozen=True)
class CatalystCandidate:
    ticker: str
    catalyst_type: str
    catalyst_date: str | None
    description: str
    source_url: str
    confidence: float
    status: str = "candidate"


def extract_catalysts_from_evidence(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    limit: int = 200,
) -> list[CatalystCandidate]:
    params: list[object] = []
    where = "related_ticker IS NOT NULL"
    where += " AND related_module IN (%s)" % ",".join("?" for _ in SOURCE_MODULES)
    params.extend(sorted(SOURCE_MODULES))
    if ticker:
        where += " AND related_ticker = ?"
        params.append(ticker.upper())
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            related_ticker, related_module, source_type, source_name, url,
            published_at, raw_title, summary, evidence_snippet, confidence
        FROM evidence_items
        WHERE {where}
        ORDER BY fetched_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    candidates: list[CatalystCandidate] = []
    for row in rows:
        if _is_ir_page_snapshot(row):
            continue
        text = " ".join(
            part
            for part in [
                row["source_type"] or "",
                row["raw_title"] or "",
                row["summary"] or "",
                row["evidence_snippet"] or "",
            ]
            if part
        )
        for catalyst_type, keywords in CATALYST_RULES.items():
            matched = [keyword for keyword in keywords if keyword in text.lower()]
            if not matched:
                continue
            confidence = min(0.85, max(0.35, float(row["confidence"] or 0.35)) + min(0.15, 0.03 * len(matched)))
            candidates.append(
                CatalystCandidate(
                    ticker=str(row["related_ticker"]).upper(),
                    catalyst_type=catalyst_type,
                    catalyst_date=row["published_at"],
                    description=_description(text, matched[0]),
                    source_url=row["url"],
                    confidence=confidence,
                )
            )
    candidates.extend(_candidates_from_filings(conn, ticker=ticker, limit=limit))
    return _dedupe(candidates)


def _is_ir_page_snapshot(row: sqlite3.Row) -> bool:
    if row["related_module"] != "ir_monitor":
        return False
    source_name = (row["source_name"] or "").strip().lower()
    summary = (row["summary"] or "").strip().lower()
    evidence_snippet = (row["evidence_snippet"] or "").strip().lower()
    return (
        source_name == "company ir page"
        or "ir page hash check" in summary
        or "ir page observed with title" in evidence_snippet
    )


def _candidates_from_filings(
    conn: sqlite3.Connection,
    *,
    ticker: str | None,
    limit: int,
) -> list[CatalystCandidate]:
    params: list[object] = list(sorted(FILING_CATALYST_FORMS))
    where = "form IN (%s)" % ",".join("?" for _ in FILING_CATALYST_FORMS)
    if ticker:
        where += " AND ticker = ?"
        params.append(ticker.upper())
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT ticker, form, filed_at, filing_url, document_url
        FROM filings
        WHERE ticker IS NOT NULL
          AND ticker != ''
          AND {where}
        ORDER BY COALESCE(filed_at, '') DESC, updated_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    candidates: list[CatalystCandidate] = []
    for row in rows:
        source_url = row["document_url"] or row["filing_url"]
        if not source_url:
            continue
        form = str(row["form"]).upper()
        ticker_value = str(row["ticker"]).upper()
        candidates.append(
            CatalystCandidate(
                ticker=ticker_value,
                catalyst_type="earnings_or_filing",
                catalyst_date=row["filed_at"],
                description=(
                    f"SEC {form} filing metadata for {ticker_value}. "
                    "This is a filing-review catalyst candidate only; review the original filing "
                    "before drawing company-level conclusions."
                ),
                source_url=source_url,
                confidence=0.75 if form in {"10-K", "10-Q"} else 0.65,
            )
        )
    return candidates


def store_catalysts(conn: sqlite3.Connection, candidates: list[CatalystCandidate]) -> int:
    inserted = 0
    now = utc_now_iso()
    for candidate in candidates:
        existing = conn.execute(
            """
            SELECT id FROM catalysts
            WHERE ticker = ?
              AND catalyst_type = ?
              AND COALESCE(catalyst_date, '') = COALESCE(?, '')
              AND description = ?
              AND source_url = ?
            """,
            (
                candidate.ticker.upper(),
                candidate.catalyst_type,
                candidate.catalyst_date,
                candidate.description,
                candidate.source_url,
            ),
        ).fetchone()
        if existing:
            continue
        conn.execute(
            """
            INSERT INTO catalysts (
                ticker, catalyst_type, catalyst_date, description,
                source_url, confidence, status, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.ticker.upper(),
                candidate.catalyst_type,
                candidate.catalyst_date,
                candidate.description,
                candidate.source_url,
                candidate.confidence,
                candidate.status,
                now,
            ),
        )
        inserted += 1
    return inserted


def _description(text: str, keyword: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    index = compact.lower().find(keyword.lower())
    if index < 0:
        return compact[:500]
    left = max(0, index - 220)
    right = min(len(compact), index + len(keyword) + 220)
    prefix = "... " if left > 0 else ""
    suffix = " ..." if right < len(compact) else ""
    return f"{prefix}{compact[left:right].strip()}{suffix}"


def _dedupe(candidates: list[CatalystCandidate]) -> list[CatalystCandidate]:
    best: dict[tuple[str, str, str], CatalystCandidate] = {}
    for candidate in candidates:
        key = (candidate.ticker.upper(), candidate.catalyst_type, candidate.source_url)
        current = best.get(key)
        if current is None or candidate.confidence > current.confidence:
            best[key] = candidate
    return sorted(best.values(), key=lambda item: (item.ticker, item.catalyst_type, -item.confidence))
