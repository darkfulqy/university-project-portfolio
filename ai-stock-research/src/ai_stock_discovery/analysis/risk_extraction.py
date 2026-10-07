from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import sqlite3

from ai_stock_discovery.sources.risks import RiskFlag, upsert_risk_flags


@dataclass(frozen=True)
class RiskRule:
    risk_type: str
    severity: str
    confidence: float
    terms: tuple[str, ...]


RISK_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        risk_type="customer_concentration",
        severity="high",
        confidence=0.62,
        terms=(
            "customer concentration",
            "limited number of customers",
            "single customer",
            "major customer",
            "significant customer",
            "substantial portion of our revenue",
            "loss of one or more customers",
        ),
    ),
    RiskRule(
        risk_type="dilution",
        severity="medium",
        confidence=0.55,
        terms=(
            "dilution",
            "dilutive",
            "issue additional shares",
            "equity financing",
            "convertible notes",
            "warrants to purchase",
            "at-the-market offering",
        ),
    ),
    RiskRule(
        risk_type="margin_deterioration",
        severity="medium",
        confidence=0.56,
        terms=(
            "gross margin declined",
            "gross margins declined",
            "gross margin may decline",
            "pressure on margins",
            "pricing pressure",
            "lower gross margin",
            "margin compression",
        ),
    ),
    RiskRule(
        risk_type="export_control",
        severity="high",
        confidence=0.62,
        terms=(
            "export control",
            "export controls",
            "trade restrictions",
            "sanctions",
            "entity list",
            "china restrictions",
            "geopolitical tensions",
        ),
    ),
    RiskRule(
        risk_type="supply_chain",
        severity="medium",
        confidence=0.58,
        terms=(
            "supply chain disruption",
            "supply chain disruptions",
            "component shortages",
            "supplier concentration",
            "single source supplier",
            "sole supplier",
            "limited suppliers",
            "manufacturing constraints",
        ),
    ),
    RiskRule(
        risk_type="accounting_quality",
        severity="high",
        confidence=0.66,
        terms=(
            "material weakness",
            "internal control over financial reporting",
            "restatement",
            "going concern",
            "substantial doubt",
            "audit opinion",
        ),
    ),
    RiskRule(
        risk_type="litigation",
        severity="medium",
        confidence=0.54,
        terms=(
            "material litigation",
            "class action",
            "lawsuit",
            "legal proceedings",
            "intellectual property litigation",
            "patent infringement",
        ),
    ),
    RiskRule(
        risk_type="regulatory",
        severity="medium",
        confidence=0.54,
        terms=(
            "regulatory investigation",
            "government investigation",
            "regulatory approval",
            "regulatory proceedings",
            "compliance with regulations",
            "privacy regulations",
        ),
    ),
    RiskRule(
        risk_type="cybersecurity",
        severity="medium",
        confidence=0.5,
        terms=(
            "cybersecurity incident",
            "data breach",
            "security breach",
            "ransomware",
            "unauthorized access",
        ),
    ),
)

SOURCE_MODULES = {"filing_monitor", "ir_monitor", "news_monitor", "transcript_snippet"}


def scan_risk_text(
    *,
    ticker: str,
    text: str,
    source_type: str,
    source_name: str,
    source_url: str,
    risk_date: str | None = None,
    status: str = "watch",
    max_hits_per_type: int = 3,
) -> list[RiskFlag]:
    normalized = re.sub(r"\s+", " ", text)
    flags: list[RiskFlag] = []
    for rule in RISK_RULES:
        hits = 0
        for term in rule.terms:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(normalized):
                snippet = _snippet(normalized, match.start(), match.end())
                flags.append(
                    _make_flag(
                        ticker=ticker,
                        risk_date=risk_date,
                        source_type=source_type,
                        source_name=source_name,
                        source_url=source_url,
                        risk_type=rule.risk_type,
                        severity=rule.severity,
                        description=f"Automated source-text risk candidate: {snippet}",
                        confidence=rule.confidence,
                        status=status,
                    )
                )
                hits += 1
                if hits >= max_hits_per_type:
                    break
            if hits >= max_hits_per_type:
                break
    return _dedupe(flags)


def extract_risk_flags_from_evidence(
    conn: sqlite3.Connection,
    *,
    ticker: str | None = None,
    limit: int = 200,
) -> list[RiskFlag]:
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
        SELECT related_ticker, source_type, source_name, url, published_at,
               raw_title, summary, evidence_snippet
        FROM evidence_items
        WHERE {where}
        ORDER BY fetched_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    flags: list[RiskFlag] = []
    for row in rows:
        text = " ".join(
            part
            for part in [
                row["raw_title"] or "",
                row["summary"] or "",
                row["evidence_snippet"] or "",
            ]
            if part
        )
        flags.extend(
            scan_risk_text(
                ticker=row["related_ticker"],
                text=text,
                source_type=row["source_type"],
                source_name=row["source_name"],
                source_url=row["url"],
                risk_date=row["published_at"],
            )
        )
    return _dedupe(flags)


def store_extracted_risk_flags(conn: sqlite3.Connection, flags: list[RiskFlag]) -> int:
    return upsert_risk_flags(conn, flags)


def _make_flag(
    *,
    ticker: str,
    risk_date: str | None,
    source_type: str,
    source_name: str,
    source_url: str,
    risk_type: str,
    severity: str,
    description: str,
    confidence: float,
    status: str,
) -> RiskFlag:
    ticker = ticker.upper()
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                risk_date or "",
                source_type,
                source_name,
                source_url,
                risk_type,
                severity,
                description,
                status,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return RiskFlag(
        ticker=ticker,
        risk_date=risk_date,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        risk_type=risk_type,
        severity=severity,
        description=description,
        confidence=confidence,
        status=status,
        content_hash=content_hash,
    )


def _snippet(text: str, start: int, end: int, window: int = 260) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "... " if left > 0 else ""
    suffix = " ..." if right < len(text) else ""
    return f"{prefix}{text[left:right].strip()}{suffix}"


def _dedupe(flags: list[RiskFlag]) -> list[RiskFlag]:
    best: dict[tuple[str, str, str], RiskFlag] = {}
    for flag in flags:
        key = (flag.ticker, flag.risk_type, flag.content_hash)
        best.setdefault(key, flag)
    return sorted(best.values(), key=lambda item: (item.ticker, item.risk_type, item.source_url))
