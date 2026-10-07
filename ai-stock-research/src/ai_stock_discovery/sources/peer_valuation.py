from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class PeerValuationComparison:
    ticker: str
    comparison_date: str | None
    peer_group: str | None
    peer_tickers: str
    metric: str
    target_value: float
    peer_median: float
    peer_mean: float | None
    discount_premium_pct: float | None
    direction: str
    source_type: str
    source_name: str
    source_url: str
    description: str
    confidence: float
    content_hash: str


def import_peer_valuation_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return 0
        required = {
            "ticker",
            "source_url",
            "peer_tickers",
            "metric",
            "target_value",
            "peer_median",
            "description",
            "confidence",
        }
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"Peer valuation CSV missing required columns: {', '.join(sorted(missing))}")
        comparisons = [_comparison_from_row(row, csv_path) for row in reader]
    return upsert_peer_valuation_comparisons(conn, comparisons)


def upsert_peer_valuation_comparisons(
    conn: sqlite3.Connection,
    comparisons: list[PeerValuationComparison],
) -> int:
    now = utc_now_iso()
    conn.executemany(
        """
        INSERT INTO peer_valuation_comparisons (
            ticker, comparison_date, peer_group, peer_tickers, metric,
            target_value, peer_median, peer_mean, discount_premium_pct,
            direction, source_type, source_name, source_url, description,
            confidence, content_hash, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            comparison_date=excluded.comparison_date,
            peer_group=excluded.peer_group,
            peer_tickers=excluded.peer_tickers,
            metric=excluded.metric,
            target_value=excluded.target_value,
            peer_median=excluded.peer_median,
            peer_mean=excluded.peer_mean,
            discount_premium_pct=excluded.discount_premium_pct,
            direction=excluded.direction,
            source_type=excluded.source_type,
            source_name=excluded.source_name,
            source_url=excluded.source_url,
            description=excluded.description,
            confidence=excluded.confidence,
            updated_at=excluded.updated_at
        """,
        [
            (
                comparison.ticker,
                comparison.comparison_date,
                comparison.peer_group,
                comparison.peer_tickers,
                comparison.metric,
                comparison.target_value,
                comparison.peer_median,
                comparison.peer_mean,
                comparison.discount_premium_pct,
                comparison.direction,
                comparison.source_type,
                comparison.source_name,
                comparison.source_url,
                comparison.description,
                comparison.confidence,
                comparison.content_hash,
                now,
            )
            for comparison in comparisons
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
                comparison.source_type,
                comparison.source_name,
                comparison.source_url,
                comparison.comparison_date,
                now,
                f"{comparison.ticker} peer valuation comparison",
                comparison.description,
                _evidence_snippet(comparison),
                comparison.confidence,
                comparison.ticker,
                "peer_valuation",
                f"peer-valuation:{comparison.content_hash}",
            )
            for comparison in comparisons
        ],
    )
    return len(comparisons)


def _comparison_from_row(row: dict[str, str], csv_path: Path) -> PeerValuationComparison:
    ticker = _field(row, "ticker").upper()
    source_url = _field(row, "source_url")
    peer_tickers = _field(row, "peer_tickers", "peers")
    metric = _normalize_metric(_field(row, "metric"))
    target_value = _required_float(row, "target_value")
    peer_median = _required_float(row, "peer_median")
    peer_mean = _optional_float(_field(row, "peer_mean"))
    discount_premium_pct = _optional_float(_field(row, "discount_premium_pct", "discount_pct"))
    if discount_premium_pct is None:
        discount_premium_pct = _calculate_discount(metric, target_value, peer_median)
    direction = _normalize_direction(_field(row, "direction"), discount_premium_pct)
    description = _field(row, "description")
    confidence = _bounded_float(_field(row, "confidence"), default=0.5, minimum=0.0, maximum=1.0)
    comparison_date = _field(row, "comparison_date", "date") or None
    peer_group = _field(row, "peer_group", "group") or None
    source_type = _field(row, "source_type") or "peer_valuation_csv"
    source_name = _field(row, "source_name") or csv_path.name
    if not ticker:
        raise ValueError("Peer valuation CSV contains a row without ticker.")
    if not source_url:
        raise ValueError(f"Peer valuation CSV row for {ticker} is missing source_url.")
    if not peer_tickers:
        raise ValueError(f"Peer valuation CSV row for {ticker} is missing peer_tickers.")
    if not metric:
        raise ValueError(f"Peer valuation CSV row for {ticker} is missing metric.")
    if not description:
        raise ValueError(f"Peer valuation CSV row for {ticker} is missing description.")
    content_hash = hashlib.sha256(
        "|".join(
            [
                ticker,
                comparison_date or "",
                peer_group or "",
                peer_tickers,
                metric,
                f"{target_value:g}",
                f"{peer_median:g}",
                f"{peer_mean:g}" if peer_mean is not None else "",
                f"{discount_premium_pct:g}" if discount_premium_pct is not None else "",
                direction,
                source_type,
                source_name,
                source_url,
                description,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return PeerValuationComparison(
        ticker=ticker,
        comparison_date=comparison_date,
        peer_group=peer_group,
        peer_tickers=peer_tickers,
        metric=metric,
        target_value=target_value,
        peer_median=peer_median,
        peer_mean=peer_mean,
        discount_premium_pct=discount_premium_pct,
        direction=direction,
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        description=description,
        confidence=confidence,
        content_hash=content_hash,
    )


def _calculate_discount(metric: str, target_value: float, peer_median: float) -> float | None:
    if peer_median == 0:
        return None
    if metric == "fcf_yield":
        return (target_value - peer_median) / abs(peer_median)
    return (peer_median - target_value) / abs(peer_median)


def _normalize_metric(raw: str) -> str:
    value = raw.strip().lower().replace("/", "_").replace("-", "_")
    aliases = {
        "ev_sales": "ev_sales",
        "ev_to_sales": "ev_sales",
        "ev_ebitda": "ev_ebitda",
        "ev_to_ebitda": "ev_ebitda",
        "pe": "pe",
        "p_e": "pe",
        "fcf_yield": "fcf_yield",
        "free_cash_flow_yield": "fcf_yield",
    }
    return aliases.get(value, value)


def _normalize_direction(raw: str, discount_premium_pct: float | None) -> str:
    value = raw.strip().lower()
    if value in {"supports_discount", "neutral", "contradicts_discount"}:
        return value
    if discount_premium_pct is None:
        return "neutral"
    if discount_premium_pct >= 0.15:
        return "supports_discount"
    if discount_premium_pct <= -0.15:
        return "contradicts_discount"
    return "neutral"


def _evidence_snippet(comparison: PeerValuationComparison) -> str:
    discount = (
        "NA"
        if comparison.discount_premium_pct is None
        else f"{comparison.discount_premium_pct:.2%}"
    )
    return (
        f"{comparison.metric}: target={comparison.target_value:g}, "
        f"peer_median={comparison.peer_median:g}, "
        f"discount_premium_pct={discount}, direction={comparison.direction}, "
        f"peers={comparison.peer_tickers}. {comparison.description}"
    )


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _required_float(row: dict[str, str], name: str) -> float:
    value = _optional_float(_field(row, name))
    if value is None:
        ticker = _field(row, "ticker") or "unknown"
        raise ValueError(f"Peer valuation CSV row for {ticker} is missing numeric {name}.")
    return value


def _optional_float(raw: str) -> float | None:
    if not raw:
        return None
    return float(raw.replace(",", ""))


def _bounded_float(raw: str, *, default: float, minimum: float, maximum: float) -> float:
    value = default if not raw else float(raw)
    return max(minimum, min(maximum, value))
