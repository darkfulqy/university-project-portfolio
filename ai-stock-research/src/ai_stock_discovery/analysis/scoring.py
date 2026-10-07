from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from ai_stock_discovery.timeutils import utc_now_iso


@dataclass(frozen=True)
class ScoreResult:
    ticker: str
    status: str
    score_total: float
    ai_relevance_score: float
    expectation_gap_score: float
    fundamental_score: float
    valuation_score: float
    catalyst_score: float
    market_confirmation_score: float
    risk_penalty: float
    thesis: str
    risk_summary: str
    invalidating_conditions: str
    notes: list[str]


def score_ticker(conn: sqlite3.Connection, ticker: str) -> ScoreResult:
    ticker = ticker.upper()
    ai_score, ai_note = _score_ai_relevance(conn, ticker)
    fundamental_score, fundamental_note = _score_fundamentals(conn, ticker)
    catalyst_score, catalyst_note = _score_catalysts(conn, ticker)
    expectation_gap_score, expectation_gap_note = _score_expectation_gap(conn, ticker)
    valuation_score, valuation_note = _score_valuation(conn, ticker)
    market_confirmation_score, market_confirmation_note = _score_market_confirmation(conn, ticker)
    risk_penalty, risk_note, risk_summary = _score_risks(conn, ticker)

    total = (
        ai_score
        + expectation_gap_score
        + fundamental_score
        + valuation_score
        + catalyst_score
        + market_confirmation_score
        + risk_penalty
    )
    status = _status_for_score(total)
    notes = [
        ai_note,
        fundamental_note,
        valuation_note,
        catalyst_note,
        expectation_gap_note,
        market_confirmation_note,
        risk_note,
    ]
    thesis = _build_thesis(ticker, total, notes)
    invalidating_conditions = "Thesis should fail if AI relevance cannot be tied to source-backed business impact, fundamentals deteriorate, valuation already discounts the change, or key catalysts do not materialize."
    return ScoreResult(
        ticker=ticker,
        status=status,
        score_total=round(total, 2),
        ai_relevance_score=round(ai_score, 2),
        expectation_gap_score=round(expectation_gap_score, 2),
        fundamental_score=round(fundamental_score, 2),
        valuation_score=round(valuation_score, 2),
        catalyst_score=round(catalyst_score, 2),
        market_confirmation_score=market_confirmation_score,
        risk_penalty=risk_penalty,
        thesis=thesis,
        risk_summary=risk_summary,
        invalidating_conditions=invalidating_conditions,
        notes=notes,
    )


def upsert_research_card(conn: sqlite3.Connection, result: ScoreResult) -> None:
    conn.execute(
        """
        INSERT INTO research_cards (
            ticker, status, score_total, ai_relevance_score, expectation_gap_score,
            fundamental_score, valuation_score, catalyst_score, market_confirmation_score,
            risk_penalty, thesis, risk_summary, invalidating_conditions, last_reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            status=excluded.status,
            score_total=excluded.score_total,
            ai_relevance_score=excluded.ai_relevance_score,
            expectation_gap_score=excluded.expectation_gap_score,
            fundamental_score=excluded.fundamental_score,
            valuation_score=excluded.valuation_score,
            catalyst_score=excluded.catalyst_score,
            market_confirmation_score=excluded.market_confirmation_score,
            risk_penalty=excluded.risk_penalty,
            thesis=excluded.thesis,
            risk_summary=excluded.risk_summary,
            invalidating_conditions=excluded.invalidating_conditions,
            last_reviewed_at=excluded.last_reviewed_at
        """,
        (
            result.ticker,
            result.status,
            result.score_total,
            result.ai_relevance_score,
            result.expectation_gap_score,
            result.fundamental_score,
            result.valuation_score,
            result.catalyst_score,
            result.market_confirmation_score,
            result.risk_penalty,
            result.thesis,
            result.risk_summary,
            result.invalidating_conditions,
            utc_now_iso(),
        ),
    )


def _score_ai_relevance(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    row = conn.execute(
        """
        SELECT
            COALESCE(MAX(ai_relevance_level), 0) AS max_level,
            COUNT(*) AS signal_count,
            COALESCE(MAX(confidence), 0) AS max_confidence
        FROM ai_relevance_signals
        WHERE ticker = ?
        """,
        (ticker,),
    ).fetchone()
    max_level = int(row["max_level"] or 0)
    signal_count = int(row["signal_count"] or 0)
    confidence = float(row["max_confidence"] or 0)
    score = min(25.0, (max_level / 4.0) * 25.0 + min(signal_count, 6) * confidence * 0.6)
    if signal_count:
        note = f"AI relevance has {signal_count} source-backed keyword/context signal(s); max auto level is {max_level}."
    else:
        note = "AI relevance score is 0 because no source-backed AI evidence has been loaded."
    return score, note


def _score_fundamentals(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT period, fiscal_year, fiscal_quarter, revenue, gross_profit, operating_income, free_cash_flow
        FROM financial_facts
        WHERE ticker = ?
        ORDER BY fiscal_year, fiscal_quarter, period
        """,
        (ticker,),
    ).fetchall()
    pair = _comparable_period_pair(rows)
    guidance_score, guidance_note = _score_guidance_events(conn, ticker)
    if not pair:
        note = "Fundamental score has fewer than two comparable source-backed financial periods available."
        if guidance_note:
            note += " " + guidance_note
        else:
            note += " No source-backed guidance events are available."
        return min(20.0, guidance_score), note
    previous, latest = pair
    score = guidance_score
    details: list[str] = []
    score += _trend_points(previous["revenue"], latest["revenue"], 8.0, "revenue", details)
    score += _trend_points(previous["gross_profit"], latest["gross_profit"], 4.0, "gross profit", details)
    score += _trend_points(previous["operating_income"], latest["operating_income"], 4.0, "operating income", details)
    score += _trend_points(previous["free_cash_flow"], latest["free_cash_flow"], 4.0, "free cash flow", details)
    period_note = (
        f"Compared {previous['fiscal_quarter']} {previous['fiscal_year']} "
        f"to {latest['fiscal_quarter']} {latest['fiscal_year']}."
    )
    note = "Fundamental trend evidence: " + period_note + " " + (
        "; ".join(details) if details else "No improving metric detected in comparable loaded periods."
    )
    if guidance_note:
        note += " " + guidance_note
    return min(20.0, score), note


def _score_guidance_events(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT metric, direction, confidence, source_name
        FROM guidance_events
        WHERE ticker = ?
        ORDER BY COALESCE(guidance_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return 0.0, ""
    positive = [row for row in rows if row["direction"] in {"raised", "initiated"}]
    neutral = [row for row in rows if row["direction"] in {"reiterated", "neutral"}]
    negative = [row for row in rows if row["direction"] in {"lowered", "withdrawn"}]
    score = 0.0
    for row in positive:
        score += _guidance_metric_weight(row["metric"]) * float(row["confidence"] or 0)
    source_names = sorted({str(row["source_name"]) for row in rows if row["source_name"]})
    note = (
        f"Guidance context uses {len(rows)} source-backed event(s) "
        f"({len(positive)} positive, {len(neutral)} neutral, {len(negative)} negative)"
    )
    if source_names:
        note += " from " + ", ".join(source_names[:3])
    note += "; guidance events are not treated as realized revenue, orders, or consensus estimate changes."
    return min(6.0, score), note


def _guidance_metric_weight(metric: str) -> float:
    weights = {
        "revenue": 2.5,
        "eps": 2.0,
        "ebitda": 2.0,
        "operating_income": 2.0,
        "gross_margin": 1.8,
        "margin": 1.5,
        "free_cash_flow": 2.0,
        "rpo": 2.2,
        "backlog": 2.2,
        "billings": 2.0,
        "ai_revenue": 2.5,
    }
    return weights.get(str(metric).lower(), 1.5)


def _score_catalysts(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS catalyst_count, COALESCE(MAX(confidence), 0) AS max_confidence
        FROM catalysts
        WHERE ticker = ? AND status IN ('scheduled', 'candidate', 'confirmed')
        """,
        (ticker,),
    ).fetchone()
    count = int(row["catalyst_count"] or 0)
    confidence = float(row["max_confidence"] or 0)
    score = min(10.0, count * 2.0 + confidence * 3.0)
    if count:
        return score, f"Catalyst score uses {count} source-backed catalyst record(s)."
    return 0.0, "Catalyst score is 0 because no source-backed catalyst records are available."


def _score_valuation(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    row = conn.execute(
        """
        SELECT date, price, market_cap, enterprise_value, ev_sales, ev_ebitda,
               pe, fcf_yield, sector_percentile, source
        FROM valuation_snapshots
        WHERE ticker = ?
        ORDER BY date DESC, updated_at DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    if not row:
        peer_score, peer_details = _peer_valuation_context_score(conn, ticker)
        if peer_details:
            return (
                peer_score,
                "Valuation score has no source-backed valuation snapshot yet; "
                + " ".join(peer_details)
                + " Peer comparisons do not by themselves prove undervaluation.",
            )
        return 0.0, "Valuation score is 0 because no source-backed valuation snapshot is available."
    score = 0.0
    details: list[str] = []
    if row["price"] is not None:
        score += 2.0
        details.append("price")
    if row["market_cap"] is not None:
        score += 2.0
        details.append("market cap")
    if row["enterprise_value"] is not None:
        score += 2.0
        details.append("enterprise value")
    ratio_fields = ["ev_sales", "ev_ebitda", "pe", "fcf_yield"]
    ratio_count = sum(1 for field in ratio_fields if row[field] is not None)
    score += min(4.0, ratio_count * 1.5)
    if ratio_count:
        details.append(f"{ratio_count} valuation ratio(s)")
    if row["sector_percentile"] is not None:
        score += 2.0
        details.append("sector percentile")
    context_score, context_details = _valuation_context_score(conn, ticker, row)
    score += context_score
    peer_score, peer_details = _peer_valuation_context_score(conn, ticker)
    score += peer_score
    context_notes = context_details + peer_details
    context_text = (" ".join(context_notes) + " ") if context_notes else ""
    note = (
        f"Valuation snapshot from {row['source']} on {row['date']} includes "
        f"{', '.join(details) if details else 'no populated valuation fields'}; "
        f"{context_text}this does not by itself prove undervaluation."
    )
    return min(15.0, score), note


def _valuation_context_score(
    conn: sqlite3.Connection,
    ticker: str,
    latest: sqlite3.Row,
) -> tuple[float, list[str]]:
    rows = conn.execute(
        """
        SELECT date, ev_sales, ev_ebitda, pe, fcf_yield
        FROM valuation_snapshots
        WHERE ticker = ?
        ORDER BY date, updated_at
        """,
        (ticker,),
    ).fetchall()
    score = 0.0
    details: list[str] = []
    for field, lower_is_supportive in (
        ("ev_sales", True),
        ("ev_ebitda", True),
        ("pe", True),
        ("fcf_yield", False),
    ):
        current = _float_or_none(latest[field])
        values = [_float_or_none(row[field]) for row in rows]
        clean_values = [value for value in values if value is not None]
        if current is None or len(clean_values) < 3:
            continue
        percentile = _percentile_rank(clean_values, current)
        if lower_is_supportive and percentile <= 0.35:
            score += 1.25
            details.append(
                f"{field} is around the {percentile * 100:.0f}th percentile of "
                f"{len(clean_values)} loaded historical snapshot(s)."
            )
        elif not lower_is_supportive and percentile >= 0.65:
            score += 1.25
            details.append(
                f"{field} is around the {percentile * 100:.0f}th percentile of "
                f"{len(clean_values)} loaded historical snapshot(s)."
            )
    sector_percentile = _normalize_percentile(latest["sector_percentile"])
    if sector_percentile is not None and sector_percentile <= 0.35:
        score += 1.0
        details.append(
            f"source-supplied sector_percentile is around {sector_percentile * 100:.0f}%."
        )
    if details:
        details.append("Historical/sector context is a review input, not a valuation conclusion.")
    return min(5.0, score), details


def _peer_valuation_context_score(conn: sqlite3.Connection, ticker: str) -> tuple[float, list[str]]:
    rows = conn.execute(
        """
        SELECT metric, direction, discount_premium_pct, confidence, source_name
        FROM peer_valuation_comparisons
        WHERE ticker = ?
        ORDER BY COALESCE(comparison_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return 0.0, []
    supportive = [row for row in rows if row["direction"] == "supports_discount"]
    neutral = [row for row in rows if row["direction"] == "neutral"]
    contradicting = [row for row in rows if row["direction"] == "contradicts_discount"]
    score = 0.0
    for row in supportive:
        discount = abs(_float_or_none(row["discount_premium_pct"]) or 0.0)
        score += (1.0 + min(discount, 0.5)) * float(row["confidence"] or 0)
    source_names = sorted({str(row["source_name"]) for row in rows if row["source_name"]})
    detail = (
        f"Peer valuation context uses {len(rows)} source-backed comparison(s) "
        f"({len(supportive)} supportive, {len(neutral)} neutral, {len(contradicting)} contradicting)"
    )
    if source_names:
        detail += " from " + ", ".join(source_names[:3])
    detail += "; peer discounts are review inputs, not proof of undervaluation."
    return min(3.0, score), [detail]


def _percentile_rank(values: list[float], current: float) -> float:
    lower = sum(1 for value in values if value < current)
    equal = sum(1 for value in values if value == current)
    return (lower + 0.5 * equal) / len(values)


def _normalize_percentile(value: object) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    if 0 <= parsed <= 1:
        return parsed
    if 1 < parsed <= 100:
        return parsed / 100.0
    return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_expectation_gap(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT signal_type, direction, magnitude, description, confidence, source_name
        FROM expectation_gap_signals
        WHERE ticker = ?
        ORDER BY COALESCE(signal_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    analyst_score, analyst_note = _score_analyst_estimate_events(conn, ticker)
    if not rows:
        if analyst_note:
            return analyst_score, analyst_note
        return (
            0.0,
            "Expectation-gap score is 0 until analyst coverage, media attention, valuation-label evidence, or comparable evidence is loaded.",
        )
    supporting = [row for row in rows if row["direction"] == "supports_gap"]
    neutral = [row for row in rows if row["direction"] == "neutral"]
    contradicting = [row for row in rows if row["direction"] == "contradicts_gap"]
    support_points = sum(_expectation_signal_weight(row["signal_type"]) * float(row["confidence"] or 0) for row in supporting)
    neutral_points = sum(0.6 * float(row["confidence"] or 0) for row in neutral)
    contradiction_points = sum(3.0 * float(row["confidence"] or 0) for row in contradicting)
    score = max(0.0, min(20.0, support_points + neutral_points - contradiction_points))
    source_names = sorted({str(row["source_name"]) for row in rows if row["source_name"]})
    note = (
        f"Expectation-gap score uses {len(rows)} source-backed signal(s) "
        f"({len(supporting)} supporting, {len(neutral)} neutral, {len(contradicting)} contradicting)"
    )
    if source_names:
        note += " from " + ", ".join(source_names[:3])
    note += "; no unsupported market-understanding claims are inferred."
    if analyst_note:
        note += " " + analyst_note
    return min(20.0, score + analyst_score), note


def _score_analyst_estimate_events(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT event_type, direction, metric, confidence, source_name
        FROM analyst_estimate_events
        WHERE ticker = ?
        ORDER BY COALESCE(event_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return 0.0, ""
    supporting = [row for row in rows if row["direction"] == "supports_gap"]
    neutral = [row for row in rows if row["direction"] == "neutral"]
    contradicting = [row for row in rows if row["direction"] == "contradicts_gap"]
    support_points = sum(
        _analyst_estimate_event_weight(row["event_type"]) * float(row["confidence"] or 0)
        for row in supporting
    )
    neutral_points = sum(0.4 * float(row["confidence"] or 0) for row in neutral)
    contradiction_points = sum(3.5 * float(row["confidence"] or 0) for row in contradicting)
    score = max(0.0, min(6.0, support_points + neutral_points - contradiction_points))
    source_names = sorted({str(row["source_name"]) for row in rows if row["source_name"]})
    note = (
        f"Analyst estimate context uses {len(rows)} source-backed event(s) "
        f"({len(supporting)} supporting, {len(neutral)} neutral, {len(contradicting)} contradicting)"
    )
    if source_names:
        note += " from " + ", ".join(source_names[:3])
    note += "; analyst estimates are not inferred, and source-backed events do not by themselves prove a market expectation gap."
    return score, note


def _analyst_estimate_event_weight(event_type: str) -> float:
    weights = {
        "estimate_lag": 4.5,
        "low_analyst_coverage": 4.0,
        "coverage_change": 3.0,
        "estimate_revision": 2.5,
        "price_target_revision": 2.0,
        "rating_change": 2.0,
    }
    return weights.get(str(event_type).lower(), 2.0)


def _expectation_signal_weight(signal_type: str) -> float:
    weights = {
        "low_analyst_coverage": 4.0,
        "low_media_attention": 3.0,
        "legacy_market_label": 4.0,
        "valuation_multiple_lag": 4.0,
        "estimate_lag": 5.0,
        "peer_rerating_gap": 4.0,
        "management_tone_change": 3.0,
        "narrative_change": 2.5,
    }
    return weights.get(str(signal_type).lower(), 2.0)


def _score_market_confirmation(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT signal_type, direction, magnitude, confidence, source_name
        FROM market_confirmation_signals
        WHERE ticker = ?
        ORDER BY COALESCE(signal_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    institutional_score, institutional_note = _score_institutional_holding_events(conn, ticker)
    if not rows:
        if institutional_note:
            return institutional_score, institutional_note
        return (
            0.0,
            "Market-confirmation score is 0 until trend, relative strength, volume, premarket, or options-flow evidence is loaded.",
        )
    bullish = [row for row in rows if row["direction"] == "bullish"]
    neutral = [row for row in rows if row["direction"] == "neutral"]
    bearish = [row for row in rows if row["direction"] == "bearish"]
    max_confidence = max(float(row["confidence"] or 0) for row in rows)
    score = min(10.0, len(bullish) * 1.8 + len(neutral) * 0.5 + max_confidence * 2.5)
    if not bullish and bearish:
        score = 0.0
    source_names = sorted({str(row["source_name"]) for row in rows if row["source_name"]})
    note = (
        f"Market confirmation uses {len(rows)} local/source-backed signal(s) "
        f"({len(bullish)} bullish, {len(neutral)} neutral, {len(bearish)} bearish)"
    )
    if source_names:
        note += " from " + ", ".join(source_names[:3])
    note += "; this is only a confirmation layer, not a core thesis."
    if institutional_note:
        note += " " + institutional_note
    return min(10.0, score + institutional_score), note


def _score_institutional_holding_events(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT event_type, direction, confidence, source_name
        FROM institutional_holding_events
        WHERE ticker = ?
        ORDER BY COALESCE(event_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return 0.0, ""
    bullish = [row for row in rows if row["direction"] == "bullish"]
    neutral = [row for row in rows if row["direction"] == "neutral"]
    bearish = [row for row in rows if row["direction"] == "bearish"]
    score = 0.0
    for row in bullish:
        score += _institutional_event_weight(row["event_type"]) * float(row["confidence"] or 0)
    if not bullish and bearish:
        score = 0.0
    else:
        score = max(0.0, score - 0.8 * len(bearish))
    source_names = sorted({str(row["source_name"]) for row in rows if row["source_name"]})
    note = (
        f"Institutional holding context uses {len(rows)} source-backed event(s) "
        f"({len(bullish)} bullish, {len(neutral)} neutral, {len(bearish)} bearish)"
    )
    if source_names:
        note += " from " + ", ".join(source_names[:3])
    note += "; institutional holding changes are delayed review inputs, not proof of informed buying or selling."
    return min(3.0, score), note


def _institutional_event_weight(event_type: str) -> float:
    weights = {
        "activist_stake": 1.8,
        "new_position": 1.3,
        "increased_position": 1.0,
        "passive_large_holder": 0.8,
        "unchanged_position": 0.3,
    }
    return weights.get(str(event_type).lower(), 0.6)


def _score_risks(conn: sqlite3.Connection, ticker: str) -> tuple[float, str, str]:
    flag_penalty, flag_summary = _score_risk_flags(conn, ticker)
    row = conn.execute(
        """
        SELECT period, cash, debt, operating_cash_flow, free_cash_flow
        FROM financial_facts
        WHERE ticker = ?
          AND (cash IS NOT NULL OR debt IS NOT NULL OR operating_cash_flow IS NOT NULL OR free_cash_flow IS NOT NULL)
        ORDER BY period DESC, updated_at DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    if not row:
        if flag_penalty:
            note = f"Risk penalty {flag_penalty:g} from source-backed non-financial risk flag(s); no source-backed financial risk fields are available yet."
            return flag_penalty, note, flag_summary
        note = "Risk penalty is 0 because no source-backed financial risk fields or source-backed risk flags are available yet."
        summary = "Risk assessment is incomplete until SEC risk factors, financial statements, leverage, customer concentration, dilution, and source-backed red flags are parsed."
        return 0.0, note, summary

    financial_penalty = 0.0
    details: list[str] = []
    free_cash_flow = row["free_cash_flow"]
    operating_cash_flow = row["operating_cash_flow"]
    cash = row["cash"]
    debt = row["debt"]
    if free_cash_flow is not None and free_cash_flow < 0:
        financial_penalty -= 6.0
        details.append(f"negative free cash flow ({free_cash_flow:g})")
    elif operating_cash_flow is not None and operating_cash_flow < 0:
        financial_penalty -= 4.0
        details.append(f"negative operating cash flow ({operating_cash_flow:g})")
    if debt is not None and cash is not None and cash >= 0:
        if cash == 0 and debt > 0:
            financial_penalty -= 8.0
            details.append("debt with no reported cash")
        elif cash > 0 and debt > cash * 2:
            financial_penalty -= 6.0
            details.append(f"debt is more than 2x cash ({debt:g} vs {cash:g})")
        elif cash > 0 and debt > cash:
            financial_penalty -= 3.0
            details.append(f"debt exceeds cash ({debt:g} vs {cash:g})")
    penalty = max(-30.0, financial_penalty + flag_penalty)
    summary_parts: list[str] = []
    if details:
        summary_parts.append("Source-backed financial risk flags: " + "; ".join(details) + ".")
    else:
        summary_parts.append(
            f"No leverage or cash-flow penalty detected in the latest loaded source-backed financial facts for period {row['period']}."
        )
    if flag_summary:
        summary_parts.append(flag_summary)
    elif not details:
        summary_parts.append("Non-financial risks still require parsing.")
    note_sources: list[str] = []
    if financial_penalty:
        note_sources.append(f"source-backed financial facts for period {row['period']}")
    if flag_penalty:
        note_sources.append("source-backed risk flags")
    if note_sources:
        note = f"Risk penalty {penalty:g} from " + " and ".join(note_sources) + "."
        if details:
            note += " Financial flags: " + "; ".join(details) + "."
    else:
        note = f"Risk penalty is 0 from loaded source-backed financial facts for period {row['period']} and no active source-backed risk flags."
    return penalty, note, " ".join(summary_parts)


def _score_risk_flags(conn: sqlite3.Connection, ticker: str) -> tuple[float, str]:
    rows = conn.execute(
        """
        SELECT risk_type, severity, description, confidence, source_name
        FROM risk_flags
        WHERE ticker = ? AND status IN ('active', 'watch')
        ORDER BY COALESCE(risk_date, ''), updated_at
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return 0.0, ""
    penalty = 0.0
    details: list[str] = []
    for row in rows:
        points = _risk_flag_points(row["risk_type"], row["severity"]) * float(row["confidence"] or 0)
        penalty -= points
        details.append(f"{row['severity']} {row['risk_type']}: {row['description']}")
    penalty = max(-30.0, penalty)
    return penalty, "Source-backed non-financial risk flags: " + "; ".join(details[:5]) + "."


def _risk_flag_points(risk_type: str, severity: str) -> float:
    severity_multiplier = {
        "low": 0.6,
        "medium": 1.0,
        "high": 1.6,
        "critical": 2.2,
    }.get(str(severity).lower(), 1.0)
    base_points = {
        "customer_concentration": 4.0,
        "dilution": 4.0,
        "margin_deterioration": 3.5,
        "export_control": 4.5,
        "geopolitical": 4.0,
        "supply_chain": 3.5,
        "accounting_quality": 5.0,
        "management_change": 3.5,
        "management_credibility": 4.5,
        "insider_selling": 3.0,
        "litigation": 3.5,
        "regulatory": 3.5,
        "one_time_order": 3.0,
        "cycle_top": 3.0,
        "no_ai_financial_evidence": 3.5,
    }.get(str(risk_type).lower(), 2.5)
    return base_points * severity_multiplier


def _trend_points(previous: float | None, latest: float | None, points: float, label: str, details: list[str]) -> float:
    if previous is None or latest is None:
        return 0.0
    if latest > previous:
        details.append(f"{label} improved from {previous:g} to {latest:g}")
        return points
    return 0.0


def _comparable_period_pair(rows: list[sqlite3.Row]) -> tuple[sqlite3.Row, sqlite3.Row] | None:
    candidates = [
        row
        for row in rows
        if row["period"] and row["fiscal_year"] is not None and row["fiscal_quarter"]
    ]
    groups: dict[str, dict[int, sqlite3.Row]] = {}
    for row in candidates:
        fiscal_quarter = str(row["fiscal_quarter"])
        fiscal_year = int(row["fiscal_year"])
        year_group = groups.setdefault(fiscal_quarter, {})
        existing = year_group.get(fiscal_year)
        if existing is None or (row["period"] or "") > (existing["period"] or ""):
            year_group[fiscal_year] = row
    comparable_groups = []
    for year_group in groups.values():
        if len(year_group) >= 2:
            comparable_groups.append([year_group[year] for year in sorted(year_group)])
    if not comparable_groups:
        return None
    comparable_groups.sort(key=lambda group: (group[-1]["fiscal_year"], group[-1]["period"] or ""))
    selected = comparable_groups[-1]
    return selected[-2], selected[-1]


def _status_for_score(total: float) -> str:
    if total >= 80:
        return "重点研究"
    if total >= 65:
        return "观察"
    if total >= 50:
        return "等待确认"
    return "排除或暂不跟踪"


def _build_thesis(ticker: str, total: float, notes: list[str]) -> str:
    if total <= 0:
        return f"{ticker}: no source-backed thesis yet. Load SEC filings, financial facts, valuation, catalysts, and risk evidence before drawing conclusions."
    return f"{ticker}: preliminary evidence-backed research score is {total:.2f}. This is a research triage result, not an investment recommendation. " + " ".join(notes[:3])
