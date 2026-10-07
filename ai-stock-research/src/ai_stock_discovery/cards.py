from __future__ import annotations

from pathlib import Path
import sqlite3

from ai_stock_discovery.analysis.scoring import ScoreResult, score_ticker, upsert_research_card
from ai_stock_discovery.research_pool import ResearchPoolRow, build_research_pool
from ai_stock_discovery.timeutils import utc_now_iso


def generate_card(conn: sqlite3.Connection, ticker: str, *, output_dir: Path | None = None) -> str:
    ticker = ticker.upper()
    result = score_ticker(conn, ticker)
    upsert_research_card(conn, result)
    profile = conn.execute(
        "SELECT company_name, sector, industry, ir_url FROM company_profile WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    ai_signals = conn.execute(
        """
        SELECT keyword, ai_relevance_level, confidence, source_url, context_snippet
        FROM ai_relevance_signals
        WHERE ticker = ?
        ORDER BY ai_relevance_level DESC, confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    financial_rows = conn.execute(
        """
        SELECT period, revenue, gross_profit, operating_income, free_cash_flow, source
        FROM financial_facts
        WHERE ticker = ?
        ORDER BY period DESC
        LIMIT 4
        """,
        (ticker,),
    ).fetchall()
    guidance_rows = conn.execute(
        """
        SELECT guidance_date, fiscal_period, metric, previous_value, current_value,
               unit, direction, source_type, source_name, source_url,
               description, confidence
        FROM guidance_events
        WHERE ticker = ?
        ORDER BY COALESCE(guidance_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    transcript_rows = conn.execute(
        """
        SELECT transcript_date, fiscal_period, event_type, speaker, source_name,
               source_url, title, transcript_excerpt, confidence
        FROM transcript_snippets
        WHERE ticker = ?
        ORDER BY COALESCE(transcript_date, ''), confidence DESC
        LIMIT 6
        """,
        (ticker,),
    ).fetchall()
    valuation_rows = conn.execute(
        """
        SELECT date, price, market_cap, enterprise_value, ev_sales, ev_ebitda,
               pe, fcf_yield, sector_percentile, source
        FROM valuation_snapshots
        WHERE ticker = ?
        ORDER BY date DESC, updated_at DESC
        LIMIT 4
        """,
        (ticker,),
    ).fetchall()
    peer_valuation_rows = conn.execute(
        """
        SELECT comparison_date, peer_group, peer_tickers, metric, target_value,
               peer_median, peer_mean, discount_premium_pct, direction,
               source_name, source_url, description, confidence
        FROM peer_valuation_comparisons
        WHERE ticker = ?
        ORDER BY COALESCE(comparison_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    industry_tag_rows = conn.execute(
        """
        SELECT tag, confidence, source_type, source_url, evidence_snippet
        FROM ai_industry_tags
        WHERE ticker = ?
        ORDER BY confidence DESC, tag
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    catalyst_rows = conn.execute(
        """
        SELECT catalyst_type, catalyst_date, description, source_url, confidence, status
        FROM catalysts
        WHERE ticker = ?
        ORDER BY COALESCE(catalyst_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    expectation_gap_rows = conn.execute(
        """
        SELECT signal_date, source_type, source_name, source_url, signal_type,
               direction, magnitude, description, confidence
        FROM expectation_gap_signals
        WHERE ticker = ?
        ORDER BY COALESCE(signal_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    analyst_estimate_rows = conn.execute(
        """
        SELECT event_date, fiscal_period, metric, event_type, direction,
               previous_value, current_value, unit, analyst_firm, source_name,
               source_url, description, confidence
        FROM analyst_estimate_events
        WHERE ticker = ?
        ORDER BY COALESCE(event_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    short_sale_rows = conn.execute(
        """
        SELECT trade_date, market, short_volume, short_exempt_volume,
               total_volume, short_volume_ratio, source_name, source_url
        FROM short_sale_volume
        WHERE ticker = ?
        ORDER BY trade_date DESC, updated_at DESC
        LIMIT 4
        """,
        (ticker,),
    ).fetchall()
    insider_rows = conn.execute(
        """
        SELECT owner_name, relationship, transaction_date, transaction_code,
               acquired_disposed_code, transaction_shares, transaction_price,
               shares_owned_following, source_type, source_name, source_url
        FROM insider_transactions
        WHERE ticker = ?
        ORDER BY transaction_date DESC, updated_at DESC
        LIMIT 6
        """,
        (ticker,),
    ).fetchall()
    institutional_rows = conn.execute(
        """
        SELECT event_date, report_period, filing_type, institution_name,
               event_type, direction, shares_held, shares_change, market_value,
               percent_shares_outstanding, source_name, source_url,
               description, confidence
        FROM institutional_holding_events
        WHERE ticker = ?
        ORDER BY COALESCE(event_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    risk_flag_rows = conn.execute(
        """
        SELECT risk_date, source_type, source_name, source_url, risk_type,
               severity, description, confidence, status
        FROM risk_flags
        WHERE ticker = ? AND status IN ('active', 'watch')
        ORDER BY COALESCE(risk_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    market_signal_rows = conn.execute(
        """
        SELECT signal_date, source_type, source_name, source_path, signal_type,
               direction, magnitude, description, confidence
        FROM market_confirmation_signals
        WHERE ticker = ?
        ORDER BY COALESCE(signal_date, ''), confidence DESC
        LIMIT 8
        """,
        (ticker,),
    ).fetchall()
    macro_rows = conn.execute(
        """
        SELECT series_id, source_name, metric_name, category, geography,
               frequency, period, value, unit, source_url
        FROM macro_indicators
        ORDER BY period DESC, fetched_at DESC, series_id
        LIMIT 8
        """
    ).fetchall()
    sources = _source_links(
        ai_signals,
        financial_rows,
        guidance_rows,
        transcript_rows,
        valuation_rows,
        peer_valuation_rows,
        industry_tag_rows,
        catalyst_rows,
        expectation_gap_rows,
        analyst_estimate_rows,
        short_sale_rows,
        insider_rows,
        institutional_rows,
        risk_flag_rows,
        market_signal_rows,
        macro_rows,
    )
    cards_dir = output_dir or Path("reports/cards")
    research_pool_rows = build_research_pool(conn, tickers=[ticker], limit=1, cards_dir=cards_dir)
    research_pool_row = research_pool_rows[0] if research_pool_rows else None
    company_name = profile["company_name"] if profile and profile["company_name"] else ""
    sector = profile["sector"] if profile and profile["sector"] else ""
    industry = profile["industry"] if profile and profile["industry"] else ""
    content = _render_card(
        result=result,
        company_name=company_name,
        sector=sector,
        industry=industry,
        ai_signals=ai_signals,
        financial_rows=financial_rows,
        guidance_rows=guidance_rows,
        transcript_rows=transcript_rows,
        valuation_rows=valuation_rows,
        peer_valuation_rows=peer_valuation_rows,
        industry_tag_rows=industry_tag_rows,
        catalyst_rows=catalyst_rows,
        expectation_gap_rows=expectation_gap_rows,
        analyst_estimate_rows=analyst_estimate_rows,
        short_sale_rows=short_sale_rows,
        insider_rows=insider_rows,
        institutional_rows=institutional_rows,
        risk_flag_rows=risk_flag_rows,
        market_signal_rows=market_signal_rows,
        macro_rows=macro_rows,
        sources=sources,
        research_pool_row=research_pool_row,
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{ticker}.md").write_text(content, encoding="utf-8")
    return content


def _render_card(
    *,
    result: ScoreResult,
    company_name: str,
    sector: str,
    industry: str,
    ai_signals: list[sqlite3.Row],
    financial_rows: list[sqlite3.Row],
    guidance_rows: list[sqlite3.Row],
    transcript_rows: list[sqlite3.Row],
    valuation_rows: list[sqlite3.Row],
    peer_valuation_rows: list[sqlite3.Row],
    industry_tag_rows: list[sqlite3.Row],
    catalyst_rows: list[sqlite3.Row],
    expectation_gap_rows: list[sqlite3.Row],
    analyst_estimate_rows: list[sqlite3.Row],
    short_sale_rows: list[sqlite3.Row],
    insider_rows: list[sqlite3.Row],
    institutional_rows: list[sqlite3.Row],
    risk_flag_rows: list[sqlite3.Row],
    market_signal_rows: list[sqlite3.Row],
    macro_rows: list[sqlite3.Row],
    sources: list[str],
    research_pool_row: ResearchPoolRow | None,
) -> str:
    ai_level = max([int(row["ai_relevance_level"]) for row in ai_signals], default=0)
    signal_lines = [
        f"- `{row['keyword']}` level {row['ai_relevance_level']} confidence {row['confidence']:.2f}: {row['context_snippet']}"
        for row in ai_signals
    ]
    financial_lines = [
        "- "
        + ", ".join(
            [
                f"period={row['period']}",
                f"revenue={_fmt(row['revenue'])}",
                f"gross_profit={_fmt(row['gross_profit'])}",
                f"operating_income={_fmt(row['operating_income'])}",
                f"free_cash_flow={_fmt(row['free_cash_flow'])}",
            ]
        )
        for row in financial_rows
    ]
    guidance_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['guidance_date'] or 'date NA'}",
                f"period={row['fiscal_period'] or 'period NA'}",
                f"metric={row['metric']}",
                f"direction={row['direction']}",
                f"previous={_fmt(row['previous_value'])}",
                f"current={_fmt(row['current_value'])}",
                f"unit={row['unit'] or 'NA'}",
                f"confidence={row['confidence']:.2f}",
                f"description={row['description']}",
            ]
        )
        for row in guidance_rows
    ]
    transcript_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['transcript_date'] or 'date NA'}",
                f"period={row['fiscal_period'] or 'period NA'}",
                f"event={row['event_type']}",
                f"speaker={row['speaker'] or 'NA'}",
                f"confidence={row['confidence']:.2f}",
                f"excerpt={_truncate(row['transcript_excerpt'], 500)}",
            ]
        )
        for row in transcript_rows
    ]
    valuation_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['date']}",
                f"price={_fmt(row['price'])}",
                f"market_cap={_fmt(row['market_cap'])}",
                f"enterprise_value={_fmt(row['enterprise_value'])}",
                f"ev_sales={_fmt(row['ev_sales'])}",
                f"ev_ebitda={_fmt(row['ev_ebitda'])}",
                f"pe={_fmt(row['pe'])}",
                f"fcf_yield={_fmt(row['fcf_yield'])}",
                f"sector_percentile={_fmt(row['sector_percentile'])}",
                f"source={row['source']}",
            ]
        )
        for row in valuation_rows
    ]
    peer_valuation_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['comparison_date'] or 'date NA'}",
                f"group={row['peer_group'] or 'NA'}",
                f"peers={row['peer_tickers']}",
                f"metric={row['metric']}",
                f"target={_fmt(row['target_value'])}",
                f"peer_median={_fmt(row['peer_median'])}",
                f"discount_premium_pct={_fmt(row['discount_premium_pct'])}",
                f"direction={row['direction']}",
                f"confidence={row['confidence']:.2f}",
                f"description={row['description']}",
            ]
        )
        for row in peer_valuation_rows
    ]
    industry_tag_lines = [
        f"- `{row['tag']}` confidence {row['confidence']:.2f} "
        f"from {row['source_type']}: {row['evidence_snippet']}"
        for row in industry_tag_rows
    ]
    catalyst_lines = [
        f"- `{row['catalyst_type']}` {row['catalyst_date'] or 'date NA'} "
        f"confidence {row['confidence']:.2f} status {row['status']}: {row['description']}"
        for row in catalyst_rows
    ]
    expectation_gap_lines = [
        f"- `{row['signal_type']}` {row['signal_date'] or 'date NA'} "
        f"{row['direction']} confidence {row['confidence']:.2f}: {row['description']}"
        for row in expectation_gap_rows
    ]
    analyst_estimate_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['event_date'] or 'date NA'}",
                f"period={row['fiscal_period'] or 'period NA'}",
                f"type={row['event_type']}",
                f"direction={row['direction']}",
                f"metric={row['metric'] or 'NA'}",
                f"previous={_fmt(row['previous_value'])}",
                f"current={_fmt(row['current_value'])}",
                f"unit={row['unit'] or 'NA'}",
                f"firm={row['analyst_firm'] or row['source_name']}",
                f"confidence={row['confidence']:.2f}",
                f"description={row['description']}",
            ]
        )
        for row in analyst_estimate_rows
    ]
    short_sale_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['trade_date']}",
                f"market={row['market'] or 'NA'}",
                f"short_volume={_fmt(row['short_volume'])}",
                f"short_exempt_volume={_fmt(row['short_exempt_volume'])}",
                f"total_volume={_fmt(row['total_volume'])}",
                f"ratio={_fmt(row['short_volume_ratio'])}",
            ]
        )
        for row in short_sale_rows
    ]
    insider_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['transaction_date']}",
                f"owner={row['owner_name']}",
                f"relationship={row['relationship'] or 'NA'}",
                f"code={row['transaction_code'] or 'NA'}",
                f"acquired_disposed={row['acquired_disposed_code'] or 'NA'}",
                f"shares={_fmt(row['transaction_shares'])}",
                f"price={_fmt(row['transaction_price'])}",
                f"owned_following={_fmt(row['shares_owned_following'])}",
            ]
        )
        for row in insider_rows
    ]
    institutional_lines = [
        "- "
        + ", ".join(
            [
                f"date={row['event_date'] or 'date NA'}",
                f"period={row['report_period'] or 'period NA'}",
                f"filing={row['filing_type'] or 'NA'}",
                f"institution={row['institution_name']}",
                f"event={row['event_type']}",
                f"direction={row['direction']}",
                f"shares_held={_fmt(row['shares_held'])}",
                f"shares_change={_fmt(row['shares_change'])}",
                f"market_value={_fmt(row['market_value'])}",
                f"ownership_pct={_fmt(row['percent_shares_outstanding'])}",
                f"confidence={row['confidence']:.2f}",
                f"description={row['description']}",
            ]
        )
        for row in institutional_rows
    ]
    market_signal_lines = [
        f"- `{row['signal_type']}` {row['signal_date'] or 'date NA'} "
        f"{row['direction']} confidence {row['confidence']:.2f}: {row['description']}"
        for row in market_signal_rows
    ]
    macro_lines = [
        f"- `{row['series_id']}` {row['period']} {row['metric_name']}="
        f"{_fmt(row['value'])}{(' ' + row['unit']) if row['unit'] else ''} "
        f"source={row['source_name']} geography={row['geography'] or 'NA'}"
        for row in macro_rows
    ]
    risk_flag_lines = [
        f"- `{row['risk_type']}` {row['risk_date'] or 'date NA'} "
        f"severity {row['severity']} confidence {row['confidence']:.2f}: {row['description']}"
        for row in risk_flag_rows
    ]
    research_pool_lines = _research_pool_lines(research_pool_row)
    tracking_frequency = (
        research_pool_row.tracking_frequency
        if research_pool_row
        else "自动证据更新后复核；高分候选需要人工核验来源。"
    )
    next_source_action = (
        f"- 下一步来源动作: {research_pool_row.next_action}"
        if research_pool_row
        else "- 下一步来源动作: 先补齐 SEC/IR/公告/API 或本地来源证据。"
    )
    source_lines = [f"- {source}" for source in sources] or ["- 暂无来源；不应形成公司级结论。"]
    return f"""# {result.ticker} 研究卡片

> 研究辅助，不是投资建议。所有结论必须回到来源核验。

Ticker: {result.ticker}
公司名称: {company_name}
行业/AI 产业链位置: {sector} / {industry}
AI 相关性等级: {ai_level}

## AI 产业链标签

{chr(10).join(industry_tag_lines) if industry_tag_lines else "- 暂无 AI 产业链标签；可运行 `tag-ai-chain` 或导入人工标签。"}

## 初版评分

- 总分: {result.score_total}
- 当前状态: {result.status}
- AI 真实相关性: {result.ai_relevance_score}
- 预期差/关注度不足: {result.expectation_gap_score}
- 基本面改善: {result.fundamental_score}
- 估值修复空间: {result.valuation_score}
- 催化剂明确度: {result.catalyst_score}
- 市场行为确认: {result.market_confirmation_score}
- 风险惩罚: {result.risk_penalty}

## 研究池与来源链状态

{chr(10).join(research_pool_lines)}

## 核心 Thesis

{result.thesis}

## 为什么市场可能低估它

{chr(10).join(expectation_gap_lines) if expectation_gap_lines else "暂无可验证预期差证据。需要补充分析师覆盖、媒体关注度、估值标签变化、同行比较或公司口径变化来源。"}

## 分析师预期/覆盖事件

{chr(10).join(analyst_estimate_lines) if analyst_estimate_lines else "- 暂无来源支持的分析师预期变化或覆盖度事件。不得凭空判断一致预期或分析师态度。"}

## 基本面改善证据

{chr(10).join(financial_lines) if financial_lines else "- 暂无 source-backed financial facts 抽取结果。"}

## 管理层指引事件

{chr(10).join(guidance_lines) if guidance_lines else "- 暂无来源支持的管理层指引事件。指引事件不是已实现收入、订单或共识预期变化。"}

## AI 相关证据

{chr(10).join(signal_lines) if signal_lines else "- 暂无 SEC/IR/公告来源中的 AI 关键词上下文。"}

## 电话会/纪要片段

{chr(10).join(transcript_lines) if transcript_lines else "- 暂无来源支持的电话会或会议纪要片段。片段仅供回到原文核验，不等同完整 transcript 解析。"}

## 估值是否仍有空间

{chr(10).join(valuation_lines) if valuation_lines else "- 暂无估值快照或同行分位证据。不得凭空判断便宜或昂贵。"}

## 同行估值比较

{chr(10).join(peer_valuation_lines) if peer_valuation_lines else "- 暂无来源支持的同行估值比较。同行折价/溢价必须回到来源复核。"}

## 未来 3-12 个月催化剂

{chr(10).join(catalyst_lines) if catalyst_lines else "- 暂无来源支持的催化剂记录。"}

## 市场行为确认

{chr(10).join(market_signal_lines) if market_signal_lines else "- 暂无成交量、相对强度、盘前确认或期权流等来源支持的确认信号。"}

## 宏观/电力背景

{chr(10).join(macro_lines) if macro_lines else "- 暂无 FRED/EIA 或本地宏观、电力背景指标。注意：宏观/电力数据不是公司级订单、客户或收入证据。"}

注意：宏观/电力数据不是公司级订单、客户或收入证据，只能作为背景变量。

## 短卖成交量辅助

{chr(10).join(short_sale_lines) if short_sale_lines else "- 暂无 FINRA short sale volume 记录。注意：FINRA daily short sale volume 不等于 short interest。"}

## Insider/Form 4 events

{chr(10).join(insider_lines) if insider_lines else "- No source-backed insider transaction events are loaded."}

Note: insider transaction events are review inputs only; a single event is not automatically a risk conclusion.

## 机构持仓变化事件

{chr(10).join(institutional_lines) if institutional_lines else "- 暂无来源支持的机构持仓变化事件。13F/13D/G 等持仓信息有披露延迟，不等同机构实时观点。"}

Note: institutional holding events are delayed review inputs only; they are not proof of informed buying or selling.

## 需要验证的数据

- SEC filings 中 AI、data center、GPU、liquid cooling、power、automation、cybersecurity 等上下文是否对应收入、订单、RPO、backlog、毛利或指引。
- 最近多个财务周期的收入、利润率、现金流、capex、债务趋势。
- 估值指标与历史分位、同行折价/溢价。
- 催化剂日期和来源：财报、新产品、订单交付、行业会议、监管事件等。
{next_source_action}

## 主要风险

{result.risk_summary}

{chr(10).join(risk_flag_lines) if risk_flag_lines else "- 暂无来源支持的非财务风险红旗记录。"}

## 什么情况说明 thesis 错了

{result.invalidating_conditions}

## 跟踪频率

{tracking_frequency}

## 来源链接

{chr(10).join(source_lines)}

最后更新时间: {utc_now_iso()}
"""


def _source_links(
    ai_signals: list[sqlite3.Row],
    financial_rows: list[sqlite3.Row],
    guidance_rows: list[sqlite3.Row],
    transcript_rows: list[sqlite3.Row],
    valuation_rows: list[sqlite3.Row],
    peer_valuation_rows: list[sqlite3.Row],
    industry_tag_rows: list[sqlite3.Row],
    catalyst_rows: list[sqlite3.Row],
    expectation_gap_rows: list[sqlite3.Row],
    analyst_estimate_rows: list[sqlite3.Row],
    short_sale_rows: list[sqlite3.Row],
    insider_rows: list[sqlite3.Row],
    institutional_rows: list[sqlite3.Row],
    risk_flag_rows: list[sqlite3.Row],
    market_signal_rows: list[sqlite3.Row],
    macro_rows: list[sqlite3.Row],
) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for row in ai_signals:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in financial_rows:
        url = row["source"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in guidance_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in transcript_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in valuation_rows:
        url = row["source"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in peer_valuation_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in industry_tag_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in catalyst_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in expectation_gap_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in analyst_estimate_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in short_sale_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in insider_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in institutional_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in risk_flag_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in market_signal_rows:
        url = row["source_path"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    for row in macro_rows:
        url = row["source_url"]
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


def _research_pool_lines(row: ResearchPoolRow | None) -> list[str]:
    if row is None:
        return [
            "- 研究池分层: 未生成。",
            "- 来源链 gate: 未生成；不得据此形成公司级 thesis。",
        ]
    return [
        f"- 研究池分层: {row.score_layer}",
        f"- 研究池状态: {row.research_pool_status}",
        f"- Thesis gate: {row.thesis_gate}",
        f"- 人工复核 bucket: {row.review_bucket}",
        f"- 人工复核优先级: {row.review_priority}",
        f"- 证据覆盖分: {row.evidence_coverage_score:g}",
        f"- 来源链接数量: {row.source_link_count}",
        f"- 下一步来源动作: {row.next_action}",
        f"- 备注: {row.pool_notes}",
    ]


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:g}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 4].rstrip() + " ..."
