from __future__ import annotations

from typing import Any

from ai_stock_discovery.http import FetchError, HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


FMP_FINANCIAL_STATEMENTS_SOURCE_NAME = "Financial Modeling Prep financial statements API"
FMP_FINANCIAL_STATEMENTS_BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_fmp_financial_facts(
    client: HttpClient,
    ticker: str,
    api_key: str,
    *,
    period: str = "annual",
    limit: int = 4,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    period = period.strip().lower()
    if period not in {"annual", "quarter"}:
        raise ValueError("period must be annual or quarter.")
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required.")

    income_rows = _fetch_statement(client, "income-statement", ticker=ticker, api_key=api_key, period=period, limit=limit)
    balance_rows = _fetch_statement(client, "balance-sheet-statement", ticker=ticker, api_key=api_key, period=period, limit=limit)
    cash_flow_rows = _fetch_statement(client, "cash-flow-statement", ticker=ticker, api_key=api_key, period=period, limit=limit)
    if not income_rows:
        raise ValueError(f"No FMP income statement rows returned for {ticker}.")

    balance_by_key = {_statement_key(row): row for row in balance_rows if _statement_key(row)}
    cash_flow_by_key = {_statement_key(row): row for row in cash_flow_rows if _statement_key(row)}
    source_url = _combined_source_url(ticker=ticker, period=period, limit=limit)
    now = utc_now_iso()
    facts: list[dict[str, Any]] = []
    for income in income_rows[:limit]:
        key = _statement_key(income)
        if not key:
            continue
        date_value, fiscal_year, fiscal_quarter = key
        balance = balance_by_key.get(key, {})
        cash_flow = cash_flow_by_key.get(key, {})
        operating_cash_flow = _number(
            _first_present(cash_flow, "operatingCashFlow", "netCashProvidedByOperatingActivities")
        )
        capex = _positive_abs(_number(_first_present(cash_flow, "capitalExpenditure", "capitalExpenditures")))
        free_cash_flow = _number(_first_present(cash_flow, "freeCashFlow"))
        if free_cash_flow is None and operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow - capex
        facts.append(
            {
                "ticker": ticker,
                "period": date_value,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "revenue": _number(_first_present(income, "revenue")),
                "gross_profit": _number(_first_present(income, "grossProfit")),
                "operating_income": _number(_first_present(income, "operatingIncome")),
                "net_income": _number(_first_present(income, "netIncome", "bottomLineNetIncome")),
                "eps": _number(_first_present(income, "epsDiluted", "eps")),
                "operating_cash_flow": operating_cash_flow,
                "free_cash_flow": free_cash_flow,
                "capex": capex,
                "cash": _number(_first_present(balance, "cashAndCashEquivalents", "cashAndShortTermInvestments")),
                "debt": _debt(balance),
                "shares_outstanding": _number(
                    _first_present(income, "weightedAverageShsOutDil", "weightedAverageShsOut")
                ),
                "source": source_url,
                "updated_at": now,
            }
        )
    if not facts:
        raise ValueError(f"No usable FMP financial statement rows returned for {ticker}.")
    facts.sort(key=lambda row: (row["period"] or "", row["fiscal_quarter"] or ""))
    return facts[-limit:]


def _fetch_statement(
    client: HttpClient,
    statement: str,
    *,
    ticker: str,
    api_key: str,
    period: str,
    limit: int,
) -> list[dict[str, Any]]:
    url = _statement_url(statement, ticker=ticker, period=period, limit=limit, api_key=api_key)
    try:
        payload = client.get_json(url)
    except FetchError as exc:
        raise FetchError(str(exc).replace(api_key, "***")) from exc
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected FMP {statement} payload for {ticker}.")
    return [row for row in payload if isinstance(row, dict)]


def _statement_url(statement: str, *, ticker: str, period: str, limit: int, api_key: str) -> str:
    return (
        f"{FMP_FINANCIAL_STATEMENTS_BASE_URL}/{statement}"
        f"?symbol={ticker}&period={period}&limit={limit}&apikey={api_key}"
    )


def _combined_source_url(*, ticker: str, period: str, limit: int) -> str:
    return ";".join(
        _statement_url(statement, ticker=ticker, period=period, limit=limit, api_key="***")
        for statement in ("income-statement", "balance-sheet-statement", "cash-flow-statement")
    )


def _statement_key(row: dict[str, Any]) -> tuple[str, int | None, str | None] | None:
    date_value = _text(row.get("date"))
    if not date_value:
        return None
    fiscal_year = _int(row.get("fiscalYear") or row.get("calendarYear"))
    fiscal_quarter = _text(row.get("period")) or None
    return date_value, fiscal_year, fiscal_quarter


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _debt(balance: dict[str, Any]) -> float | None:
    total = _number(_first_present(balance, "totalDebt", "debtAndFinanceLeaseObligations"))
    if total is not None:
        return total
    short_term = _number(_first_present(balance, "shortTermDebt", "shortTermDebtAndCurrentMaturities"))
    long_term = _number(_first_present(balance, "longTermDebt", "longTermDebtNonCurrent"))
    if short_term is None and long_term is None:
        return None
    return (short_term or 0.0) + (long_term or 0.0)


def _number(value: Any) -> float | None:
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


def _positive_abs(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(value)


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
