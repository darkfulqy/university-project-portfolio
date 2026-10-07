from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
import sqlite3
from typing import Any
from urllib.parse import urljoin

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.timeutils import utc_now_iso


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


FINANCIAL_CONCEPTS = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit", "USD")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss", "USD")],
    "net_income": [
        ("us-gaap", "NetIncomeLoss", "USD"),
        ("us-gaap", "ProfitLoss", "USD"),
    ],
    "eps": [
        ("us-gaap", "EarningsPerShareDiluted", "USD/shares"),
        ("us-gaap", "EarningsPerShareBasic", "USD/shares"),
    ],
    "operating_cash_flow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD")
    ],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets", "USD"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "USD"),
    ],
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding", "shares"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
    ],
}

DEBT_TOTAL_CONCEPTS = [
    ("us-gaap", "LongTermDebt", "USD"),
    ("us-gaap", "DebtAndFinanceLeaseObligations", "USD"),
    ("us-gaap", "LongTermDebtAndFinanceLeaseObligations", "USD"),
]

DEBT_PART_CONCEPTS = {
    "current": [
        ("us-gaap", "LongTermDebtCurrent", "USD"),
        ("us-gaap", "DebtCurrent", "USD"),
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent", "USD"),
    ],
    "noncurrent": [
        ("us-gaap", "LongTermDebtNoncurrent", "USD"),
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "USD"),
    ],
}


@dataclass(frozen=True)
class FilingDocument:
    ticker: str
    cik: str
    form: str
    filed_at: str
    accession_number: str
    document_url: str
    text: str


def cik10(cik: str | int) -> str:
    return str(cik).strip().zfill(10)


def sync_company_tickers(conn: sqlite3.Connection, client: HttpClient) -> int:
    payload = client.get_json(COMPANY_TICKERS_URL)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected SEC company_tickers payload")
    now = utc_now_iso()
    rows = []
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        title = str(item.get("title", "")).strip()
        cik = cik10(str(item.get("cik_str", "")))
        if ticker and cik:
            rows.append((ticker, cik, title, "SEC company_tickers", now))
    conn.executemany(
        """
        INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            cik=excluded.cik,
            company_name=COALESCE(company_profile.company_name, excluded.company_name),
            source=excluded.source,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    return len(rows)


def get_cik(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        "SELECT cik FROM company_profile WHERE ticker = ?",
        (ticker.upper(),),
    ).fetchone()
    return row["cik"] if row and row["cik"] else None


def fetch_submissions(client: HttpClient, cik: str) -> dict[str, Any]:
    payload = client.get_json(SUBMISSIONS_URL.format(cik=cik10(cik)))
    if not isinstance(payload, dict):
        raise ValueError("Unexpected SEC submissions payload")
    return payload


def fetch_companyfacts(client: HttpClient, cik: str) -> dict[str, Any]:
    payload = client.get_json(COMPANYFACTS_URL.format(cik=cik10(cik)))
    if not isinstance(payload, dict):
        raise ValueError("Unexpected SEC companyfacts payload")
    return payload


def upsert_recent_filings(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    cik: str,
    submissions: dict[str, Any],
    forms: set[str] | None = None,
    limit: int | None = None,
) -> int:
    filings = list_recent_filings(ticker=ticker, cik=cik, submissions=submissions, forms=forms)
    if limit:
        filings = filings[:limit]
    return upsert_filings(conn, filings)


def upsert_filings(conn: sqlite3.Connection, filings: list[dict[str, str]]) -> int:
    now = utc_now_iso()
    rows = [
        (
            filing["ticker"],
            filing["cik"],
            filing["form"],
            filing["filed_at"],
            filing.get("period"),
            filing["accession_number"],
            filing["filing_url"],
            filing["document_url"],
            "new",
            now,
        )
        for filing in filings
    ]
    conn.executemany(
        """
        INSERT INTO filings (
            ticker, cik, form, filed_at, period, accession_number,
            filing_url, document_url, parsed_status, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, accession_number) DO UPDATE SET
            form=excluded.form,
            filed_at=excluded.filed_at,
            period=excluded.period,
            filing_url=excluded.filing_url,
            document_url=excluded.document_url,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    return len(rows)


def enqueue_recent_filings(
    conn: sqlite3.Connection,
    filings: list[dict[str, str]],
    *,
    source: str = "SEC submissions",
) -> int:
    now = utc_now_iso()
    changed = 0
    for filing in filings:
        filing_url = filing.get("filing_url") or ""
        if not filing_url:
            continue
        document_url = filing.get("document_url") or None
        existing = conn.execute(
            "SELECT id FROM filing_queue WHERE filing_url = ? LIMIT 1",
            (filing_url,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE filing_queue
                SET ticker = ?, cik = ?, form = ?, accession_number = ?,
                    document_url = COALESCE(?, document_url)
                WHERE id = ?
                """,
                (
                    filing["ticker"],
                    filing["cik"],
                    filing["form"],
                    filing["accession_number"],
                    document_url,
                    existing["id"],
                ),
            )
            changed += 1
            continue
        conn.execute(
            """
            INSERT INTO filing_queue (
                ticker, cik, form, accession_number, filing_url, document_url,
                source, status, queued_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)
            ON CONFLICT(filing_url, source) DO UPDATE SET
                ticker=excluded.ticker,
                cik=excluded.cik,
                form=excluded.form,
                accession_number=excluded.accession_number,
                document_url=excluded.document_url
            """,
            (
                filing["ticker"],
                filing["cik"],
                filing["form"],
                filing["accession_number"],
                filing_url,
                document_url,
                source,
                now,
            ),
        )
        changed += 1
    return changed


def list_recent_filings(
    *,
    ticker: str,
    cik: str,
    submissions: dict[str, Any],
    forms: set[str] | None = None,
) -> list[dict[str, str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return []
    form_values = recent.get("form", [])
    accession_values = recent.get("accessionNumber", [])
    filed_values = recent.get("filingDate", [])
    report_values = recent.get("reportDate", [])
    document_values = recent.get("primaryDocument", [])
    cik_int = str(int(cik10(cik)))
    filings: list[dict[str, str]] = []
    for index, form in enumerate(form_values):
        if forms and form not in forms:
            continue
        accession = _safe_list_get(accession_values, index)
        primary_doc = _safe_list_get(document_values, index)
        if not accession or not primary_doc:
            continue
        accession_compact = accession.replace("-", "")
        document_url = f"{ARCHIVES_BASE_URL}/{cik_int}/{accession_compact}/{primary_doc}"
        filing_url = f"{ARCHIVES_BASE_URL}/{cik_int}/{accession_compact}/{accession}-index.html"
        filings.append(
            {
                "ticker": ticker.upper(),
                "cik": cik10(cik),
                "form": str(form),
                "filed_at": _safe_list_get(filed_values, index),
                "period": _safe_list_get(report_values, index),
                "accession_number": accession,
                "filing_url": filing_url,
                "document_url": document_url,
            }
        )
    return filings


def fetch_filing_document(
    client: HttpClient,
    *,
    ticker: str,
    cik: str,
    form: str,
    filed_at: str,
    accession_number: str,
    document_url: str,
) -> FilingDocument:
    html = client.get_text(document_url, accept="text/html,application/xhtml+xml,text/plain")
    return FilingDocument(
        ticker=ticker.upper(),
        cik=cik10(cik),
        form=form,
        filed_at=filed_at,
        accession_number=accession_number,
        document_url=document_url,
        text=html_to_text(html),
    )


def resolve_primary_document_url(
    client: HttpClient,
    *,
    filing_url: str,
    accession_number: str | None = None,
) -> str:
    if not filing_url.endswith("-index.htm") and not filing_url.endswith("-index.html"):
        return filing_url
    html = client.get_text(filing_url, accept="text/html,application/xhtml+xml,text/plain")
    links = _extract_archive_links(html, filing_url)
    if accession_number:
        compact = accession_number.replace("-", "")
        for link in links:
            if compact in link and not link.endswith(("-index.htm", "-index.html")):
                return link
    for link in links:
        lower = link.lower()
        if lower.endswith((".htm", ".html", ".xml")) and not lower.endswith(("-index.htm", "-index.html")):
            return link
    raise ValueError(f"Could not resolve primary document from SEC filing index: {filing_url}")


def extract_financial_facts(
    *,
    ticker: str,
    cik: str,
    companyfacts: dict[str, Any],
    max_periods: int = 16,
) -> list[dict[str, Any]]:
    facts = companyfacts.get("facts", {})
    period_map: dict[tuple[str, int | None, str | None], dict[str, Any]] = {}
    for metric, candidates in FINANCIAL_CONCEPTS.items():
        for taxonomy, concept, unit in candidates:
            values = _fact_values(facts, taxonomy, concept, unit)
            if not values:
                continue
            _merge_metric(period_map, metric, values)
            break
    total_debt_values = []
    for taxonomy, concept, unit in DEBT_TOTAL_CONCEPTS:
        total_debt_values = _fact_values(facts, taxonomy, concept, unit)
        if total_debt_values:
            _merge_metric(period_map, "debt", total_debt_values)
            break
    if not total_debt_values:
        current_values = _first_available_fact_values(facts, DEBT_PART_CONCEPTS["current"])
        noncurrent_values = _first_available_fact_values(facts, DEBT_PART_CONCEPTS["noncurrent"])
        _merge_debt_parts(period_map, current_values, noncurrent_values)

    rows = []
    source_url = COMPANYFACTS_URL.format(cik=cik10(cik))
    for key, values in period_map.items():
        period, fy, fp = key
        if not period:
            continue
        operating_cash_flow = values.get("operating_cash_flow")
        capex = values.get("capex")
        free_cash_flow = None
        if operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow - abs(capex)
        rows.append(
            {
                "ticker": ticker.upper(),
                "period": period,
                "fiscal_year": fy,
                "fiscal_quarter": fp,
                "revenue": values.get("revenue"),
                "gross_profit": values.get("gross_profit"),
                "operating_income": values.get("operating_income"),
                "net_income": values.get("net_income"),
                "eps": values.get("eps"),
                "operating_cash_flow": operating_cash_flow,
                "free_cash_flow": free_cash_flow,
                "capex": capex,
                "cash": values.get("cash"),
                "debt": values.get("debt"),
                "shares_outstanding": values.get("shares_outstanding"),
                "source": source_url,
                "updated_at": utc_now_iso(),
            }
        )
    rows.sort(key=lambda row: (row["period"] or "", row["fiscal_quarter"] or ""))
    return rows[-max_periods:]


def upsert_financial_facts(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    conn.executemany(
        """
        INSERT INTO financial_facts (
            ticker, period, fiscal_year, fiscal_quarter, revenue, gross_profit,
            operating_income, net_income, eps, operating_cash_flow, free_cash_flow,
            capex, cash, debt, shares_outstanding, source, updated_at
        )
        VALUES (
            :ticker, :period, :fiscal_year, :fiscal_quarter, :revenue, :gross_profit,
            :operating_income, :net_income, :eps, :operating_cash_flow, :free_cash_flow,
            :capex, :cash, :debt, :shares_outstanding, :source, :updated_at
        )
        ON CONFLICT(ticker, period, fiscal_year, fiscal_quarter, source) DO UPDATE SET
            revenue=excluded.revenue,
            gross_profit=excluded.gross_profit,
            operating_income=excluded.operating_income,
            net_income=excluded.net_income,
            eps=excluded.eps,
            operating_cash_flow=excluded.operating_cash_flow,
            free_cash_flow=excluded.free_cash_flow,
            capex=excluded.capex,
            cash=excluded.cash,
            debt=excluded.debt,
            shares_outstanding=excluded.shares_outstanding,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    return len(rows)


def mark_source_status(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    status: str,
    reason: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO data_source_status (source_name, status, reason, last_checked_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_name) DO UPDATE SET
            status=excluded.status,
            reason=excluded.reason,
            last_checked_at=excluded.last_checked_at
        """,
        (source_name, status, reason, utc_now_iso()),
    )


def html_to_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return re.sub(r"\s+", " ", parser.text()).strip()


def _extract_archive_links(html: str, base_url: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(html)
    links: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        if "/Archives/edgar/data/" not in href and not href.startswith("/ixviewer/doc/action"):
            continue
        absolute = urljoin(base_url, href)
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def _fact_values(
    facts: dict[str, Any],
    taxonomy: str,
    concept: str,
    unit: str,
) -> list[dict[str, Any]]:
    concept_payload = facts.get(taxonomy, {}).get(concept, {})
    unit_values = concept_payload.get("units", {}).get(unit, [])
    if not isinstance(unit_values, list):
        return []
    accepted_forms = {"10-K", "10-Q", "20-F", "40-F"}
    return [
        value
        for value in unit_values
        if isinstance(value, dict)
        and value.get("form") in accepted_forms
        and value.get("end")
        and value.get("val") is not None
    ]


def _first_available_fact_values(
    facts: dict[str, Any],
    candidates: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    for taxonomy, concept, unit in candidates:
        values = _fact_values(facts, taxonomy, concept, unit)
        if values:
            return values
    return []


def _merge_metric(
    period_map: dict[tuple[str, int | None, str | None], dict[str, Any]],
    metric: str,
    values: list[dict[str, Any]],
) -> None:
    for value in values:
        key = _period_key(value)
        current = period_map.setdefault(key, {})
        current[metric] = _numeric(value.get("val"))


def _merge_debt_parts(
    period_map: dict[tuple[str, int | None, str | None], dict[str, Any]],
    current_values: list[dict[str, Any]],
    noncurrent_values: list[dict[str, Any]],
) -> None:
    current_by_key = {_period_key(value): _numeric(value.get("val")) for value in current_values}
    noncurrent_by_key = {_period_key(value): _numeric(value.get("val")) for value in noncurrent_values}
    for key in set(current_by_key) | set(noncurrent_by_key):
        total = (current_by_key.get(key) or 0) + (noncurrent_by_key.get(key) or 0)
        if total:
            period_map.setdefault(key, {})["debt"] = total


def _period_key(value: dict[str, Any]) -> tuple[str, int | None, str | None]:
    fy = value.get("fy")
    try:
        fy_int = int(fy) if fy is not None else None
    except (TypeError, ValueError):
        fy_int = None
    fp = str(value.get("fp")) if value.get("fp") is not None else None
    return str(value.get("end") or ""), fy_int, fp


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_list_get(values: list[Any], index: int) -> str:
    if index >= len(values):
        return ""
    value = values[index]
    return "" if value is None else str(value)


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "table"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "table"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def text(self) -> str:
        return " ".join(self._parts)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)
