from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
from urllib.parse import urlencode

from ai_stock_discovery.http import HttpClient
from ai_stock_discovery.sources.news_rss import NewsEvent


GDELT_DOC_API_SOURCE_NAME = "GDELT DOC API"
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DOC_API_DOCS_URL = "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/"


@dataclass(frozen=True)
class GdeltQueryResult:
    query: str
    events: list[NewsEvent]
    source_url: str


def build_gdelt_doc_url(
    *,
    query: str,
    limit: int = 25,
    timespan: str = "7d",
    sort: str = "datedesc",
) -> str:
    if not query.strip():
        raise ValueError("GDELT query must not be blank.")
    params = {
        "query": query.strip(),
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max(1, min(limit, 250))),
        "timespan": timespan.strip() or "7d",
        "sort": sort.strip() or "datedesc",
    }
    return GDELT_DOC_API_URL + "?" + urlencode(params)


def fetch_gdelt_doc_news(
    client: HttpClient,
    *,
    query: str,
    limit: int = 25,
    timespan: str = "7d",
    sort: str = "datedesc",
    source_name: str = GDELT_DOC_API_SOURCE_NAME,
) -> GdeltQueryResult:
    source_url = build_gdelt_doc_url(query=query, limit=limit, timespan=timespan, sort=sort)
    payload = client.get_json(source_url)
    events = parse_gdelt_doc_articles(payload, source_name=source_name, limit=limit)
    return GdeltQueryResult(query=query, events=events, source_url=source_url)


def parse_gdelt_doc_articles(
    payload: object,
    *,
    source_name: str = GDELT_DOC_API_SOURCE_NAME,
    limit: int = 25,
) -> list[NewsEvent]:
    if not isinstance(payload, dict):
        raise ValueError("Unexpected GDELT DOC API payload.")
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("Unexpected GDELT DOC API articles payload.")
    events: list[NewsEvent] = []
    for article in articles[:limit]:
        if not isinstance(article, dict):
            continue
        title = _text(article, "title")
        url = _text(article, "url")
        if not title or not url:
            continue
        published_at = _text(article, "seendate", "date", "datetime") or None
        summary = _summary(article)
        content_hash = hashlib.sha256(
            f"{source_name}|{title}|{url}|{published_at}".encode("utf-8")
        ).hexdigest()
        events.append(
            NewsEvent(
                source_name=source_name,
                title=title,
                url=url,
                published_at=published_at,
                summary=summary,
                related_ticker=None,
                content_hash=content_hash,
            )
        )
    return events


def _text(article: dict[str, Any], *names: str) -> str:
    for name in names:
        value = article.get(name)
        if value is not None:
            return str(value).strip()
    return ""


def _summary(article: dict[str, Any]) -> str:
    parts = [
        "GDELT article metadata only; review original article before company-level use.",
    ]
    domain = _text(article, "domain", "source")
    language = _text(article, "language")
    country = _text(article, "sourcecountry")
    if domain:
        parts.append(f"domain={domain}")
    if language:
        parts.append(f"language={language}")
    if country:
        parts.append(f"sourcecountry={country}")
    return "; ".join(parts)
