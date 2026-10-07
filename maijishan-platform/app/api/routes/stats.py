import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_request_id
from app.core.config import get_settings
from app.core.response import success_response
from app.db.models.document import Document
from app.db.session import get_db

router = APIRouter(prefix="/stats", tags=["stats"])

_KEYWORD_SPLIT_RE = re.compile(r"[;,，；、|/\\\n\t]+")
_AUTHOR_SPLIT_RE = re.compile(r"[;,，；、|/\\\n\t和与]+")
_YEAR_RE = re.compile(r"(19|20)\d{2}")

_COUNTRY_PATTERNS = {
    "中国": ["中国", "中华", "cnki", "知网", "wanfang", "vip", "cqvip"],
    "美国": ["usa", "united states", "u.s.", "america"],
    "日本": ["日本", "japan"],
    "英国": ["uk", "united kingdom", "england", "britain"],
    "法国": ["france", "french", "法国"],
    "德国": ["germany", "german", "德国"],
    "意大利": ["italy", "italian", "意大利"],
    "俄罗斯": ["russia", "russian", "俄罗斯"],
    "韩国": ["korea", "韩国"],
    "印度": ["india", "indian", "印度"],
}
_TLD_FALLBACK = {
    ".cn": "中国",
    ".jp": "日本",
    ".uk": "英国",
    ".fr": "法国",
    ".de": "德国",
    ".it": "意大利",
    ".ru": "俄罗斯",
    ".kr": "韩国",
    ".in": "印度",
    ".us": "美国",
}

_cache_lock = threading.Lock()
_cache_data: dict | None = None
_cache_expires_at: float = 0.0


def _sanitize_token(token: str, max_len: int = 40) -> str | None:
    token = (token or "").strip()
    if not token:
        return None
    token = re.sub(r"\s+", " ", token)
    if len(token) > max_len:
        return None
    if re.fullmatch(r"[\d\W_]+", token):
        return None
    return token


def _split_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    raw = raw.replace("；", ";").replace("，", ",").replace("、", ",")
    parts = _KEYWORD_SPLIT_RE.split(raw)
    out: list[str] = []
    for item in parts:
        token = _sanitize_token(item)
        if token and len(token) >= 2:
            out.append(token)
    return out


def _split_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = _AUTHOR_SPLIT_RE.split(raw)
    out: list[str] = []
    for item in parts:
        token = _sanitize_token(item)
        if token:
            out.append(token)
    return out


def _extract_year(publication_date: str | None) -> int | None:
    if not publication_date:
        return None
    match = _YEAR_RE.search(publication_date)
    if not match:
        return None
    year = int(match.group(0))
    if year < 1900 or year > datetime.now().year + 1:
        return None
    return year


def _resolve_region(author_unit: str | None, source_db: str | None, source_type: str | None, url: str | None) -> tuple[str | None, str]:
    haystack = " ".join([author_unit or "", source_db or "", source_type or ""]).lower()
    for region, patterns in _COUNTRY_PATTERNS.items():
        if any(p.lower() in haystack for p in patterns):
            return region, "high"

    if url:
        try:
            host = (urlparse(url).hostname or "").lower()
            for tld, region in _TLD_FALLBACK.items():
                if host.endswith(tld):
                    return region, "low"
        except Exception:
            pass

    return None, "low"


def _build_partial_data_meta(datasets: dict[str, list]) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for key, values in datasets.items():
        if values:
            meta[key] = {"available": True, "reason": None}
        else:
            meta[key] = {"available": False, "reason": "no_usable_data"}
    return meta


def _compute_home_stats_payload(db: Session, top_n: int = 30) -> dict:
    rows = (
        db.query(
            Document.keywords,
            Document.author,
            Document.publication_date,
            Document.author_unit,
            Document.source_db,
            Document.source_type,
            Document.url,
        )
        .all()
    )

    keyword_counter: Counter[str] = Counter()
    author_counter: Counter[str] = Counter()
    year_counter: Counter[int] = Counter()
    hotspot_counter: Counter[tuple[str, str]] = Counter()

    for keywords, author, publication_date, author_unit, source_db, source_type, url in rows:
        for token in _split_keywords(keywords):
            keyword_counter[token] += 1
        for person in _split_authors(author):
            author_counter[person] += 1
        year = _extract_year(publication_date)
        if year:
            year_counter[year] += 1

        region, confidence = _resolve_region(author_unit, source_db, source_type, url)
        if region:
            hotspot_counter[(region, confidence)] += 1

    keywords = [
        {"keyword": keyword, "count": count}
        for keyword, count in sorted(keyword_counter.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    ]
    top_authors = [
        {"author": author, "count": count}
        for author, count in sorted(author_counter.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    ]
    year_series = [{"year": year, "count": year_counter[year]} for year in sorted(year_counter.keys())]
    geo_hotspots = [
        {"region": region, "count": count, "confidence": confidence}
        for (region, confidence), count in sorted(hotspot_counter.items(), key=lambda x: (-x[1], x[0][0]))[:20]
    ]

    datasets = {
        "keywords": keywords,
        "year_series": year_series,
        "top_authors": top_authors,
        "geo_hotspots": geo_hotspots,
    }
    return {
        **datasets,
        "partial_data": _build_partial_data_meta(datasets),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_home_stats_cached(db: Session) -> tuple[dict, bool]:
    global _cache_data, _cache_expires_at
    settings = get_settings()
    now = time.time()
    with _cache_lock:
        if _cache_data is not None and now < _cache_expires_at:
            return _cache_data, True

    payload = _compute_home_stats_payload(db)

    with _cache_lock:
        _cache_data = payload
        _cache_expires_at = now + max(1, settings.homepage_stats_cache_ttl_seconds)
    return payload, False


@router.get("/home")
def homepage_stats_api(request: Request, db: Session = Depends(get_db)):
    settings = get_settings()
    if not settings.homepage_stats_enabled:
        # Keep a soft-disabled response so frontend can gracefully fallback.
        return success_response(
            {
                "disabled": True,
                "keywords": [],
                "year_series": [],
                "top_authors": [],
                "geo_hotspots": [],
                "partial_data": {
                    "keywords": {"available": False, "reason": "feature_disabled"},
                    "year_series": {"available": False, "reason": "feature_disabled"},
                    "top_authors": {"available": False, "reason": "feature_disabled"},
                    "geo_hotspots": {"available": False, "reason": "feature_disabled"},
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "cache": {
                    "hit": False,
                    "ttl_seconds": settings.homepage_stats_cache_ttl_seconds,
                },
            },
            get_request_id(request),
        )

    payload, cache_hit = _get_home_stats_cached(db)
    data = {
        **payload,
        "disabled": False,
        "cache": {
            "hit": cache_hit,
            "ttl_seconds": settings.homepage_stats_cache_ttl_seconds,
        },
    }
    return success_response(data, get_request_id(request))
