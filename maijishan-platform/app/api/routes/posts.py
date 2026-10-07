import re
import time
from urllib.parse import urljoin, urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_request_id
from app.core.exceptions import ApiException
from app.core.response import success_response
from app.db.models import PostImage, User
from app.db.session import get_db
from app.schemas.post import PostCreate, PostUpdate
from app.services.post_service import create_post, list_posts, get_post, update_post, delete_post


router = APIRouter(prefix="/posts", tags=["posts"])
_preview_cache: dict[str, tuple[float, dict]] = {}
_preview_ttl_seconds = 6 * 60 * 60


def _extract_image_from_content(content: str | None) -> str | None:
    if not content:
        return None
    patterns = [
        r"<img[^>]+src=[\"']([^\"']+)[\"']",
        r"!\[[^\]]*\]\((https?://[^)]+)\)",
        r"(https?://\S+\.(?:png|jpg|jpeg|gif|webp))",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_meta_from_html(html: str, source_url: str) -> tuple[str | None, str | None]:
    site_name_match = re.search(
        r"<meta[^>]+(?:property|name)=[\"'](?:og:site_name|twitter:site)[\"'][^>]+content=[\"']([^\"']+)[\"']",
        html,
        re.IGNORECASE,
    )
    image_match = re.search(
        r"<meta[^>]+(?:property|name)=[\"'](?:og:image|twitter:image)[\"'][^>]+content=[\"']([^\"']+)[\"']",
        html,
        re.IGNORECASE,
    )
    site_name = site_name_match.group(1).strip() if site_name_match else None
    image_url = image_match.group(1).strip() if image_match else None
    if image_url:
        image_url = urljoin(source_url, image_url)
    return site_name, image_url


def _extract_image_from_html(html: str, source_url: str) -> str | None:
    # Priority: known meta/image fields -> first meaningful image in body.
    patterns = [
        r"<meta[^>]+(?:property|name)=[\"'](?:og:image|twitter:image|image)[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<meta[^>]+itemprop=[\"']image[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<img[^>]+(?:data-original|data-actualsrc|data-src|src)=[\"']([^\"']+)[\"']",
        r"(https?://[^\\\"'\\s>]+\\.(?:jpg|jpeg|png|webp|gif))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            candidate = (match.group(1) or "").strip()
            if not candidate:
                continue
            candidate = urljoin(source_url, candidate)
            lower = candidate.lower()
            if any(
                bad in lower
                for bad in [
                    "favicon",
                    "avatar",
                    "logo",
                    "sprite",
                    ".svg",
                    "data:image/",
                ]
            ):
                continue
            return candidate
    return None


def _resolve_post_preview(post, fetch_remote: bool = False) -> dict:
    source_url = (post.post_url or "").strip() or None
    image_url = _extract_image_from_content(post.content)
    source_name = (post.platform or "").strip() or None
    if not source_url:
        return {"source_name": source_name, "source_url": None, "image_url": image_url}

    cache = _preview_cache.get(source_url)
    now = time.time()
    if cache and now - cache[0] <= _preview_ttl_seconds:
        cached = dict(cache[1])
    elif fetch_remote:
        cached = {"source_name": source_name, "image_url": image_url}
        try:
            req = UrlRequest(
                source_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
            with urlopen(req, timeout=5) as resp:
                encoding = resp.headers.get_content_charset() or "utf-8"
                html = resp.read(120_000).decode(encoding, errors="ignore")
            html_source_name, html_image = _extract_meta_from_html(html, source_url)
            if html_source_name:
                cached["source_name"] = html_source_name
            if html_image and not cached.get("image_url"):
                cached["image_url"] = html_image
            if not cached.get("image_url"):
                first_image = _extract_image_from_html(html, source_url)
                if first_image:
                    cached["image_url"] = first_image
        except Exception:
            pass
        _preview_cache[source_url] = (now, dict(cached))
    else:
        cached = {"source_name": source_name, "image_url": image_url}

    hostname = urlparse(source_url).netloc
    if not cached.get("source_name"):
        cached["source_name"] = hostname or source_url
    if not cached.get("image_url") and hostname:
        cached["image_url"] = f"https://www.google.com/s2/favicons?domain={hostname}&sz=256"

    return {
        "source_name": cached.get("source_name"),
        "source_url": source_url,
        "image_url": cached.get("image_url"),
    }


def _serialize_post(post):
    preview = _resolve_post_preview(post, fetch_remote=False)
    return {
        "id": post.id,
        "title": post.title,
        "content": post.content,
        "platform": post.platform,
        "post_url": post.post_url,
        "author_name": post.author_name,
        "likes_count": post.likes_count,
        "comments_count": post.comments_count,
        "shares_count": post.shares_count,
        "views_count": post.views_count,
        "publish_date": post.publish_date,
        "source_name": preview.get("source_name"),
        "source_url": preview.get("source_url"),
        "image_url": preview.get("image_url"),
        "created_at": post.created_at.isoformat() if post.created_at else None,
        "updated_at": post.updated_at.isoformat() if post.updated_at else None,
    }


@router.get("")
def list_posts_api(
    request: Request,
    page: int = 1,
    page_size: int = 10,
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    items, total = list_posts(db, page, page_size, keyword)
    data = {
        "items": [_serialize_post(item) for item in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
    return success_response(data, get_request_id(request))


@router.post("")
def create_post_api(
    request: Request,
    payload: PostCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = create_post(
        db,
        current_user.username,
        payload.title,
        payload.content,
        payload.platform,
        payload.post_url,
    )
    return success_response(_serialize_post(post), get_request_id(request), status_code=201)


@router.get("/{post_id}")
def get_post_api(request: Request, post_id: int, db: Session = Depends(get_db)):
    post = get_post(db, post_id)
    return success_response(_serialize_post(post), get_request_id(request))


@router.get("/{post_id}/preview")
def get_post_preview_api(request: Request, post_id: int, db: Session = Depends(get_db)):
    post = get_post(db, post_id)
    return success_response(_resolve_post_preview(post, fetch_remote=True), get_request_id(request))


@router.get("/{post_id}/images")
def list_post_images_api(
    request: Request,
    post_id: int,
    limit: int | None = 6,
    db: Session = Depends(get_db),
):
    get_post(db, post_id)
    query = (
        db.query(PostImage)
        .filter(PostImage.post_id == post_id)
        .order_by(PostImage.image_order.asc(), PostImage.id.asc())
    )
    if limit is not None and limit > 0:
        query = query.limit(limit)
    images = query.all()
    data = {
        "items": [
            {
                "id": img.id,
                "image_url": img.image_url,
                "order": img.image_order,
            }
            for img in images
            if img.image_url
        ],
        "total": len(images),
    }
    return success_response(data, get_request_id(request))


@router.put("/{post_id}")
def update_post_api(
    request: Request,
    post_id: int,
    payload: PostUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = get_post(db, post_id)
    if post.author_name and post.author_name != current_user.username:
        raise ApiException(code="forbidden", message="forbidden", status_code=403)
    post = update_post(db, post, payload.title, payload.content, payload.platform, payload.post_url)
    return success_response(_serialize_post(post), get_request_id(request))


@router.delete("/{post_id}")
def delete_post_api(
    request: Request,
    post_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    post = get_post(db, post_id)
    if post.author_name and post.author_name != current_user.username:
        raise ApiException(code="forbidden", message="forbidden", status_code=403)
    delete_post(db, post)
    return success_response({"deleted": True}, get_request_id(request))
