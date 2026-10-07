import base64
import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session, object_session

from app.api.deps import get_current_user, get_request_id
from app.core.exceptions import ApiException
from app.core.response import success_response
from app.db.models import EmbeddedDocumentFile, User, UserUploadedDocumentFile
from app.db.session import get_db
from app.schemas.document import DocumentCreate, DocumentUpdate
from app.services.document_service import (
    build_document_graph,
    build_document_variants_map,
    create_document,
    delete_document,
    get_document,
    get_document_family,
    list_documents,
    update_document,
)
from app.utils.keywords import split_keywords
from app.utils.text_repair import repair_mojibake


router = APIRouter(prefix="/documents", tags=["documents"])
DOCUMENTS_BASE_DIR = Path("/data/maijidownloads")
LOCAL_DOC_EXTS = {".pdf", ".caj"}
_title_index: dict[str, dict[str, Path]] | None = None
_normalized_title_index: dict[str, dict[str, Path]] | None = None
_punct_re = re.compile(r"[\s\W_]+", re.UNICODE)
_date_like_re = re.compile(r"^\s*\d{4}([-/?]\d{1,2})?([-/?]\d{1,2})?")
_summary_lock = threading.Lock()
_summary_by_title: dict[str, str] | None = None
_summary_by_normalized_title: dict[str, str] | None = None
_summary_json_path = Path("/app/all_merged.json")


@dataclass
class ResolvedDocumentAsset:
    ext: str
    kind: str
    path: Path | None = None
    embedded: EmbeddedDocumentFile | None = None

    @property
    def filename(self) -> str:
        if self.path is not None:
            return self.path.name
        if self.embedded is not None:
            return self.embedded.filename
        return f"document{self.ext}"

    @property
    def media_type(self) -> str:
        return "application/pdf" if self.ext == ".pdf" else "application/octet-stream"

    @property
    def debug_path(self) -> str | None:
        if self.path is not None:
            return str(self.path)
        if self.embedded is not None:
            return f"embedded://{self.embedded.storage_key}"
        return None


def _normalize_title(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.strip().lower()
    text = re.sub(r"[_\-]\d+$", "", text)
    text = _punct_re.sub("", text)
    return text


def _ensure_title_index() -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, Path]]]:
    global _title_index, _normalized_title_index
    if _title_index is not None and _normalized_title_index is not None:
        return _title_index, _normalized_title_index

    by_title: dict[str, dict[str, Path]] = {}
    by_norm: dict[str, dict[str, Path]] = {}
    if DOCUMENTS_BASE_DIR.exists():
        for path in DOCUMENTS_BASE_DIR.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in LOCAL_DOC_EXTS:
                continue
            ext = path.suffix.lower()
            title = path.stem.strip()
            if title:
                by_title.setdefault(title, {})
                by_title[title].setdefault(ext, path)
            norm = _normalize_title(title)
            if norm:
                by_norm.setdefault(norm, {})
                by_norm[norm].setdefault(ext, path)

    _title_index = by_title
    _normalized_title_index = by_norm
    return by_title, by_norm


def _find_local_paths_by_title(title: str | None) -> dict[str, Path]:
    if not title:
        return {}
    by_title, by_norm = _ensure_title_index()
    exact = by_title.get(title.strip())
    if exact:
        return dict(exact)
    matched = by_norm.get(_normalize_title(title))
    return dict(matched) if matched else {}


def _looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\")


def _resolve_local_document_path(raw_url: str) -> Path:
    raw_value = (raw_url or "").strip()
    if not raw_value:
        raise ApiException(code="document_path_invalid", message="Invalid document path", status_code=400)
    if _looks_like_windows_path(raw_value):
        raise ApiException(code="document_path_invalid", message="Invalid document path", status_code=400)
    if len(raw_value) > 1024:
        raise ApiException(code="document_path_invalid", message="Invalid document path", status_code=400)

    path = Path(raw_value)
    if not path.is_absolute():
        path = DOCUMENTS_BASE_DIR / raw_value.lstrip("/\\")

    resolved = path.resolve(strict=False)
    if DOCUMENTS_BASE_DIR not in resolved.parents and resolved != DOCUMENTS_BASE_DIR:
        raise ApiException(code="document_path_invalid", message="Invalid document path", status_code=400)
    try:
        exists = resolved.exists()
    except OSError as exc:
        raise ApiException(code="document_path_invalid", message="Invalid document path", status_code=400) from exc
    if not exists:
        raise ApiException(code="document_file_missing", message="Document file not found", status_code=404)
    return resolved


def _storage_key_from_raw_url(raw_url: str | None) -> str | None:
    text = (raw_url or "").strip()
    if not text or text.startswith(("http://", "https://")) or _looks_like_windows_path(text):
        return None
    normalized = text.replace("\\", "/")
    base = DOCUMENTS_BASE_DIR.as_posix().rstrip("/")
    if normalized.startswith(base + "/"):
        normalized = normalized[len(base) + 1 :]
    else:
        normalized = normalized.lstrip("/")
    return normalized or None


def _repair_utf8_mojibake_simple(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    candidates = [text]
    for source_encoding in ("latin1", "cp1252"):
        try:
            repaired = text.encode(source_encoding).decode("utf-8")
        except Exception:
            continue
        if repaired not in candidates:
            candidates.append(repaired)
    return candidates[-1]


def _candidate_title_values(member) -> list[str]:
    candidates: list[str] = []

    def add(value: str | None) -> None:
        text = (value or "").strip()
        if text:
            candidates.append(text)

    raw_title = getattr(member, "title", None)
    add(raw_title)
    add(_clean_text(raw_title))
    add(_repair_utf8_mojibake_simple(raw_title))

    raw_url = (getattr(member, "url", None) or "").strip()
    if raw_url:
        add(Path(raw_url).stem)
        repaired_url = _repair_utf8_mojibake_simple(raw_url)
        if repaired_url:
            add(Path(repaired_url).stem)
            add(_clean_text(Path(repaired_url).stem))

    return _dedupe_preserve_order(candidates)


def _find_embedded_file_by_storage_key(db: Session | None, storage_key: str | None) -> EmbeddedDocumentFile | None:
    if db is None or not storage_key:
        return None
    return (
        db.query(EmbeddedDocumentFile)
        .filter(EmbeddedDocumentFile.storage_key == storage_key)
        .first()
    )


def _find_embedded_files_by_title(db: Session | None, title: str | None) -> dict[str, EmbeddedDocumentFile]:
    if db is None or not title:
        return {}
    normalized = _normalize_title(title)
    if not normalized:
        return {}
    rows = (
        db.query(EmbeddedDocumentFile)
        .filter(EmbeddedDocumentFile.normalized_title == normalized)
        .all()
    )
    result: dict[str, EmbeddedDocumentFile] = {}
    for row in rows:
        ext = (row.file_ext or "").lower()
        if ext in LOCAL_DOC_EXTS:
            result.setdefault(ext, row)
    return result


def _resolve_document_assets(doc, family=None) -> dict[str, ResolvedDocumentAsset]:
    family = list(family or [doc])
    db = object_session(doc) or next((object_session(member) for member in family if object_session(member) is not None), None)
    resolved: dict[str, ResolvedDocumentAsset] = {}

    raw_urls = _dedupe_preserve_order(
        [
            (member.url or "").strip()
            for member in family
            if (member.url or "").strip() and not (member.url or "").strip().startswith(("http://", "https://"))
        ]
    )
    for raw_url in raw_urls:
        try:
            local_path = _resolve_local_document_path(raw_url)
            ext = local_path.suffix.lower()
            if ext in LOCAL_DOC_EXTS:
                resolved.setdefault(ext, ResolvedDocumentAsset(ext=ext, kind="local", path=local_path))
        except (ApiException, OSError):
            pass
        embedded = _find_embedded_file_by_storage_key(db, _storage_key_from_raw_url(raw_url))
        if embedded is not None:
            ext = (embedded.file_ext or "").lower()
            if ext in LOCAL_DOC_EXTS:
                resolved.setdefault(ext, ResolvedDocumentAsset(ext=ext, kind="embedded", embedded=embedded))

    title_candidates: list[str] = []
    for member in family:
        title_candidates.extend(_candidate_title_values(member))

    for title in _dedupe_preserve_order(title_candidates):
        for ext, path in _find_local_paths_by_title(title).items():
            resolved.setdefault(ext, ResolvedDocumentAsset(ext=ext, kind="local", path=path))
        for ext, embedded in _find_embedded_files_by_title(db, title).items():
            resolved.setdefault(ext, ResolvedDocumentAsset(ext=ext, kind="embedded", embedded=embedded))

    return resolved


def _iter_embedded_asset_bytes(asset: ResolvedDocumentAsset):
    if asset.embedded is None:
        return
    for chunk in asset.embedded.chunks:
        payload = chunk.content_base64
        if payload is None:
            continue
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, str):
            payload = payload.encode("ascii")
        yield base64.b64decode(payload)


def _build_content_disposition(filename: str, disposition: str) -> str:
    return f"{disposition}; filename*=UTF-8''{quote(filename)}"


def _ensure_summary_index() -> tuple[dict[str, str], dict[str, str]]:
    global _summary_by_title, _summary_by_normalized_title
    if _summary_by_title is not None and _summary_by_normalized_title is not None:
        return _summary_by_title, _summary_by_normalized_title

    with _summary_lock:
        if _summary_by_title is not None and _summary_by_normalized_title is not None:
            return _summary_by_title, _summary_by_normalized_title

        by_title: dict[str, str] = {}
        by_norm: dict[str, str] = {}
        if _summary_json_path.exists():
            try:
                with _summary_json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        parsed = item.get("parsed")
                        if not isinstance(parsed, dict):
                            continue
                        title = (parsed.get("title") or "").strip()
                        summary = (parsed.get("summary") or "").strip()
                        if not title or not summary:
                            continue
                        by_title.setdefault(title, summary)
                        norm = _normalize_title(title)
                        if norm:
                            by_norm.setdefault(norm, summary)
            except Exception:
                by_title = {}
                by_norm = {}

        _summary_by_title = by_title
        _summary_by_normalized_title = by_norm
        return _summary_by_title, _summary_by_normalized_title


def _find_ai_summary_by_title(title: str | None) -> str | None:
    if not title:
        return None
    by_title, by_norm = _ensure_summary_index()
    exact = by_title.get(title.strip())
    if exact:
        return exact
    return by_norm.get(_normalize_title(title))


def _looks_like_misplaced_abstract(publication_date: str | None, abstract: str | None) -> bool:
    if abstract and abstract.strip():
        return False
    if not publication_date:
        return False
    text = publication_date.strip()
    if len(text) < 40:
        return False
    if _date_like_re.match(text):
        return False
    return ("?" in text) or ("?" in text) or ("," in text and "." in text)


def _clean_text(value: str | None) -> str | None:
    return repair_mojibake(value)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = (value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _date_completeness_score(value: str | None) -> tuple[int, int, int, str]:
    text = (value or "").strip()
    if not text:
        return (0, 0, 0, "")
    score = 0
    if re.search(r"\d{4}", text):
        score += 1
    if re.search(r"(?:[-/?])\d{1,2}", text):
        score += 1
    if re.search(r"(?:[-/?])\d{1,2}", text):
        score += 1
    return (score, len(text), sum(ch.isdigit() for ch in text), text)


def _pick_best_date(family: list) -> str | None:
    candidates = [(_date_completeness_score(member.publication_date), (member.publication_date or "").strip()) for member in family if (member.publication_date or "").strip()]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _pick_best_text(values: list[str | None]) -> str | None:
    cleaned = [item for item in (_clean_text(value) for value in values) if item]
    if not cleaned:
        return None
    cleaned.sort(key=lambda item: (len(item), item), reverse=True)
    return cleaned[0]


def _extract_source_table(raw_url: str | None) -> str | None:
    text = (raw_url or "").strip()
    if not text:
        return None
    try:
        parsed = urlparse(text)
        table = parse_qs(parsed.query).get("table", [None])[0]
        if table:
            return table.strip()
    except Exception:
        pass
    matched = re.search(r"(?:^|[?&])table=([^&#]+)", text)
    return matched.group(1).strip() if matched else None


def _serialize_uploaded_document(doc, uploaded_document):
    primary_file = next((item for item in uploaded_document.files if item.is_primary), None)
    available_file_types = sorted(item.file_format for item in uploaded_document.files)
    file_url = f"/documents/{doc.id}/file" if primary_file else None
    preview_url = file_url if primary_file and primary_file.file_format == "pdf" else None
    download_url = f"{file_url}?download=1" if primary_file else None
    return {
        "id": doc.id,
        "requested_document_id": doc.id,
        "primary_document_id": doc.id,
        "variant_ids": [doc.id],
        "variant_count": 1,
        "source_group_id": doc.source_group_id,
        "title": _clean_text(doc.title),
        "author": _clean_text(doc.author),
        "publication_date": doc.publication_date,
        "source_type": _clean_text(doc.source_type),
        "source_types": [_clean_text(doc.source_type)] if _clean_text(doc.source_type) else [],
        "source_db": _clean_text(doc.source_db),
        "source_dbs": [_clean_text(doc.source_db)] if _clean_text(doc.source_db) else [],
        "abstract": _clean_text(doc.abstract),
        "keywords": split_keywords(doc.keywords),
        "doi": None,
        "isbn": None,
        "issn": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "school": None,
        "advisor": None,
        "author_unit": None,
        "url": file_url,
        "source_url": None,
        "source_urls": [],
        "source_tables": [],
        "source_records": [{"id": doc.id, "title": _clean_text(doc.title), "author": _clean_text(doc.author), "publication_date": doc.publication_date, "source_db": _clean_text(doc.source_db), "source_type": _clean_text(doc.source_type), "source_url": None, "source_table": None}],
        "preview_url": preview_url,
        "download_url": download_url,
        "pdf_download_url": f"{file_url}?download=1&format=pdf" if "pdf" in available_file_types else None,
        "caj_download_url": f"{file_url}?download=1&format=caj" if "caj" in available_file_types else None,
        "available_file_types": available_file_types,
        "file_path": primary_file.path if primary_file else None,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


def _serialize_document(doc, family=None):
    uploaded_document = getattr(doc, "_uploaded_document", None)
    if uploaded_document is not None:
        return _serialize_uploaded_document(doc, uploaded_document)

    family = list(family or [doc])
    family.sort(key=lambda member: (len((_clean_text(member.abstract) or "")), bool((_clean_text(member.publication_date) or "")), int(member.id or 0)), reverse=True)
    primary_doc = family[0]

    keyword_candidates: list[str] = []
    for member in family:
        keyword_source = [kw.name for kw in member.keywords_rel] if member.keywords_rel else split_keywords(member.keywords)
        keyword_candidates.extend([item for item in (_clean_text(name) for name in keyword_source) if item])
    keyword_list = _dedupe_preserve_order(keyword_candidates)

    raw_source_urls = _dedupe_preserve_order([(member.url or "").strip() for member in family if (member.url or "").strip()])
    source_urls = raw_source_urls
    external_source_url = next((url for url in source_urls if url.startswith(("http://", "https://"))), None)

    resolved_assets = _resolve_document_assets(primary_doc, family)
    pdf_asset = resolved_assets.get(".pdf")
    caj_asset = resolved_assets.get(".caj")
    default_asset = pdf_asset or caj_asset

    file_url = None
    file_path = None
    preview_url = None
    download_url = None

    if default_asset:
        file_url = f"/documents/{primary_doc.id}/file"
        file_path = default_asset.debug_path
        preview_url = f"{file_url}?format=pdf" if pdf_asset else None
        download_url = (
            f"{file_url}?download=1&format=pdf"
            if pdf_asset
            else (f"{file_url}?download=1&format=caj" if caj_asset else None)
        )
    elif external_source_url:
        file_url = external_source_url

    title = _pick_best_text([member.title for member in family]) or _clean_text(primary_doc.title)
    author = _pick_best_text([member.author for member in family]) or _clean_text(primary_doc.author)
    publication_date = _pick_best_date(family)
    source_type = _pick_best_text([member.source_type for member in family]) or _clean_text(primary_doc.source_type)
    source_db = _pick_best_text([member.source_db for member in family]) or _clean_text(primary_doc.source_db)
    source_dbs = _dedupe_preserve_order([item for item in (_clean_text(member.source_db) for member in family) if item])
    source_types = _dedupe_preserve_order([item for item in (_clean_text(member.source_type) for member in family) if item])
    abstract = _pick_best_text([member.abstract for member in family])
    school = _pick_best_text([member.school for member in family]) or _clean_text(primary_doc.school)
    advisor = _pick_best_text([member.advisor for member in family]) or _clean_text(primary_doc.advisor)
    author_unit = _pick_best_text([member.author_unit for member in family]) or _clean_text(primary_doc.author_unit)
    if _looks_like_misplaced_abstract(publication_date, abstract):
        abstract = publication_date
        publication_date = None
    if not (abstract or "").strip():
        fallback_summary = _find_ai_summary_by_title(title or primary_doc.title)
        if fallback_summary:
            abstract = _clean_text(fallback_summary)

    source_tables = _dedupe_preserve_order([item for item in (_extract_source_table(url) for url in source_urls) if item])
    source_records = []
    for member in family:
        member_url = (member.url or "").strip() or None
        source_records.append(
            {
                "id": member.id,
                "title": _clean_text(member.title),
                "author": _clean_text(member.author),
                "publication_date": (member.publication_date or "").strip() or None,
                "source_db": _clean_text(member.source_db),
                "source_type": _clean_text(member.source_type),
                "source_url": member_url,
                "source_table": _extract_source_table(member_url),
            }
        )

    return {
        "id": primary_doc.id,
        "requested_document_id": doc.id,
        "primary_document_id": primary_doc.id,
        "variant_ids": [member.id for member in family],
        "variant_count": len(family),
        "source_group_id": primary_doc.source_group_id,
        "title": title,
        "author": author,
        "publication_date": publication_date,
        "source_type": source_type,
        "source_types": source_types,
        "source_db": source_db,
        "source_dbs": source_dbs,
        "abstract": abstract,
        "keywords": keyword_list,
        "doi": primary_doc.doi,
        "isbn": primary_doc.isbn,
        "issn": primary_doc.issn,
        "volume": primary_doc.volume,
        "issue": primary_doc.issue,
        "pages": primary_doc.pages,
        "school": school,
        "advisor": advisor,
        "author_unit": author_unit,
        "url": file_url,
        "source_url": external_source_url or (source_urls[0] if source_urls else None),
        "source_urls": source_urls,
        "source_tables": source_tables,
        "source_records": source_records,
        "preview_url": preview_url,
        "download_url": download_url,
        "pdf_download_url": f"/documents/{primary_doc.id}/file?download=1&format=pdf" if pdf_asset else None,
        "caj_download_url": f"/documents/{primary_doc.id}/file?download=1&format=caj" if caj_asset else None,
        "available_file_types": [ext.lstrip(".") for ext in sorted(resolved_assets.keys())],
        "file_path": file_path,
        "created_at": primary_doc.created_at.isoformat() if primary_doc.created_at else None,
        "updated_at": primary_doc.updated_at.isoformat() if primary_doc.updated_at else None,
    }


@router.get("")
def list_documents_api(
    request: Request,
    page: int = 1,
    page_size: int = 10,
    keyword: str | None = None,
    search_mode: str = "basic",
    db: Session = Depends(get_db),
):
    if search_mode not in {"basic", "enhanced"}:
        raise ApiException(code="document_search_mode_invalid", message="Invalid search mode", status_code=400)
    items, total = list_documents(db, page, page_size, keyword, search_mode)
    family_map = build_document_variants_map(db, items)
    data = {
        "items": [_serialize_document(item, family_map.get(int(item.id), [item])) for item in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
    return success_response(data, get_request_id(request))


@router.get("/graph")
def document_graph_api(
    request: Request,
    keyword: str | None = None,
    limit: int = 24,
    search_mode: str = "enhanced",
    db: Session = Depends(get_db),
):
    if search_mode not in {"basic", "enhanced"}:
        raise ApiException(code="document_search_mode_invalid", message="Invalid search mode", status_code=400)
    graph = build_document_graph(db, keyword, limit, search_mode)
    return success_response(graph, get_request_id(request))


@router.post("")
def create_document_api(
    request: Request,
    payload: DocumentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ = current_user
    doc = create_document(db, payload.model_dump())
    family = get_document_family(db, doc)
    return success_response(_serialize_document(doc, family), get_request_id(request), status_code=201)


@router.get("/{document_id}")
def get_document_api(request: Request, document_id: int, db: Session = Depends(get_db)):
    doc = get_document(db, document_id)
    family = get_document_family(db, doc)
    return success_response(_serialize_document(doc, family), get_request_id(request))


@router.get("/{document_id}/file")
def get_document_file_api(
    request: Request,
    document_id: int,
    download: bool = False,
    format: str | None = None,
    db: Session = Depends(get_db),
):
    _ = request
    doc = get_document(db, document_id)
    uploaded_document = getattr(doc, "_uploaded_document", None)
    if uploaded_document is not None:
        preferred = (format or "").strip().lower()
        if preferred and preferred not in {"pdf", "caj", "doc", "docx", "txt"}:
            raise ApiException(code="document_format_invalid", message="Unsupported document format", status_code=400)
        file_record = None
        if preferred:
            file_record = next((item for item in uploaded_document.files if item.file_format == preferred), None)
        if not file_record:
            file_record = next((item for item in uploaded_document.files if item.is_primary), None) or (uploaded_document.files[0] if uploaded_document.files else None)
        if not file_record:
            raise ApiException(code="document_file_missing_local", message="Local document file not found", status_code=404)
        media_type = file_record.mime_type or "application/octet-stream"
        disposition = "attachment" if download else "inline"
        return FileResponse(path=file_record.path, media_type=media_type, filename=file_record.filename, content_disposition_type=disposition)

    family = get_document_family(db, doc)
    resolved_assets = _resolve_document_assets(doc, family)
    preferred = (format or "").strip().lower()
    if preferred and preferred not in {"pdf", "caj"}:
        raise ApiException(code="document_format_invalid", message="Unsupported document format", status_code=400)

    asset = None
    if preferred:
        asset = resolved_assets.get(f".{preferred}")
    if not asset:
        asset = resolved_assets.get(".pdf") or resolved_assets.get(".caj")
    if not asset:
        raise ApiException(code="document_file_missing_local", message="Local document file not found for this record", status_code=404)

    if asset.kind == "local" and asset.path is not None:
        if download:
            return FileResponse(path=str(asset.path), media_type=asset.media_type, filename=asset.filename, content_disposition_type="attachment")
        return FileResponse(path=str(asset.path), media_type=asset.media_type, filename=asset.filename, content_disposition_type="inline")

    headers = {"Content-Disposition": _build_content_disposition(asset.filename, "attachment" if download else "inline")}
    return StreamingResponse(_iter_embedded_asset_bytes(asset), media_type=asset.media_type, headers=headers)


@router.put("/{document_id}")
def update_document_api(
    request: Request,
    document_id: int,
    payload: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ = current_user
    doc = get_document(db, document_id)
    doc = update_document(db, doc, payload.model_dump())
    family = get_document_family(db, doc)
    return success_response(_serialize_document(doc, family), get_request_id(request))


@router.delete("/{document_id}")
def delete_document_api(
    request: Request,
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ = current_user
    doc = get_document(db, document_id)
    delete_document(db, doc)
    return success_response({"deleted": True}, get_request_id(request))
