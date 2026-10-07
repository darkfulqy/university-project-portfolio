from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.core.exceptions import ApiException
from app.db.models import Document, Keyword, UserUploadedDocument
from app.utils.keywords import split_keywords
from app.utils.text_repair import repair_mojibake

_CJK_RE = re.compile(r"[㐀-䶿一-鿿]+")
_ASCII_RE = re.compile(r"[a-z0-9]+")
_TITLE_KEY_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_AUTHOR_SPLIT_RE = re.compile(r"[\s,;/|、，；]+")


@dataclass
class UploadedDocumentProxy:
    id: int
    source_group_id: int
    title: str
    author: str | None
    publication_date: str | None
    source_type: str | None
    source_db: str | None
    abstract: str | None
    keywords: str | None
    doi: str | None
    isbn: str | None
    issn: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    school: str | None
    advisor: str | None
    author_unit: str | None
    url: str | None
    created_at: object
    updated_at: object
    keywords_rel: list
    _uploaded_document: UserUploadedDocument


@lru_cache(maxsize=4096)
def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    return text.lower().strip()


@lru_cache(maxsize=4096)
def _tokenize_text(value: str) -> tuple[str, ...]:
    text = _normalize_text(value)
    if not text:
        return ()

    tokens: list[str] = []
    for token in _ASCII_RE.findall(text):
        if len(token) >= 2:
            tokens.append(token)
    for block in _CJK_RE.findall(text):
        if not block:
            continue
        tokens.append(block)
        tokens.extend(list(block))
        if len(block) >= 2:
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
        if len(block) >= 3:
            tokens.extend(block[i : i + 3] for i in range(len(block) - 2))

    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    return tuple(deduped)


def _safe_text(value: str | None) -> str:
    return (repair_mojibake(value) or "").strip()


def _normalize_title_key(value: str | None) -> str:
    return _TITLE_KEY_RE.sub("", _normalize_text(_safe_text(value)))


def _normalize_author_key(value: str | None) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    parts: list[str] = []
    for part in _AUTHOR_SPLIT_RE.split(text):
        cleaned = _TITLE_KEY_RE.sub("", _normalize_text(part))
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return _TITLE_KEY_RE.sub("", _normalize_text(text))
    return "|".join(sorted(dict.fromkeys(parts)))


def _document_merge_key(doc: Document) -> str:
    if getattr(doc, "_uploaded_document", None) is not None:
        return f"uploaded:{abs(int(doc.id or 0))}"
    title_key = _normalize_title_key(doc.title)
    if not title_key:
        return f"id:{doc.id}"
    author_key = _normalize_author_key(doc.author)
    return f"{title_key}::{author_key}"


def _document_keywords(doc: Document) -> list[str]:
    if doc.keywords_rel:
        return [fixed for kw in doc.keywords_rel if (fixed := _safe_text(kw.name))]
    return [fixed for fixed in (_safe_text(name) for name in split_keywords(doc.keywords)) if fixed]


def _weighted_document_tokens(doc: Document) -> list[str]:
    title_tokens = list(_tokenize_text(_safe_text(doc.title)))
    keyword_tokens: list[str] = []
    for keyword in _document_keywords(doc):
        keyword_tokens.extend(_tokenize_text(keyword))
    author_tokens = list(_tokenize_text(" ".join(filter(None, [_safe_text(doc.author), _safe_text(doc.author_unit)]))))
    meta_tokens = list(
        _tokenize_text(
            " ".join(
                filter(
                    None,
                    [
                        _safe_text(doc.abstract),
                        _safe_text(doc.source_type),
                        _safe_text(doc.source_db),
                        _safe_text(doc.school),
                        _safe_text(doc.advisor),
                    ],
                )
            )
        )
    )
    return title_tokens * 5 + keyword_tokens * 4 + author_tokens * 3 + meta_tokens


def _field_texts(doc: Document) -> dict[str, str]:
    keywords_text = " ".join(_document_keywords(doc))
    return {
        "title": _normalize_text(_safe_text(doc.title)),
        "author": _normalize_text(_safe_text(doc.author)),
        "author_unit": _normalize_text(_safe_text(doc.author_unit)),
        "keywords": _normalize_text(keywords_text),
        "abstract": _normalize_text(_safe_text(doc.abstract)),
        "meta": _normalize_text(
            " ".join(
                filter(
                    None,
                    [
                        _safe_text(doc.source_type),
                        _safe_text(doc.source_db),
                        _safe_text(doc.school),
                        _safe_text(doc.advisor),
                    ],
                )
            )
        ),
    }


def _document_quality_score(doc: Document) -> tuple[float, int]:
    score = 0.0
    for value in [
        doc.abstract,
        doc.publication_date,
        doc.author,
        doc.author_unit,
        doc.source_db,
        doc.source_type,
        doc.url,
    ]:
        if _safe_text(value):
            score += 1.0
    score += min(len(_document_keywords(doc)), 8) * 0.2
    return score, int(doc.id or 0)


def _group_documents(documents: list[Document]) -> tuple[list[str], dict[str, list[Document]]]:
    ordered_keys: list[str] = []
    grouped: dict[str, list[Document]] = {}
    for doc in documents:
        key = _document_merge_key(doc)
        if key not in grouped:
            ordered_keys.append(key)
            grouped[key] = []
        grouped[key].append(doc)
    for key in grouped:
        grouped[key].sort(key=lambda item: (_document_quality_score(item), int(item.id or 0)), reverse=True)
    return ordered_keys, grouped


def _as_uploaded_proxy(item: UserUploadedDocument) -> UploadedDocumentProxy:
    return UploadedDocumentProxy(
        id=-item.id,
        source_group_id=-item.id,
        title=item.title,
        author=item.author,
        publication_date=item.publication_date,
        source_type=item.source_type,
        source_db=item.source_db,
        abstract=item.abstract,
        keywords=item.keywords,
        doi=None,
        isbn=None,
        issn=None,
        volume=None,
        issue=None,
        pages=None,
        school=None,
        advisor=None,
        author_unit=None,
        url=None,
        created_at=item.created_at,
        updated_at=item.updated_at,
        keywords_rel=[],
        _uploaded_document=item,
    )


def _load_uploaded_documents(db: Session) -> list[UploadedDocumentProxy]:
    uploaded_items = db.query(UserUploadedDocument).options(selectinload(UserUploadedDocument.files)).all()
    return [_as_uploaded_proxy(item) for item in uploaded_items]


def _load_all_documents(db: Session) -> list[Document]:
    base_items = db.query(Document).options(selectinload(Document.keywords_rel)).all()
    return base_items + _load_uploaded_documents(db)


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    ordered_keys, grouped = _group_documents(documents)
    return [grouped[key][0] for key in ordered_keys]


def build_document_variants_map(db: Session, documents: list[Document]) -> dict[int, list[Document]]:
    if not documents:
        return {}
    _, grouped = _group_documents(_load_all_documents(db))
    variants_map: dict[int, list[Document]] = {}
    for doc in documents:
        variants_map[int(doc.id)] = list(grouped.get(_document_merge_key(doc), [doc]))
    return variants_map


def get_document_family(db: Session, doc: Document) -> list[Document]:
    return build_document_variants_map(db, [doc]).get(int(doc.id), [doc])


def _enhanced_score(doc: Document, raw_keyword: str, query_tokens: list[str], bm25_score: float) -> float:
    fields = _field_texts(doc)
    raw = _normalize_text(raw_keyword)
    title = fields["title"]
    keywords_text = fields["keywords"]
    author = fields["author"]
    author_unit = fields["author_unit"]
    abstract = fields["abstract"]
    meta = fields["meta"]

    score = max(0.0, float(bm25_score))
    if raw:
        if raw in title:
            score += 12.0
        if raw in keywords_text:
            score += 9.0
        if raw in author or raw in author_unit:
            score += 7.0
        if raw in abstract:
            score += 5.0
        if raw in meta:
            score += 3.0

    covered = 0
    for token in query_tokens:
        if token in title:
            score += 2.4
            covered += 1
        elif token in keywords_text:
            score += 2.0
            covered += 1
        elif token in author or token in author_unit:
            score += 1.6
            covered += 1
        elif token in abstract:
            score += 1.0
            covered += 1
        elif token in meta:
            score += 0.6
            covered += 1

    if query_tokens:
        score += (covered / len(query_tokens)) * 4.0

    if title and query_tokens and all(token in title for token in query_tokens):
        score += 6.0
    if keywords_text and query_tokens and all(token in keywords_text for token in query_tokens):
        score += 4.0
    return score


def _fallback_like_query(query, keyword: str):
    like = f"%{keyword}%"
    return query.outerjoin(Document.keywords_rel).filter(
        or_(
            Document.title.like(like),
            Document.abstract.like(like),
            Document.author.like(like),
            Document.author_unit.like(like),
            Keyword.name.like(like),
        )
    ).distinct()


def _paginate_documents(documents: list[Document], page: int, page_size: int) -> tuple[list[Document], int]:
    total = len(documents)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return documents[start:end], total


def _listing_sort_key(doc: Document) -> tuple[int, object, int]:
    created_at = getattr(doc, "created_at", None)
    return (1 if created_at is not None else 0, created_at, int(doc.id or 0))


def _matches_basic_keyword(doc: Document, keyword: str) -> bool:
    raw = _normalize_text(keyword)
    if not raw:
        return True
    fields = _field_texts(doc)
    return any(raw in fields[field_name] for field_name in ("title", "author", "author_unit", "keywords", "abstract", "meta"))


def _basic_list_documents(db: Session, page: int, page_size: int, keyword: str | None):
    documents = _load_all_documents(db)
    if keyword:
        documents = [doc for doc in documents if _matches_basic_keyword(doc, keyword)]
    documents.sort(key=_listing_sort_key, reverse=True)
    deduped = _dedupe_documents(documents)
    return _paginate_documents(deduped, page, page_size)


def _enhanced_list_documents(db: Session, page: int, page_size: int, keyword: str):
    query_tokens = list(_tokenize_text(keyword))
    if not query_tokens:
        return _basic_list_documents(db, page, page_size, keyword)

    documents = _load_all_documents(db)
    if not documents:
        return [], 0

    corpus = [_weighted_document_tokens(doc) for doc in documents]
    if not any(corpus):
        return _basic_list_documents(db, page, page_size, keyword)

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    ranked: list[tuple[float, Document]] = []
    for doc, bm25_score in zip(documents, bm25_scores):
        score = _enhanced_score(doc, keyword, query_tokens, float(bm25_score))
        if score > 0:
            ranked.append((score, doc))

    if not ranked:
        return _basic_list_documents(db, page, page_size, keyword)

    ranked.sort(key=lambda item: (item[0], _document_quality_score(item[1]), item[1].id), reverse=True)
    deduped: list[Document] = []
    seen_keys: set[str] = set()
    for _, doc in ranked:
        key = _document_merge_key(doc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(doc)
    return _paginate_documents(deduped, page, page_size)


def _split_author_names(value: str | None) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    parts = [part.strip() for part in _AUTHOR_SPLIT_RE.split(text) if part.strip()]
    return parts[:6] if parts else [text]


def build_document_graph(db: Session, keyword: str | None, limit: int = 24, search_mode: str = "enhanced") -> dict:
    page_size = max(12, min(limit * 4, 120))
    documents, total = list_documents(db, 1, page_size, keyword, search_mode)
    documents = documents[: max(8, min(limit, 36))]

    nodes: list[dict] = []
    links: list[dict] = []
    node_map: dict[str, dict] = {}
    edge_set: set[tuple[str, str, str]] = set()

    def ensure_node(node_id: str, label: str, kind: str, *, document_id: int | None = None, weight: float = 1.0, meta: dict | None = None):
        node = node_map.get(node_id)
        if node is None:
            node = {
                "id": node_id,
                "label": label,
                "kind": kind,
                "weight": weight,
                "document_id": document_id,
                "documents": [],
                "meta": meta or {},
            }
            node_map[node_id] = node
            nodes.append(node)
        else:
            node["weight"] = max(float(node.get("weight", 1.0)), float(weight))
            if meta:
                node["meta"].update({k: v for k, v in meta.items() if v})
        return node

    def attach_document(node: dict, doc: Document):
        ref = {"id": doc.id, "title": _safe_text(doc.title) or f"Document {doc.id}"}
        if ref not in node["documents"]:
            node["documents"].append(ref)

    def add_link(source: str, target: str, relation: str):
        key = (source, target, relation)
        if key in edge_set:
            return
        edge_set.add(key)
        links.append({"source": source, "target": target, "relation": relation})

    for doc in documents:
        doc_id = f"doc:{doc.id}"
        doc_node = ensure_node(
            doc_id,
            _safe_text(doc.title) or f"Document {doc.id}",
            "document",
            document_id=doc.id,
            weight=5.0,
            meta={
                "author": _safe_text(doc.author),
                "publication_date": _safe_text(doc.publication_date),
                "source_db": _safe_text(doc.source_db),
                "source_type": _safe_text(doc.source_type),
            },
        )
        attach_document(doc_node, doc)

        for author in _split_author_names(doc.author):
            author_id = f"author:{_normalize_text(author)}"
            author_node = ensure_node(author_id, author, "author", weight=3.0)
            attach_document(author_node, doc)
            add_link(author_id, doc_id, "AUTHORED")

        for keyword_name in _document_keywords(doc)[:10]:
            keyword_label = _safe_text(keyword_name)
            if not keyword_label:
                continue
            keyword_id = f"keyword:{_normalize_text(keyword_label)}"
            keyword_node = ensure_node(keyword_id, keyword_label, "keyword", weight=2.0)
            attach_document(keyword_node, doc)
            add_link(doc_id, keyword_id, "HAS_KEYWORD")

        if _safe_text(doc.source_db):
            source_id = f"source:{_normalize_text(doc.source_db)}"
            source_node = ensure_node(source_id, _safe_text(doc.source_db), "source", weight=1.5)
            attach_document(source_node, doc)
            add_link(doc_id, source_id, "FROM_SOURCE")

    for node in nodes:
        node["documents"] = node["documents"][:8]

    return {
        "keyword": keyword or "",
        "search_mode": search_mode,
        "total_documents": total,
        "documents": [
            {
                "id": doc.id,
                "title": _safe_text(doc.title) or f"Document {doc.id}",
                "author": _safe_text(doc.author),
                "publication_date": _safe_text(doc.publication_date),
            }
            for doc in documents
        ],
        "nodes": nodes,
        "links": links,
    }


def _sync_keywords(db: Session, doc: Document, keywords: list[str] | None) -> None:
    if keywords is None:
        return
    names = [k.strip() for k in keywords if k.strip()]
    doc.keywords = ",".join(names) if names else None
    doc.keywords_rel = []
    if not names:
        return
    existing = db.query(Keyword).filter(Keyword.name.in_(names)).all()
    existing_map = {k.name: k for k in existing}
    for name in names:
        keyword = existing_map.get(name)
        if not keyword:
            keyword = Keyword(name=name)
            db.add(keyword)
            db.flush()
            existing_map[name] = keyword
        doc.keywords_rel.append(keyword)


def create_document(db: Session, payload: dict) -> Document:
    keywords = payload.pop("keywords", None)
    doc = Document(**payload)
    db.add(doc)
    db.flush()
    _sync_keywords(db, doc, keywords)
    db.commit()
    db.refresh(doc)
    return doc


def list_documents(
    db: Session,
    page: int,
    page_size: int,
    keyword: str | None,
    search_mode: str = "basic",
):
    if search_mode == "enhanced" and keyword:
        return _enhanced_list_documents(db, page, page_size, keyword)
    return _basic_list_documents(db, page, page_size, keyword)


def get_document(db: Session, document_id: int) -> Document:
    if document_id < 0:
        uploaded = (
            db.query(UserUploadedDocument)
            .options(selectinload(UserUploadedDocument.files))
            .filter(UserUploadedDocument.id == abs(document_id))
            .first()
        )
        if not uploaded:
            raise ApiException(code="document_not_found", message="?????", status_code=404)
        return _as_uploaded_proxy(uploaded)
    doc = db.query(Document).options(selectinload(Document.keywords_rel)).filter(Document.id == document_id).first()
    if not doc:
        raise ApiException(code="document_not_found", message="?????", status_code=404)
    return doc


def update_document(db: Session, doc: Document, payload: dict) -> Document:
    keywords = payload.pop("keywords", None)
    for key, value in payload.items():
        if value is not None:
            setattr(doc, key, value)
    _sync_keywords(db, doc, keywords)
    db.commit()
    db.refresh(doc)
    return doc


def delete_document(db: Session, doc: Document) -> None:
    db.delete(doc)
    db.commit()


def migrate_keywords_from_text(db: Session) -> int:
    docs = db.query(Document).all()
    updated = 0
    for doc in docs:
        names = split_keywords(doc.keywords)
        if not names:
            continue
        _sync_keywords(db, doc, names)
        updated += 1
    db.commit()
    return updated
