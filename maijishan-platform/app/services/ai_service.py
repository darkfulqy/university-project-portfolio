from __future__ import annotations

import math
import re
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.core.config import get_settings
from app.schemas.ai import AiChatRequest, AiChatResponse, AiSearchResponse, CitationItem


SYSTEM_PROMPT = (
    "你是麦积山研究助手。请严格依据检索到的站内资料作答，"
    "优先概括可验证的信息，避免编造。"
    "如果资料不足，直接说明依据有限。"
)

DIRECT_PROMPT = (
    "你是麦积山研究助手。当前知识库检索暂不可用时，可以给出谨慎的常识性回答，"
    "但必须明确说明这次回答没有附带知识库引文。"
)

DOC_SUFFIXES = (".md", ".markdown", ".pdf", ".doc", ".docx", ".txt", ".caj")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
TERM_RE = re.compile(r"第\d+窟|\d+窟|[\u4e00-\u9fff]{2,}|[A-Za-z0-9][A-Za-z0-9._-]*")


def chat(request: AiChatRequest) -> AiChatResponse:
    settings = get_settings()
    if not settings.ai_enabled:
        return AiChatResponse(assistant_message="AI 问答功能未开启，请先配置 AI_ENABLED。")

    dependency_error = _dependency_error()
    if dependency_error:
        return AiChatResponse(assistant_message=dependency_error)

    config_error = _config_error(settings)
    if config_error:
        return AiChatResponse(assistant_message=config_error)

    try:
        search_result = search_knowledge_base(request.user_message, top_k=settings.ai_result_top_k)
        assistant_message = _generate_answer(request.user_message, search_result.context_parts, settings)
        return AiChatResponse(
            assistant_message=assistant_message,
            citations=search_result.citations,
            context_parts=search_result.context_parts,
            sources=search_result.sources,
            images=search_result.images,
            used_tools=True,
        )
    except Exception as exc:  # noqa: BLE001
        try:
            fallback_message = _generate_direct_answer(request.user_message, settings)
        except Exception as fallback_exc:  # noqa: BLE001
            return AiChatResponse(assistant_message=f"AI 问答暂时不可用: {str(fallback_exc)[:300]}")

        return AiChatResponse(
            assistant_message=(
                f"{fallback_message}\n\n"
                f"注：当前答案未附带知识库引文，原因是检索链路暂时不可用。{str(exc)[:120]}"
            ),
            used_tools=False,
        )


def search(query: str, top_k: int | None = None) -> AiSearchResponse:
    settings = get_settings()
    if not settings.ai_enabled:
        return AiSearchResponse()
    return search_knowledge_base(query, top_k=top_k or settings.ai_result_top_k)


def search_knowledge_base(query: str, top_k: int = 5) -> AiSearchResponse:
    settings = get_settings()
    citations = _hybrid_search(query, settings, top_k=top_k)
    context_parts = [
        f"[资料 {index}] 来源：{citation.source}\n相关度：{citation.score:.4f}\n内容：{citation.text}"
        for index, citation in enumerate(citations, start=1)
    ]
    images = _collect_images(citations)
    sources = [citation.source for citation in citations]
    return AiSearchResponse(
        citations=citations,
        context_parts=context_parts,
        sources=sources,
        images=images,
    )


def _dependency_error() -> str | None:
    missing = []
    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("openai")
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        missing.append("qdrant-client")
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        missing.append("rank-bm25")

    if missing:
        return f"AI 依赖未安装: {', '.join(missing)}"
    return None


def _config_error(settings) -> str | None:
    required = {
        "AI_EMBEDDING_API_KEY": settings.ai_embedding_api_key,
        "AI_EMBEDDING_BASE_URL": settings.ai_embedding_base_url,
        "AI_EMBEDDING_MODEL": settings.ai_embedding_model,
        "AI_LLM_API_KEY": settings.ai_llm_api_key,
        "AI_LLM_BASE_URL": settings.ai_llm_base_url,
        "AI_LLM_MODEL": settings.ai_llm_model,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        return f"AI 配置缺失: {', '.join(missing)}"

    if settings.ai_qdrant_url:
        return None

    if settings.ai_qdrant_path and Path(settings.ai_qdrant_path).exists():
        return None

    return "Qdrant 配置缺失，请设置 AI_QDRANT_URL 或可用的 AI_QDRANT_PATH。"


def _get_openai_client(api_key: str, base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=1)
def _get_qdrant_client(qdrant_url: str | None, qdrant_path: str | None):
    from qdrant_client import QdrantClient

    if qdrant_url:
        return QdrantClient(url=qdrant_url, timeout=30.0, check_compatibility=False)
    if qdrant_path:
        return QdrantClient(path=qdrant_path)
    raise ValueError("Qdrant client requires url or path")


@lru_cache(maxsize=1)
def _build_bm25_cache(
    qdrant_url: str | None,
    qdrant_path: str | None,
    collection_name: str,
) -> tuple[Any, list[dict[str, Any]]]:
    from rank_bm25 import BM25Okapi

    client = _get_qdrant_client(qdrant_url, qdrant_path)
    offset = None
    payloads: list[dict[str, Any]] = []
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            source_file = str(payload.get("source_file") or "unknown")
            payloads.append(
                {
                    "id": str(point.id),
                    "text": text,
                    "source_file": source_file,
                    "source_name": _display_source_name(source_file),
                    "images": payload.get("images") or [],
                }
            )
        if offset is None:
            break

    tokenized_docs = [_tokenize(f"{item['source_name']} {item['text']}") for item in payloads]
    bm25 = BM25Okapi(tokenized_docs or [["empty"]])
    return bm25, payloads


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in TERM_RE.findall(text.lower()):
        if CHINESE_RE.fullmatch(part):
            tokens.append(part)
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
            if len(part) >= 3:
                tokens.extend(part[i : i + 3] for i in range(len(part) - 2))
        else:
            tokens.append(part)
    return [token for token in tokens if token.strip()]


def _extract_query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for raw in TERM_RE.findall(query):
        term = raw.strip().lower()
        if len(term) < 2:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
        if CHINESE_RE.fullmatch(term) and len(term) > 4:
            max_size = min(6, len(term))
            min_size = 2
            for size in range(max_size, min_size - 1, -1):
                for index in range(len(term) - size + 1):
                    piece = term[index : index + size]
                    if piece not in seen:
                        seen.add(piece)
                        terms.append(piece)
    return terms


def _embed_query(query: str, settings) -> list[float]:
    client = _get_openai_client(settings.ai_embedding_api_key, settings.ai_embedding_base_url)
    response = client.embeddings.create(model=settings.ai_embedding_model, input=query)
    return response.data[0].embedding


def _hybrid_search(query: str, settings, top_k: int = 5) -> list[CitationItem]:
    client = _get_qdrant_client(settings.ai_qdrant_url, settings.ai_qdrant_path)
    vector_hits = client.query_points(
        collection_name=settings.ai_qdrant_collection_name,
        query=_embed_query(query, settings),
        limit=max(settings.ai_vector_top_k, top_k * 3),
        with_payload=True,
    ).points

    combined: dict[str, dict[str, Any]] = {}
    query_terms = _extract_query_terms(query)
    for rank, hit in enumerate(vector_hits):
        payload = hit.payload or {}
        source_file = str(payload.get("source_file") or "unknown")
        point_id = str(hit.id)
        combined[point_id] = {
            "id": point_id,
            "source_file": source_file,
            "source_name": _display_source_name(source_file),
            "text": str(payload.get("text") or ""),
            "images": payload.get("images") or [],
            "dense_rank_score": settings.ai_vector_weight / (rank + 60),
            "sparse_rank_score": 0.0,
        }

    if settings.ai_enable_bm25 and not settings.ai_qdrant_url:
        bm25, payloads = _build_bm25_cache(
            settings.ai_qdrant_url,
            settings.ai_qdrant_path,
            settings.ai_qdrant_collection_name,
        )
        bm25_scores = bm25.get_scores(_tokenize(query)) if payloads else []
        bm25_ranking = sorted(
            enumerate(bm25_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: max(settings.ai_bm25_top_k, top_k * 4)]

        for rank, (index, _) in enumerate(bm25_ranking):
            item = payloads[index]
            point_id = item["id"]
            entry = combined.setdefault(
                point_id,
                {
                    "id": point_id,
                    "source_file": item["source_file"],
                    "source_name": item["source_name"],
                    "text": item["text"],
                    "images": item["images"],
                    "dense_rank_score": 0.0,
                    "sparse_rank_score": 0.0,
                },
            )
            entry["sparse_rank_score"] += settings.ai_bm25_weight / (rank + 60)

    if query_terms:
        for rank, item in enumerate(
            _keyword_candidates_streaming(
                client,
                settings.ai_qdrant_collection_name,
                query_terms,
                limit=top_k * 4,
            )
        ):
            entry = combined.setdefault(
                item["id"],
                {
                    "id": item["id"],
                    "source_file": item["source_file"],
                    "source_name": item["source_name"],
                    "text": item["text"],
                    "images": item["images"],
                    "dense_rank_score": 0.0,
                    "sparse_rank_score": 0.0,
                },
            )
            entry["sparse_rank_score"] += 0.18 / (rank + 1)

    ranked: list[CitationItem] = []
    for item in combined.values():
        source_name = item["source_name"]
        text = item["text"]
        score = item["dense_rank_score"] + item["sparse_rank_score"]
        score += _keyword_overlap_boost(query_terms, source_name, text)
        ranked.append(
            CitationItem(
                source=source_name,
                text=text[:1000],
                score=round(score, 4),
                images=_normalize_image_urls(item["images"]),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def _keyword_overlap_boost(query_terms: list[str], source_name: str, text: str) -> float:
    if not query_terms:
        return 0.0

    source_lower = source_name.lower()
    text_lower = text.lower()
    boost = 0.0
    matched = 0

    for term in query_terms:
        if term in source_lower:
            boost += 0.14 if _is_precise_term(term) else 0.08
            matched += 1
        elif term in text_lower:
            boost += 0.06 if _is_precise_term(term) else 0.03
            matched += 1

    if matched >= 2:
        boost += 0.04 * min(matched, 4)
    if any(term.startswith("第") and term.endswith("窟") and term in text_lower for term in query_terms):
        boost += 0.2
    return min(boost, 0.42)


def _keyword_candidates_streaming(client, collection_name: str, query_terms: list[str], limit: int) -> list[dict[str, Any]]:
    heap: list[tuple[float, int, dict[str, Any]]] = []
    offset = None
    counter = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            source_file = str(payload.get("source_file") or "unknown")
            source_name = _display_source_name(source_file)
            text = str(payload.get("text") or "")
            score = _keyword_candidate_score(query_terms, source_name, text)
            if score <= 0:
                continue
            item = {
                "id": str(point.id),
                "source_file": source_file,
                "source_name": source_name,
                "text": text,
                "images": payload.get("images") or [],
            }
            counter += 1
            if len(heap) < limit:
                heappush(heap, (score, counter, item))
            else:
                if score > heap[0][0]:
                    heappop(heap)
                    heappush(heap, (score, counter, item))
        if offset is None:
            break
    ranked = sorted(heap, key=lambda row: row[0], reverse=True)
    return [item for _, _, item in ranked]


def _keyword_candidate_score(query_terms: list[str], source_name: str, text: str) -> float:
    source_lower = source_name.lower()
    text_lower = text.lower()[:2000]
    score = 0.0
    matched_source = 0
    for term in query_terms:
        if term in source_lower:
            score += 5.0 if _is_precise_term(term) else 2.5
            matched_source += 1
        elif term in text_lower:
            score += 1.0 if _is_precise_term(term) else 0.35
    if matched_source >= 2:
        score += 4.0
    if query_terms and all(term in source_lower for term in query_terms[: min(3, len(query_terms))]):
        score += 6.0
    return score


def _is_precise_term(term: str) -> bool:
    return bool(re.fullmatch(r"第\d+窟|\d+窟|[A-Za-z0-9._-]{3,}", term))


def _display_source_name(source_file: str) -> str:
    raw = (source_file or "").replace("\\", "/").strip()
    if not raw:
        return "未命名文献"

    name = raw.split("/")[-1] or raw
    lowered = name.lower()
    for suffix in DOC_SUFFIXES:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break

    name = name.strip().strip(".-_ ")
    if name:
        return name

    parent = raw.rstrip("/").split("/")[-2] if "/" in raw else raw
    return parent.strip() or "未命名文献"


def _normalize_image_urls(images: list[Any] | None) -> list[str]:
    results: list[str] = []
    for image in images or []:
        candidate = None
        if isinstance(image, dict):
            candidate = image.get("url") or image.get("image_url") or image.get("path")
        elif isinstance(image, str):
            candidate = image
        if not candidate:
            continue
        value = str(candidate).strip().replace("\\", "/")
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            results.append(value)
            continue
        results.append(f"/ai/images?path={quote(value, safe='/')}")
    return _dedupe(results, limit=4)


def _collect_images(citations: list[CitationItem]) -> list[str]:
    images: list[str] = []
    for citation in citations:
        images.extend(citation.images or [])
    return _dedupe(images, limit=8)


def _dedupe(items: list[str], limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _generate_answer(question: str, context_parts: list[str], settings) -> str:
    if not context_parts:
        return "当前未检索到可引用的站内资料，请尝试缩小问题范围或更换关键词。"

    joined_context = "\n\n".join(context_parts[:5])
    user_prompt = (
        f"用户问题：\n{question}\n\n"
        f"检索资料：\n{joined_context}\n\n"
        "请根据这些资料用中文作答，先直接回答问题，再概括关键依据。"
        "不要编造未出现在资料中的事实。"
    )

    client = _get_openai_client(settings.ai_llm_api_key, settings.ai_llm_base_url)
    response = client.chat.completions.create(
        model=settings.ai_llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip() or "暂未生成回答。"


def _generate_direct_answer(question: str, settings) -> str:
    client = _get_openai_client(settings.ai_llm_api_key, settings.ai_llm_base_url)
    response = client.chat.completions.create(
        model=settings.ai_llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": DIRECT_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return (response.choices[0].message.content or "").strip() or "暂未生成回答。"


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1 / (1 + exponent)
    exponent = math.exp(value)
    return exponent / (1 + exponent)
