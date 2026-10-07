import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.api.deps import get_request_id
from app.core.response import success_response
from app.schemas.ai import AiChatRequest, AiSearchRequest
from app.services.ai_service import chat, search


router = APIRouter(prefix="/ai", tags=["ai"])
AI_IMAGES_BASE_DIR = Path("/data/maijidownloads")
AI_IMAGE_IMPORTS_DIR = AI_IMAGES_BASE_DIR / "ai_mineru_images"
AI_IMAGE_MAPPING_FILE = AI_IMAGES_BASE_DIR / "ai_image_redirects.json"


def _resolve_under_base(base: Path, relative_path: str) -> Path:
    candidate = (base / relative_path).resolve(strict=False)
    resolved_base = base.resolve(strict=False)
    if resolved_base not in candidate.parents and candidate != resolved_base:
        raise HTTPException(status_code=400, detail="Invalid image path")
    return candidate


@lru_cache(maxsize=1)
def _load_image_redirects() -> dict[str, str]:
    if not AI_IMAGE_MAPPING_FILE.exists():
        return {}
    try:
        raw = json.loads(AI_IMAGE_MAPPING_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    redirects: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, str):
                redirects[key.replace("\\", "/").lstrip("/")] = value.replace("\\", "/").lstrip("/")
    return redirects


@router.post("/chat")
def chat_api(request: Request, payload: AiChatRequest):
    response = chat(payload)
    return success_response(response.model_dump(), get_request_id(request))


@router.post("/search")
def search_api(request: Request, payload: AiSearchRequest):
    response = search(payload.query, top_k=payload.top_k)
    return success_response(response.model_dump(), get_request_id(request))


@router.get("/images")
def get_ai_image(path: str = Query(..., min_length=1)):
    decoded = unquote(path).replace("\\", "/").lstrip("/")
    candidate = _resolve_under_base(AI_IMAGES_BASE_DIR, decoded)
    if not candidate.exists() or not candidate.is_file():
        redirect = _load_image_redirects().get(decoded)
        if redirect:
            candidate = _resolve_under_base(AI_IMAGES_BASE_DIR, redirect)
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(candidate))
