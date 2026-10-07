from fastapi import APIRouter, Request

from app.core.response import success_response


router = APIRouter()


@router.get("/health")
def health(request: Request):
    return success_response({"status": "ok"}, request.state.request_id)
