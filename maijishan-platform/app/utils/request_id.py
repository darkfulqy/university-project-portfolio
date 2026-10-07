import uuid
from fastapi import Request


def ensure_request_id(request: Request) -> str:
    header_value = request.headers.get("X-Request-ID")
    request_id = header_value if header_value else str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")
