from typing import Any
from fastapi.responses import JSONResponse


def success_response(data: Any, request_id: str, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"data": data, "error": None, "request_id": request_id},
    )


def error_response(error: dict, request_id: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"data": None, "error": error, "request_id": request_id},
    )
