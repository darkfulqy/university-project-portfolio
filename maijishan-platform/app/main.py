from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.core.exceptions import ApiException
from app.core.response import error_response
from app.db.session import wait_for_db
from app.services.embedded_document_file_service import ensure_embedded_document_file_tables
from app.services.upload_service import ensure_user_uploaded_document_tables
from app.utils.request_id import ensure_request_id
from app.api.routes import ai, auth, community, documents, health, posts, stats, tags, uploads
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="麦积山文献与帖子网站后端")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = ensure_request_id(request)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(ApiException)
async def api_exception_handler(request: Request, exc: ApiException):
    return error_response({"code": exc.code, "message": exc.message}, request.state.request_id, exc.status_code)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "http_error"
    return error_response({"code": "http_error", "message": detail}, request.state.request_id, exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return error_response({"code": "internal_error", "message": "服务器内部错误"}, request.state.request_id, 500)


@app.on_event("startup")
def on_startup():
    get_settings()
    wait_for_db()
    ensure_user_uploaded_document_tables()
    ensure_embedded_document_file_tables()


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(posts.router)
app.include_router(documents.router)
app.include_router(uploads.router)
app.include_router(tags.router)
app.include_router(ai.router)
app.include_router(stats.router)
app.include_router(community.router)
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")
app.mount("/shouye", StaticFiles(directory="static/shouye"), name="shouye")


@app.get("/", include_in_schema=False)
def spa_index():
    return FileResponse("static/index.html")


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    api_roots = {
        "documents",
        "posts",
        "auth",
        "uploads",
        "health",
        "tags",
        "ai",
        "stats",
        "community",
        "openapi.json",
        "docs",
        "redoc",
    }
    first_segment = full_path.split("/", 1)[0]
    if first_segment in api_roots:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse("static/index.html")
