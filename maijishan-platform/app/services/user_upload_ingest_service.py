import json
import threading
import time
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.db.models import UserUploadedDocument, UserUploadedDocumentFile
from app.db.session import SessionLocal


def start_user_uploaded_document_ingest_job(document_id: int) -> None:
    thread = threading.Thread(target=_process_user_uploaded_document, args=(document_id,), daemon=True)
    thread.start()


def _process_user_uploaded_document(document_id: int) -> None:
    db: Session = SessionLocal()
    try:
        document = (
            db.query(UserUploadedDocument)
            .options(selectinload(UserUploadedDocument.files))
            .filter(UserUploadedDocument.id == document_id)
            .first()
        )
        if not document:
            return

        primary_file = next((item for item in document.files if item.is_primary and item.file_format == "pdf"), None)
        if not primary_file:
            document.ingestion_status = "failed"
            document.ingestion_error = "primary_pdf_not_found"
            db.commit()
            return

        settings = get_settings()
        if not settings.mineru_token:
            document.ingestion_status = "failed"
            document.ingestion_error = "MINERU_TOKEN not configured"
            db.commit()
            return
        if not settings.qwen_api_key:
            document.ingestion_status = "failed"
            document.ingestion_error = "DASHSCOPE_API_KEY not configured"
            db.commit()
            return

        document.ingestion_status = "processing"
        document.ingestion_error = None
        db.commit()

        markdown_text = _run_mineru(primary_file.path, settings)
        metadata = _extract_with_qwen(markdown_text, primary_file.filename, settings)
        _apply_metadata(document, metadata, markdown_text)
        _upsert_markdown_file(db, document, primary_file, markdown_text)

        document.ingestion_status = "done"
        document.ingestion_error = None
        db.commit()
    except Exception as exc:  # noqa: BLE001
        document = (
            db.query(UserUploadedDocument)
            .options(selectinload(UserUploadedDocument.files))
            .filter(UserUploadedDocument.id == document_id)
            .first()
        )
        if document:
            document.ingestion_status = "failed"
            document.ingestion_error = str(exc)[:500]
            db.commit()
    finally:
        db.close()


def _json_request(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: int = 60) -> dict:
    response = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _run_mineru(file_path: str, settings) -> str:
    path = Path(file_path)
    payload = {
        "files": [{"name": path.name, "data_id": f"{path.stem[:24]}-{uuid.uuid4().hex[:8]}"}],
        "model_version": "vlm",
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }
    response = _json_request(
        "https://mineru.net/api/v4/file-urls/batch",
        payload,
        headers={"Authorization": f"Bearer {settings.mineru_token}"},
        timeout=60,
    )
    if response.get("code") != 0:
        raise RuntimeError(f"mineru_apply_failed: {response}")

    batch_data = response.get("data") or {}
    batch_id = batch_data.get("batch_id")
    file_urls = batch_data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise RuntimeError("mineru_missing_upload_url")

    with path.open("rb") as handle:
        upload_response = requests.put(file_urls[0], data=handle, timeout=600)
    upload_response.raise_for_status()

    deadline = time.time() + settings.mineru_poll_timeout_seconds
    result_item = None
    while time.time() < deadline:
        poll_response = requests.get(
            f"https://mineru.net/api/v4/extract-results/batch/{batch_id}",
            headers={"Authorization": f"Bearer {settings.mineru_token}"},
            timeout=60,
        )
        poll_response.raise_for_status()
        poll_response = poll_response.json()

        if poll_response.get("code") != 0:
            raise RuntimeError(f"mineru_poll_failed: {poll_response}")

        extract_result = (poll_response.get("data") or {}).get("extract_result") or []
        if extract_result:
            result_item = extract_result[0]
            state = result_item.get("state")
            if state == "done":
                break
            if state == "failed":
                raise RuntimeError(result_item.get("err_msg") or "mineru_extract_failed")

        time.sleep(settings.mineru_poll_interval_seconds)

    if not result_item or result_item.get("state") != "done":
        raise RuntimeError("mineru_timeout")

    zip_url = result_item.get("full_zip_url")
    if not zip_url:
        raise RuntimeError("mineru_missing_zip_url")

    with TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / f"{path.stem}.zip"
        zip_response = requests.get(zip_url, timeout=600)
        zip_response.raise_for_status()
        zip_path.write_bytes(zip_response.content)

        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        markdown_files = list(extract_dir.rglob("*.md"))
        if not markdown_files:
            raise RuntimeError("mineru_markdown_not_found")
        return markdown_files[0].read_text(encoding="utf-8", errors="ignore")


def _extract_with_qwen(markdown_text: str, filename: str, settings) -> dict:
    prompt = (
        "You extract bibliographic metadata from PDF markdown and return JSON only. "
        "Return an object with keys: title, author, publication_date, source_type, source_db, "
        "abstract, keywords, doi, isbn, issn, volume, issue, pages, school, advisor, author_unit. "
        "Use null for unknown fields. keywords must be an array of strings.\n\n"
        f"Filename: {filename}\n\n"
        f"Markdown:\n{markdown_text[:12000]}"
    )
    payload = {
        "model": settings.qwen_model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    response = _json_request(
        settings.qwen_api_base,
        payload,
        headers={"Authorization": f"Bearer {settings.qwen_api_key}"},
        timeout=settings.qwen_timeout_seconds,
    )

    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    content = str(content).strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.startswith("```")]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _apply_metadata(document: UserUploadedDocument, metadata: dict, markdown_text: str) -> None:
    keywords = metadata.get("keywords")
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",") if item.strip()]
    elif not isinstance(keywords, list):
        keywords = []

    document.title = (metadata.get("title") or document.title or "未命名文献")[:255]
    document.author = _clean_value(metadata.get("author"))
    document.publication_date = _clean_value(metadata.get("publication_date"))
    document.source_type = _clean_value(metadata.get("source_type")) or "user_upload_pdf"
    document.source_db = _clean_value(metadata.get("source_db")) or "user_upload"
    document.abstract = _clean_value(metadata.get("abstract")) or markdown_text[:2000]
    document.keywords = ",".join(str(item).strip() for item in keywords if str(item).strip()) or None


def _upsert_markdown_file(
    db: Session,
    document: UserUploadedDocument,
    primary_file: UserUploadedDocumentFile,
    markdown_text: str,
) -> None:
    markdown_path = Path(primary_file.path).with_suffix(".md")
    markdown_path.write_text(markdown_text, encoding="utf-8")

    file_record = next((item for item in document.files if item.file_format == "md"), None)
    if file_record is None:
        file_record = UserUploadedDocumentFile(
            document_id=document.id,
            upload_id=primary_file.upload_id,
            filename=f"{Path(primary_file.filename).stem}.md",
            file_format="md",
            mime_type="text/markdown",
            size=len(markdown_text.encode("utf-8")),
            path=str(markdown_path),
            is_primary=False,
        )
        db.add(file_record)
    else:
        file_record.filename = f"{Path(primary_file.filename).stem}.md"
        file_record.mime_type = "text/markdown"
        file_record.size = len(markdown_text.encode("utf-8"))
        file_record.path = str(markdown_path)


def _clean_value(value) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
