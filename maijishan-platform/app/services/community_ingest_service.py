import json
import threading
import time
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import CommunityPostAttachment, Document, Upload
from app.db.session import SessionLocal, engine


def ensure_community_attachment_table() -> None:
    CommunityPostAttachment.__table__.create(bind=engine, checkfirst=True)


def start_pdf_ingest_job(attachment_id: int) -> None:
    thread = threading.Thread(target=_process_pdf_attachment, args=(attachment_id,), daemon=True)
    thread.start()


def _process_pdf_attachment(attachment_id: int) -> None:
    db: Session = SessionLocal()
    try:
        attachment = db.query(CommunityPostAttachment).filter(CommunityPostAttachment.id == attachment_id).first()
        if not attachment:
            return

        upload = db.query(Upload).filter(Upload.id == attachment.upload_id).first()
        if not upload:
            attachment.processing_status = "failed"
            attachment.processing_error = "upload_not_found"
            db.commit()
            return

        settings = get_settings()
        if not settings.mineru_token:
            attachment.processing_status = "failed"
            attachment.processing_error = "MINERU_TOKEN not configured"
            db.commit()
            return
        if not settings.qwen_api_key:
            attachment.processing_status = "failed"
            attachment.processing_error = "DASHSCOPE_API_KEY not configured"
            db.commit()
            return

        attachment.processing_status = "processing"
        attachment.processing_error = None
        db.commit()

        markdown = _run_mineru(upload.path, settings)
        metadata = _extract_with_qwen(markdown, upload.filename, settings)
        document = _create_document_from_metadata(db, metadata, upload, markdown)

        attachment.processing_status = "done"
        attachment.processing_error = None
        attachment.linked_document_id = document.id
        db.commit()
    except Exception as exc:  # noqa: BLE001
        attachment = db.query(CommunityPostAttachment).filter(CommunityPostAttachment.id == attachment_id).first()
        if attachment:
            attachment.processing_status = "failed"
            attachment.processing_error = str(exc)[:500]
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


def _create_document_from_metadata(db: Session, metadata: dict, upload: Upload, markdown_text: str) -> Document:
    keywords = metadata.get("keywords")
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",") if item.strip()]
    elif not isinstance(keywords, list):
        keywords = []

    document = Document(
        source_group_id=999999,
        title=(metadata.get("title") or Path(upload.filename).stem)[:255],
        author=metadata.get("author"),
        publication_date=metadata.get("publication_date"),
        source_type=metadata.get("source_type") or "community_upload_pdf",
        source_db=metadata.get("source_db") or "community_post",
        abstract=(metadata.get("abstract") or markdown_text[:2000]),
        keywords=",".join(str(item).strip() for item in keywords if str(item).strip()) or None,
        doi=metadata.get("doi"),
        isbn=metadata.get("isbn"),
        issn=metadata.get("issn"),
        volume=metadata.get("volume"),
        issue=metadata.get("issue"),
        pages=metadata.get("pages"),
        school=metadata.get("school"),
        advisor=metadata.get("advisor"),
        author_unit=metadata.get("author_unit"),
        url=f"/uploads/{upload.id}/file",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document
