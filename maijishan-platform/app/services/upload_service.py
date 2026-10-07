import hashlib
import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.core.exceptions import ApiException
from app.db.models import Upload, UserUploadedDocument, UserUploadedDocumentFile
from app.db.session import engine
from app.services.user_upload_ingest_service import start_user_uploaded_document_ingest_job


DOCUMENT_EXTENSIONS = {".pdf", ".caj", ".doc", ".docx", ".txt"}


def ensure_user_uploaded_document_tables() -> None:
    UserUploadedDocument.__table__.create(bind=engine, checkfirst=True)
    UserUploadedDocumentFile.__table__.create(bind=engine, checkfirst=True)


def save_upload(db: Session, file: UploadFile, owner_id: int | None) -> tuple[Upload, UserUploadedDocument | None]:
    ensure_user_uploaded_document_tables()
    settings = get_settings()
    if file.content_type not in settings.allowed_mime_types:
        raise ApiException(code="invalid_mime", message="不允许的文件类型", status_code=400)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "file").name
    ext = Path(original_name).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = upload_dir / stored_name

    hasher = hashlib.sha256()
    total = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024

    with stored_path.open("wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                stored_path.unlink(missing_ok=True)
                raise ApiException(code="file_too_large", message="文件超过大小限制", status_code=400)
            hasher.update(chunk)
            buffer.write(chunk)

    upload = Upload(
        filename=original_name,
        file_hash=hasher.hexdigest(),
        size=total,
        mime_type=file.content_type,
        path=str(stored_path),
        owner_id=owner_id,
    )
    db.add(upload)
    db.flush()

    linked_document = None
    if ext in DOCUMENT_EXTENSIONS:
        linked_document = _create_user_uploaded_document(db, upload, owner_id)
        upload.related_type = "user_uploaded_document"
        upload.related_id = linked_document.id

    db.commit()
    db.refresh(upload)
    if linked_document is not None:
        db.refresh(linked_document)
        if ext == ".pdf":
            start_user_uploaded_document_ingest_job(linked_document.id)
    return upload, linked_document


def list_recent_user_uploaded_documents(db: Session, limit: int = 8) -> list[UserUploadedDocument]:
    limit = max(1, min(int(limit or 8), 24))
    return (
        db.query(UserUploadedDocument)
        .options(selectinload(UserUploadedDocument.files), selectinload(UserUploadedDocument.owner))
        .order_by(UserUploadedDocument.created_at.desc(), UserUploadedDocument.id.desc())
        .limit(limit)
        .all()
    )


def _create_user_uploaded_document(db: Session, upload: Upload, owner_id: int | None) -> UserUploadedDocument:
    settings = get_settings()
    source_path = Path(upload.path)
    ext = source_path.suffix.lower()
    target_dir = Path(settings.documents_dir) / "user_uploads" / (f"user_{owner_id}" if owner_id else "anonymous")
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = _safe_filename_stem(Path(upload.filename).stem)
    target_name = f"{safe_stem}_{upload.id}{ext}"
    target_path = target_dir / target_name
    shutil.copy2(source_path, target_path)

    document = UserUploadedDocument(
        owner_id=owner_id,
        title=Path(upload.filename).stem.strip() or upload.filename,
        source_type="user_upload",
        source_db="user_upload",
        abstract="用户上传文献，待解析摘要与作者信息。",
        ingestion_status="pending" if ext == ".pdf" else "done",
        ingestion_error=None,
    )
    db.add(document)
    db.flush()

    document_file = UserUploadedDocumentFile(
        document_id=document.id,
        upload_id=upload.id,
        filename=upload.filename,
        file_format=ext.lstrip(".") or "bin",
        mime_type=upload.mime_type,
        size=upload.size,
        path=str(target_path),
        is_primary=True,
    )
    db.add(document_file)
    return document


def _safe_filename_stem(value: str) -> str:
    text = (value or "document").strip()
    text = re.sub(r"[<>:\"/\\\\|?*]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text[:120] or "document"
