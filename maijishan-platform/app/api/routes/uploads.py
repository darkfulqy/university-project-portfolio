from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_request_id
from app.core.exceptions import ApiException
from app.core.response import success_response
from app.db.models import Upload, User, UserUploadedDocument
from app.db.session import get_db
from app.services.upload_service import list_recent_user_uploaded_documents, save_upload


router = APIRouter(prefix="/uploads", tags=["uploads"])


def _serialize_recent_document(item: UserUploadedDocument) -> dict:
    primary_file = next((file for file in item.files if file.is_primary), None) or (item.files[0] if item.files else None)
    available_file_types = sorted({file.file_format for file in item.files})
    document_id = -item.id
    return {
        "id": item.id,
        "document_id": document_id,
        "title": item.title,
        "author": item.author,
        "publication_date": item.publication_date,
        "source_type": item.source_type,
        "source_db": item.source_db,
        "abstract": item.abstract,
        "ingestion_status": item.ingestion_status,
        "ingestion_error": item.ingestion_error,
        "detail_url": f"/detail/documents/{document_id}",
        "document_api_url": f"/documents/{document_id}",
        "preview_url": f"/documents/{document_id}/file" if primary_file and primary_file.file_format == "pdf" else None,
        "download_url": f"/documents/{document_id}/file?download=1" if primary_file else None,
        "available_file_types": available_file_types,
        "file_format": primary_file.file_format if primary_file else None,
        "filename": primary_file.filename if primary_file else None,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


@router.get("")
def list_recent_uploads_api(request: Request, limit: int = 8, db: Session = Depends(get_db)):
    items = list_recent_user_uploaded_documents(db, limit)
    data = {"items": [_serialize_recent_document(item) for item in items], "total": len(items)}
    return success_response(data, get_request_id(request))


@router.post("")
def upload_file_api(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    upload, linked_document = save_upload(db, file, current_user.id)
    data = {
        "id": upload.id,
        "filename": upload.filename,
        "size": upload.size,
        "mime_type": upload.mime_type,
        "url": f"/uploads/{upload.id}/file",
        "document_id": (-linked_document.id if linked_document else None),
        "document_url": (f"/detail/documents/{-linked_document.id}" if linked_document else None),
        "ingestion_status": (linked_document.ingestion_status if linked_document else None),
        "ingestion_error": (linked_document.ingestion_error if linked_document else None),
        "document_title": (linked_document.title if linked_document else None),
    }
    return success_response(data, get_request_id(request), status_code=201)


@router.get("/{upload_id}/file")
def download_file_api(
    request: Request,
    upload_id: int,
    db: Session = Depends(get_db),
):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise ApiException(code="upload_not_found", message="文件不存在", status_code=404)
    return FileResponse(path=upload.path, media_type=upload.mime_type, filename=upload.filename)
