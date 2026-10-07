from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import CommunityPost, CommunityPostAttachment, Document, Upload, User
from app.db.session import get_db
from app.services.community_ingest_service import ensure_community_attachment_table, start_pdf_ingest_job
from app.services.upload_service import save_upload

router = APIRouter(prefix="/community", tags=["community"])


def _serialize_attachment(attachment: CommunityPostAttachment, upload: Upload, document: Document | None) -> dict:
    return {
        "id": attachment.id,
        "upload_id": upload.id,
        "kind": attachment.kind,
        "filename": upload.filename,
        "mime_type": upload.mime_type,
        "size": upload.size,
        "url": f"/uploads/{upload.id}/file",
        "processing_status": attachment.processing_status,
        "processing_error": attachment.processing_error,
        "linked_document_id": attachment.linked_document_id,
        "linked_document_url": f"/detail/documents/{document.id}" if document else None,
    }


def _serialize_post(db: Session, post: CommunityPost) -> dict:
    attachments = db.query(CommunityPostAttachment).filter(CommunityPostAttachment.post_id == post.id).all()
    upload_ids = [item.upload_id for item in attachments]
    uploads = db.query(Upload).filter(Upload.id.in_(upload_ids)).all() if upload_ids else []
    upload_map = {item.id: item for item in uploads}
    document_ids = [item.linked_document_id for item in attachments if item.linked_document_id]
    documents = db.query(Document).filter(Document.id.in_(document_ids)).all() if document_ids else []
    document_map = {item.id: item for item in documents}
    return {
        "id": post.id,
        "author_id": post.author_id,
        "title": post.title,
        "content": post.content,
        "views_count": post.views_count,
        "likes_count": post.likes_count,
        "comments_count": post.comments_count,
        "attachments": [
            _serialize_attachment(item, upload_map[item.upload_id], document_map.get(item.linked_document_id))
            for item in attachments
            if item.upload_id in upload_map
        ],
        "created_at": post.created_at,
        "updated_at": post.updated_at,
    }


@router.post("/")
def create_post(
    title: str = Form(...),
    content: str = Form(...),
    images: list[UploadFile] = File(default=[]),
    attachments: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensure_community_attachment_table()
    post = CommunityPost(author_id=current_user.id, title=title, content=content)
    db.add(post)
    db.commit()
    db.refresh(post)

    pending_pdf_attachment_ids: list[int] = []
    for kind, files in (("image", images), ("attachment", attachments)):
        for file in files:
            upload = save_upload(db, file, current_user.id)
            upload.related_type = "community_post"
            upload.related_id = post.id
            attachment = CommunityPostAttachment(
                post_id=post.id,
                upload_id=upload.id,
                kind=kind,
                processing_status="pending" if upload.filename.lower().endswith(".pdf") else "done",
            )
            db.add(attachment)
            db.commit()
            db.refresh(attachment)
            if upload.filename.lower().endswith(".pdf"):
                pending_pdf_attachment_ids.append(attachment.id)

    for attachment_id in pending_pdf_attachment_ids:
        start_pdf_ingest_job(attachment_id)

    return _serialize_post(db, post)


@router.get("/")
def get_posts(page: int = 1, page_size: int = 20, db: Session = Depends(get_db)):
    ensure_community_attachment_table()
    offset = (page - 1) * page_size
    total = db.execute(select(func.count()).select_from(CommunityPost)).scalar()
    posts = (
        db.execute(select(CommunityPost).order_by(CommunityPost.created_at.desc()).offset(offset).limit(page_size))
        .scalars()
        .all()
    )
    return {"total": total, "posts": [_serialize_post(db, post) for post in posts]}


@router.get("/{post_id}")
def get_post(post_id: int, db: Session = Depends(get_db)):
    ensure_community_attachment_table()
    post = db.execute(select(CommunityPost).where(CommunityPost.id == post_id)).scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="帖子不存在")
    post.views_count += 1
    db.commit()
    db.refresh(post)
    return _serialize_post(db, post)


@router.post("/{post_id}/like")
def like_post(post_id: int, db: Session = Depends(get_db)):
    post = db.execute(select(CommunityPost).where(CommunityPost.id == post_id)).scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="帖子不存在")
    post.likes_count += 1
    db.commit()
    return {"likes_count": post.likes_count}
