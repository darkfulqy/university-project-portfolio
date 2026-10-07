from datetime import datetime

from pydantic import BaseModel


class CommunityPostAttachmentOut(BaseModel):
    id: int
    upload_id: int
    kind: str
    filename: str
    mime_type: str
    size: int
    url: str
    processing_status: str
    processing_error: str | None = None
    linked_document_id: int | None = None
    linked_document_url: str | None = None


class CommunityPostBase(BaseModel):
    title: str
    content: str


class CommunityPostCreate(CommunityPostBase):
    pass


class CommunityPostUpdate(BaseModel):
    title: str | None = None
    content: str | None = None


class CommunityPostResponse(CommunityPostBase):
    id: int
    author_id: int
    views_count: int
    likes_count: int
    comments_count: int
    attachments: list[CommunityPostAttachmentOut] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CommunityPostListResponse(BaseModel):
    total: int
    posts: list[CommunityPostResponse]
