from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CommunityPostAttachment(Base):
    __tablename__ = "community_post_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(Integer, ForeignKey("community_posts.id"), index=True)
    upload_id: Mapped[int] = mapped_column(Integer, ForeignKey("uploads.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20), server_default="attachment")
    processing_status: Mapped[str] = mapped_column(String(20), server_default="pending")
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    linked_document_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
