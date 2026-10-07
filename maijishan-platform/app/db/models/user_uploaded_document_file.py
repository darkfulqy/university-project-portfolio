from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class UserUploadedDocumentFile(Base):
    __tablename__ = "user_uploaded_document_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_uploaded_documents.id"), nullable=False)
    upload_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("uploads.id"), nullable=True)
    filename: Mapped[str] = mapped_column(Text)
    file_format: Mapped[str] = mapped_column(String(16))
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    path: Mapped[str] = mapped_column(Text)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    document = relationship("UserUploadedDocument", back_populates="files")
    upload = relationship("Upload")
