from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EmbeddedDocumentFile(Base):
    __tablename__ = "embedded_document_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    storage_key: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(String(512), nullable=False)
    file_ext: Mapped[str] = mapped_column(String(16), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    chunks = relationship(
        "EmbeddedDocumentFileChunk",
        back_populates="file",
        cascade="all, delete-orphan",
        order_by="EmbeddedDocumentFileChunk.chunk_index",
    )

    __table_args__ = (
        Index("ix_embedded_document_files_normalized_title", "normalized_title"),
    )


class EmbeddedDocumentFileChunk(Base):
    __tablename__ = "embedded_document_file_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[int] = mapped_column(Integer, ForeignKey("embedded_document_files.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content_base64: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    file = relationship("EmbeddedDocumentFile", back_populates="chunks")

    __table_args__ = (
        Index("ix_embedded_document_file_chunks_file_id_chunk_index", "file_id", "chunk_index", unique=True),
    )
