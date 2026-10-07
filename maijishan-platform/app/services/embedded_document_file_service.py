from app.db.models import EmbeddedDocumentFile, EmbeddedDocumentFileChunk
from app.db.session import engine


def ensure_embedded_document_file_tables() -> None:
    EmbeddedDocumentFile.__table__.create(bind=engine, checkfirst=True)
    EmbeddedDocumentFileChunk.__table__.create(bind=engine, checkfirst=True)
