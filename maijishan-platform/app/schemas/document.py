from datetime import datetime
from pydantic import BaseModel


class DocumentCreate(BaseModel):
    source_group_id: int
    title: str
    author: str | None = None
    publication_date: str | None = None
    source_type: str | None = None
    source_db: str | None = None
    abstract: str | None = None
    keywords: list[str] | None = None
    doi: str | None = None
    isbn: str | None = None
    issn: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    school: str | None = None
    advisor: str | None = None
    author_unit: str | None = None
    url: str | None = None


class DocumentUpdate(BaseModel):
    source_group_id: int | None = None
    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    source_type: str | None = None
    source_db: str | None = None
    abstract: str | None = None
    keywords: list[str] | None = None
    doi: str | None = None
    isbn: str | None = None
    issn: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    school: str | None = None
    advisor: str | None = None
    author_unit: str | None = None
    url: str | None = None


class DocumentOut(BaseModel):
    id: int
    source_group_id: int
    title: str
    author: str | None
    publication_date: str | None
    source_type: str | None
    source_db: str | None
    abstract: str | None
    keywords: list[str] | None = None
    doi: str | None
    isbn: str | None
    issn: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    school: str | None
    advisor: str | None
    author_unit: str | None
    url: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
