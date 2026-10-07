from datetime import datetime
from pydantic import BaseModel


class PostCreate(BaseModel):
    title: str | None = None
    content: str | None = None
    platform: str
    post_url: str | None = None


class PostUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    platform: str | None = None
    post_url: str | None = None


class PostOut(BaseModel):
    id: int
    title: str | None
    content: str | None
    platform: str
    post_url: str | None
    author_name: str | None
    likes_count: int | None
    comments_count: int | None
    shares_count: int | None
    views_count: int | None
    publish_date: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
