from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.core.exceptions import ApiException
from app.db.models import Post


def create_post(
    db: Session,
    author_name: str,
    title: str | None,
    content: str | None,
    platform: str,
    post_url: str | None,
) -> Post:
    post = Post(
        title=title,
        content=content,
        platform=platform,
        post_url=post_url,
        author_name=author_name,
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def list_posts(
    db: Session,
    page: int,
    page_size: int,
    keyword: str | None,
):
    query = db.query(Post)
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(or_(Post.title.like(like), Post.content.like(like)))
    total = query.count()
    items = query.order_by(Post.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return items, total


def get_post(db: Session, post_id: int) -> Post:
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise ApiException(code="post_not_found", message="帖子不存在", status_code=404)
    return post


def update_post(
    db: Session,
    post: Post,
    title: str | None,
    content: str | None,
    platform: str | None,
    post_url: str | None,
) -> Post:
    if title is not None:
        post.title = title
    if content is not None:
        post.content = content
    if platform is not None:
        post.platform = platform
    if post_url is not None:
        post.post_url = post_url
    db.commit()
    db.refresh(post)
    return post


def delete_post(db: Session, post: Post) -> None:
    db.delete(post)
    db.commit()
