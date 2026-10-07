from sqlalchemy.orm import Session

from app.core.exceptions import ApiException
from app.db.models import Tag


def list_tags(db: Session) -> list[Tag]:
    return db.query(Tag).order_by(Tag.name.asc()).all()


def create_tag(db: Session, name: str) -> Tag:
    existing = db.query(Tag).filter(Tag.name == name).first()
    if existing:
        raise ApiException(code="tag_exists", message="标签已存在", status_code=400)
    tag = Tag(name=name)
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return tag
