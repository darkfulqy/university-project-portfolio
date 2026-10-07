from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_request_id
from app.core.response import success_response
from app.db.models import User
from app.db.session import get_db
from app.schemas.tag import TagCreate
from app.services.tag_service import list_tags, create_tag


router = APIRouter(prefix="/tags", tags=["tags"])


@router.get("")
def list_tags_api(request: Request, db: Session = Depends(get_db)):
    tags = list_tags(db)
    data = [{"id": tag.id, "name": tag.name} for tag in tags]
    return success_response(data, get_request_id(request))


@router.post("")
def create_tag_api(
    request: Request,
    payload: TagCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tag = create_tag(db, payload.name)
    data = {"id": tag.id, "name": tag.name}
    return success_response(data, get_request_id(request), status_code=201)
