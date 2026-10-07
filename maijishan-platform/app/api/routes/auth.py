from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_request_id
from app.core.response import success_response
from app.db.session import get_db
from app.schemas.auth import RegisterRequest, LoginRequest
from app.schemas.user import UserOut
from app.services.auth_service import register_user, authenticate_user


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    user = register_user(db, payload.username, payload.email, payload.password)
    data = {"user": UserOut.model_validate(user).model_dump(mode="json")}
    return success_response(data, get_request_id(request), status_code=201)


@router.post("/login")
def login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    user, token = authenticate_user(db, payload.username, payload.password)
    data = {
        "access_token": token,
        "token_type": "bearer",
        "user": UserOut.model_validate(user).model_dump(mode="json"),
    }
    return success_response(data, get_request_id(request))
