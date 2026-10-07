from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password, create_access_token
from app.core.exceptions import ApiException
from app.db.models import User


def register_user(db: Session, username: str, email: str, password: str) -> User:
    existing = (
        db.query(User)
        .filter((User.username == username) | (User.email == email))
        .first()
    )
    if existing:
        raise ApiException(code="user_exists", message="用户名或邮箱已存在", status_code=400)
    user = User(username=username, email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> tuple[User, str]:
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        raise ApiException(code="invalid_credentials", message="用户名或密码错误", status_code=401)
    token = create_access_token(str(user.id))
    return user, token
