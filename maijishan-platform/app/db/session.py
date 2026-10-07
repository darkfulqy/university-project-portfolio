import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def wait_for_db() -> None:
    timeout = settings.db_wait_timeout_seconds
    interval = settings.db_wait_interval_seconds
    start = time.time()
    last_error = None
    while True:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except Exception as exc:
            last_error = exc
            if time.time() - start >= timeout:
                raise RuntimeError("database_not_ready") from last_error
            time.sleep(interval)
