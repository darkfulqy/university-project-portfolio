from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base

from app.core.config import get_settings


def reflect_database():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    base = automap_base()
    base.prepare(engine, reflect=True)
    return base
