from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_DB_PATH = Path("data/ai_potential_stock_discovery.sqlite")


@dataclass(frozen=True)
class Settings:
    db_path: Path
    sec_user_agent: str
    local_market_source_dir: Path = Path(".")
    fmp_api_key: str | None = None
    newsapi_key: str | None = None
    fred_api_key: str | None = None
    eia_api_key: str | None = None


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> Settings:
    load_dotenv()
    db_path = Path(os.getenv("AI_STOCK_DB", str(DEFAULT_DB_PATH)))
    return Settings(
        db_path=db_path,
        sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip(),
        local_market_source_dir=Path(os.getenv("LOCAL_MARKET_SOURCE_DIR", ".")),
        fmp_api_key=_empty_to_none(os.getenv("FMP_API_KEY")),
        newsapi_key=_empty_to_none(os.getenv("NEWSAPI_KEY")),
        fred_api_key=_empty_to_none(os.getenv("FRED_API_KEY")),
        eia_api_key=_empty_to_none(os.getenv("EIA_API_KEY")),
    )


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
