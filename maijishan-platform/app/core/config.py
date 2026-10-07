from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=True)

    db_host: str = Field(alias="DB_HOST")
    db_port: int = Field(alias="DB_PORT")
    db_user: str = Field(alias="DB_USER")
    db_password: str = Field(alias="DB_PASSWORD")
    db_name: str = Field(alias="DB_NAME")
    db_charset: str = Field(alias="DB_CHARSET")

    upload_dir: str = Field(alias="UPLOAD_DIR")
    documents_dir: str = Field(default="/data/maijidownloads", alias="DOCUMENTS_DIR")

    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(alias="JWT_ALGORITHM")
    jwt_expires_minutes: int = Field(alias="JWT_EXPIRES_MINUTES")

    max_upload_mb: int = Field(alias="MAX_UPLOAD_MB")
    upload_allowed_mime: str = Field(alias="UPLOAD_ALLOWED_MIME")

    db_wait_timeout_seconds: int = Field(alias="DB_WAIT_TIMEOUT_SECONDS")
    db_wait_interval_seconds: int = Field(alias="DB_WAIT_INTERVAL_SECONDS")
    homepage_stats_enabled: bool = Field(default=True, alias="HOMEPAGE_STATS_ENABLED")
    homepage_stats_cache_ttl_seconds: int = Field(default=600, alias="HOMEPAGE_STATS_CACHE_TTL_SECONDS")
    mineru_token: str | None = Field(default=None, alias="MINERU_TOKEN")
    mineru_poll_interval_seconds: int = Field(default=5, alias="MINERU_POLL_INTERVAL_SECONDS")
    mineru_poll_timeout_seconds: int = Field(default=1800, alias="MINERU_POLL_TIMEOUT_SECONDS")

    qwen_api_key: str | None = Field(default=None, alias="DASHSCOPE_API_KEY")
    qwen_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        alias="QWEN_API_BASE",
    )
    qwen_model: str = Field(default="qwen-plus", alias="QWEN_MODEL")
    qwen_timeout_seconds: int = Field(default=120, alias="QWEN_TIMEOUT_SECONDS")

    ai_enabled: bool = Field(default=False, alias="AI_ENABLED")
    ai_qdrant_url: str | None = Field(default=None, alias="AI_QDRANT_URL")
    ai_qdrant_path: str | None = Field(
        default="maijishan_runnable_bundle_20260312_172500/qdrant_storage",
        alias="AI_QDRANT_PATH",
    )
    ai_qdrant_collection_name: str = Field(default="maijishan_documents", alias="AI_QDRANT_COLLECTION_NAME")
    ai_embedding_api_key: str | None = Field(default=None, alias="AI_EMBEDDING_API_KEY")
    ai_embedding_base_url: str | None = Field(default=None, alias="AI_EMBEDDING_BASE_URL")
    ai_embedding_model: str | None = Field(default=None, alias="AI_EMBEDDING_MODEL")
    ai_llm_api_key: str | None = Field(default=None, alias="AI_LLM_API_KEY")
    ai_llm_base_url: str | None = Field(default=None, alias="AI_LLM_BASE_URL")
    ai_llm_model: str | None = Field(default=None, alias="AI_LLM_MODEL")
    ai_vector_top_k: int = Field(default=12, alias="AI_VECTOR_TOP_K")
    ai_bm25_top_k: int = Field(default=12, alias="AI_BM25_TOP_K")
    ai_result_top_k: int = Field(default=5, alias="AI_RESULT_TOP_K")
    ai_vector_weight: float = Field(default=0.7, alias="AI_VECTOR_WEIGHT")
    ai_bm25_weight: float = Field(default=0.3, alias="AI_BM25_WEIGHT")
    ai_enable_bm25: bool = Field(default=True, alias="AI_ENABLE_BM25")

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?charset={self.db_charset}"
        )

    @property
    def allowed_mime_types(self) -> list[str]:
        return [item.strip() for item in self.upload_allowed_mime.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
