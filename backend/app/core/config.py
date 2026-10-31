"""Application settings loaded from environment variables and .env files."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_DATABASE_URL = f"sqlite:///{(BACKEND_DIR / 'data' / 'clinicpilot.db').as_posix()}"


class Settings(BaseSettings):
    """Runtime configuration.

    Environment variables override values from either .env file. The root
    project .env is preferred, with backend/.env supported for local overrides.
    """

    app_name: str = "ClinicPilot AI"
    app_version: str = "0.1.0"
    database_url: str = DEFAULT_DATABASE_URL
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    backend_base_url: str = "http://127.0.0.1:8000"
    default_patient_id: str | None = Field(default=None, pattern=r"^P\d{3}$")

    rag_index_dir: str = str(BACKEND_DIR / "data" / "rag")
    rag_embedding_model: str = "BAAI/bge-small-en-v1.5"
    rag_embedding_dimension: int = 384
    rag_embedding_cache_dir: str = str(BACKEND_DIR / "data" / "fastembed")
    rag_min_score: float = 0.48
    rag_recommend_min_score: float = 0.55
    rag_recommend_margin: float = 0.03
    rag_mmr_lambda: float = 0.7
    rag_chunk_max_chars: int = 800
    rag_chunk_overlap: int = 120
    rag_rebuild_index: bool = False
    rag_rerank_enabled: bool = True
    rag_rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rag_rerank_min_score: float = -8.0

    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("default_patient_id", "anthropic_base_url", mode="before")
    @classmethod
    def _blank_value_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        """Return configured CORS origins as a clean list."""

        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
