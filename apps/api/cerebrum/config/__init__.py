"""
Cerebrum Settings — Pydantic V2 Configuration

All configuration is loaded from environment variables with full
type validation, defaults, and documentation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All secrets must be supplied via .env or environment — never hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ── Application ───────────────────────────────────────────────────────
    APP_NAME: str = "Cerebrum"
    APP_VERSION: str = "0.1.0"
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    APP_DEBUG: bool = False
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = Field(default=8000, ge=1, le=65535)
    APP_SECRET_KEY: str = Field(min_length=32)
    ALLOWED_HOSTS: list[str] = ["*"]

    # ── Database — PostgreSQL ─────────────────────────────────────────────
    DATABASE_URL: PostgresDsn
    DATABASE_POOL_SIZE: int = Field(default=20, ge=1, le=100)
    DATABASE_MAX_OVERFLOW: int = Field(default=10, ge=0, le=50)
    DATABASE_POOL_PRE_PING: bool = True

    # ── Cache — Redis ─────────────────────────────────────────────────────
    REDIS_URL: RedisDsn
    REDIS_CACHE_TTL: int = Field(default=3600, ge=60)

    # ── Object Storage — MinIO ────────────────────────────────────────────
    MINIO_HOST: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_DATASETS: str = "cerebrum-datasets"
    MINIO_BUCKET_MODELS: str = "cerebrum-models"
    MINIO_BUCKET_REPORTS: str = "cerebrum-reports"
    MINIO_SECURE: bool = False

    # ── Vector Database — Qdrant ──────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_MEMORY: str = "cerebrum_memory"
    QDRANT_COLLECTION_DOCUMENTS: str = "cerebrum_documents"

    # ── Graph Database — Neo4j ────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    # ── LLM Providers ─────────────────────────────────────────────────────
    OPENAI_API_KEY: str | None = None
    OPENAI_DEFAULT_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = Field(default=4096, ge=256)
    OPENAI_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama3.2"

    LLM_PROVIDER: Literal["openai", "anthropic", "ollama"] = "openai"
    LLM_FALLBACK_PROVIDER: Literal["openai", "anthropic", "ollama"] = "ollama"

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    EMBEDDING_BATCH_SIZE: int = Field(default=64, ge=1)

    # ── JWT Authentication ────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1)

    # ── OAuth2 ────────────────────────────────────────────────────────────
    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    GITHUB_CLIENT_ID: str | None = None
    GITHUB_CLIENT_SECRET: str | None = None
    OAUTH_REDIRECT_URL: str = "http://localhost:3000/auth/callback"

    # ── Celery ────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_TASK_ALWAYS_EAGER: bool = False  # Set True in tests

    # ── MLflow ────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "cerebrum"
    MLFLOW_ARTIFACT_ROOT: str = "s3://cerebrum-models/mlflow"

    # ── Observability ─────────────────────────────────────────────────────
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SERVICE_NAME: str = "cerebrum-api"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # ── Security ──────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1)
    RATE_LIMIT_BURST: int = Field(default=20, ge=1)
    MAX_FILE_UPLOAD_SIZE_MB: int = Field(default=500, ge=1)
    ALLOWED_FILE_TYPES: list[str] = ["csv", "xlsx", "json", "parquet", "pdf", "txt"]

    # ── Agent Configuration ───────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = Field(default=10, ge=1, le=50)
    AGENT_TIMEOUT_SECONDS: int = Field(default=300, ge=30)
    AGENT_MEMORY_WINDOW: int = Field(default=20, ge=5)
    PLANNER_MAX_SUBTASKS: int = Field(default=15, ge=1)
    CRITIC_HALLUCINATION_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)

    # ── Email ─────────────────────────────────────────────────────────────
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAIL_FROM: str = "noreply@cerebrum.ai"

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("APP_SECRET_KEY", "JWT_SECRET_KEY", mode="before")
    @classmethod
    def validate_secrets_not_default(cls, v: str) -> str:
        forbidden = {"CHANGE_ME", "secret", "password", "changeme"}
        if v.lower() in forbidden or v.startswith("CHANGE_ME"):
            msg = "Secret key must be changed from the default value"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_llm_provider_keys(self) -> Settings:
        """Ensure at least one LLM provider has a key configured."""
        has_openai = bool(self.OPENAI_API_KEY)
        has_anthropic = bool(self.ANTHROPIC_API_KEY)
        has_ollama = True  # Ollama doesn't need an API key

        if not any([has_openai, has_anthropic, has_ollama]):
            msg = "At least one LLM provider must be configured"
            raise ValueError(msg)

        if self.LLM_PROVIDER == "openai" and not has_openai:
            msg = "OPENAI_API_KEY required when LLM_PROVIDER=openai"
            raise ValueError(msg)

        if self.LLM_PROVIDER == "anthropic" and not has_anthropic:
            msg = "ANTHROPIC_API_KEY required when LLM_PROVIDER=anthropic"
            raise ValueError(msg)

        return self

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    @property
    def is_test(self) -> bool:
        return self.APP_ENV == "test"

    @property
    def minio_endpoint(self) -> str:
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns cached application settings.
    Uses lru_cache so the .env file is only read once.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()
