from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "Aletheia API"
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # Database
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "aletheia"
    postgres_user: str = "aletheia"
    postgres_password: str = "aletheia_dev_password"

    # Models
    ultraface_path: str = "/app/models/ultraface.onnx"
    arcface_path: str = "/app/models/arcface.onnx"
    classifier_path: str = "/app/models/mobilenet.onnx"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_queue_name: str = "aletheia_jobs"
    image_ttl: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        """Construct async database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Construct sync database URL (for Alembic)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
