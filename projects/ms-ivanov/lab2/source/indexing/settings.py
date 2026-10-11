"""Connection settings of the databases (``.env`` / environment variables)."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from source import config


class InfraSettings(BaseSettings):
    """Credentials and ports; names match the variables used by docker compose."""

    model_config = SettingsConfigDict(
        env_file=config.PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: str
    postgres_db: str = "rag"

    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
