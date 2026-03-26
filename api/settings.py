from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "NeuralGCM Weather API"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    api_rate_limit: str = "60/minute"
    api_key_header: str = "X-API-Key"
    secret_key: str = "change-this-in-production-use-openssl-rand-hex-32"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "neuralgcm_weather"
    postgres_user: str = "neuralgcm"
    postgres_password: str = "neuralgcm_dev_password"

    @property
    def database_url(self) -> str:
        return (f"postgresql+asyncpg://{self.postgres_user}:"
                f"{self.postgres_password}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}")

    @property
    def database_url_sync(self) -> str:
        return (f"postgresql+psycopg2://{self.postgres_user}:"
                f"{self.postgres_password}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}")

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    cache_ttl_seconds: int = 21600  # 6 hours (matches forecast cadence)
    cache_proximity_km: float = 50.0

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_timeout: int = 300  # 5 minutes max per forecast

    # NeuralGCM
    neuralgcm_model: str = "v1/deterministic_2_8_deg.pkl"
    neuralgcm_default_days: int = 5
    neuralgcm_max_days: int = 10
    neuralgcm_cache_dir: str = "./cache"

    # ECMWF
    ecmwf_lag_hours: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
