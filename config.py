"""Configuration settings for the diarization service."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # HuggingFace token for pyannote models
    huggingface_token: str

    # Webhook URL to call back Next.js app
    webhook_url: str = "http://localhost:3000/api/webhooks/diarization"
    webhook_secret: str = ""

    # Redis for job queue (optional - falls back to in-memory)
    redis_url: str | None = None

    # Processing settings
    device: str = "cpu"  # "cpu" or "cuda"
    whisper_model: str = "base"  # tiny, base, small, medium, large-v3

    # Diarization settings
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
