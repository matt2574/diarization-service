"""Configuration settings for the diarization service."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # pyannoteAI API key (from dashboard.pyannote.ai)
    pyannote_api_key: str

    # Webhook URL to call back Next.js app
    webhook_url: str = "http://localhost:3000/api/webhooks/diarization"
    webhook_secret: str = ""

    # pyannoteAI model selection
    # Options: "precision-2" (best), "precision-1", "community-1" (cheapest)
    pyannote_model: str = "precision-2"

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
