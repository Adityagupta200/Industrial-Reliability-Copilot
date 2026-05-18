from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# PRODUCTION FIX: Universal Path Resolution
# By dynamically anchoring to this file's location, paths resolve correctly
# regardless of the execution environment, OS, or containerization state.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Pathlib naturally handles Windows (\) vs Linux (/) separators
    data_dir: str = Field(default=str(PROJECT_ROOT / "data"))
    raw_manuals_dir: str = Field(default=str(PROJECT_ROOT / "data" / "raw" / "manuals"))
    raw_procedures_dir: str = Field(default=str(PROJECT_ROOT / "data" / "raw" / "procedures"))
    raw_incidents_dir: str = Field(default=str(PROJECT_ROOT / "data" / "raw" / "incidents"))
    processed_texts_dir: str = Field(default=str(PROJECT_ROOT / "data" / "processed" / "texts"))
    processed_manifest_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "processed" / "manifest" / "ingestion_manifest.json")
    )

    postgres_dsn: str = Field(
        default="postgresql+psycopg://irc:irc_password@localhost:5432/industrial_maintenance"
    )

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)

    qdrant_collection_docs: str = Field(default="maintenance_docs")
    qdrant_collection_procedures: str = Field(default="procedures")

    embedding_provider: str = Field(default="huggingface")
    huggingface_embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")

    openai_api_key: str | None = Field(default=None)
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    openai_timeout_s: float = Field(default=30.0)
    openai_max_retries: int = Field(default=3)

    chunk_size_tokens: int = Field(default=700)
    chunk_overlap_tokens: int = Field(default=80)
    max_context_chars_per_chunk: int = Field(default=6000)

    embed_batch_size: int = Field(default=64)
    upsert_batch_size: int = Field(default=128)

    log_level: str = Field(default="INFO")


settings = Settings()
