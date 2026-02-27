from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    data_dir: str = Field(default="data")
    raw_manuals_dir: str = Field(default="data/raw/manuals")
    raw_procedures_dir: str = Field(default="data/raw/procedures")
    raw_incidents_dir: str = Field(default="data/raw/incidents")
    processed_texts_dir: str = Field(default="data/processed/texts")
    processed_manifest_path: str = Field(default="data/processed/manifest/ingestion_manifest.json")

    # Postgres
    postgres_dsn: str = Field(
        default="postgresql+psycopg://irc:irc_password@localhost:5432/industrial_maintenance"
    )

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)

    qdrant_collection_docs: str = Field(default="maintenance_docs")
    qdrant_collection_procedures: str = Field(default="procedures")

    # Embeddings
    embedding_provider: str = Field(default="openai")  # "openai" or "bge"
    openai_api_key: str | None = Field(default=None)
    openai_embedding_model: str = Field(default="text-embedding-3-large")
    bge_model_name: str = Field(default="BAAI/bge-large-en-v1.5")

    # Chunking
    chunk_size_tokens: int = Field(default=700)  # within 512–768 target range
    chunk_overlap_tokens: int = Field(default=80)  # within 50–100 target range
    max_context_chars_per_chunk: int = Field(default=6000)

    # Ingestion batching
    embed_batch_size: int = Field(default=64)
    upsert_batch_size: int = Field(default=128)

    # Logging
    log_level: str = Field(default="INFO")


settings = Settings()
