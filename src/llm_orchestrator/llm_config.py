from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    # Provider routing
    primary_provider: Literal["openai", "ollama"] = "openai"
    fallback_provider: Literal["openai", "ollama"] = "ollama"

    # OpenAI
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None  # for proxies / Azure-compatible gateways

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Generation controls
    temperature: float = 0.2
    max_tokens: int = 800
    request_timeout_s: float = 20.0

    # Reliability
    max_retries: int = 3
    retry_min_wait_s: float = 0.3
    retry_max_wait_s: float = 2.5

    # Observability
    enable_langchain_tracing: bool = False
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "industrial-reliability-copilot"


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    anomaly_service_url: str = Field(default="http://localhost:8001")
    rag_service_url: str = Field(default="http://localhost:8002")

    # Endpoints (make these match your existing services)
    anomaly_predict_anomaly_path: str = "/predict/anomaly"
    anomaly_predict_rul_path: str = "/predict/rul"

    rag_retrieve_hybrid_path: str = "/retrieve/hybrid"
    rag_retrieve_procedures_path: str = "/retrieve/procedures"
    rag_retrieve_semantic_path: str = "/retrieve/semantic"

    # DB for historical incidents (direct DB access for Step 4.3 historical chain)
    incidents_db_dsn: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/industrialmaintenance"
    )
    incidents_table: str = Field(default="incidents")


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    services: ServiceSettings


def load_settings() -> Settings:
    llm = LLMSettings()
    services = ServiceSettings()

    if llm.primary_provider == "openai" and llm.openai_api_key is None:
        # Allow running without OpenAI by setting primary_provider=ollama.
        # In production, you usually want OpenAI configured and Ollama as fallback.
        pass

    return Settings(llm=llm, services=services)
