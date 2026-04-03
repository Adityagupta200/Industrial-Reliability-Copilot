from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    # Provider routing - 2026 Production Standard
    primary_provider: Literal["openai", "ollama"] = "openai"
    fallback_provider: Literal["openai", "ollama"] = "ollama"

    # OpenAI (Production Intelligence)
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o"  # Upgraded for complex reasoning & groundedness
    openai_base_url: Optional[str] = None

    # Ollama (Local Fallback - Llama 3.1 8B fits well within a 10GB RAM allocation)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Generation controls
    temperature: float = 0.1  # Lowered for highly deterministic maintenance procedures
    max_tokens: int = 1000

    # Extended timeout to guarantee local CPU models have enough time
    request_timeout_s: float = 600.0

    # Reliability
    max_retries: int = 3
    retry_min_wait_s: float = 0.3
    retry_max_wait_s: float = 2.5

    # Observability - Mandatory for production
    enable_langchain_tracing: bool = True
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "industrial-reliability-copilot-prod"


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    anomaly_service_url: str = Field(default="http://localhost:8001")
    rag_service_url: str = Field(default="http://localhost:8002")

    # Endpoints
    anomaly_predict_anomaly_path: str = "/predict/anomaly"
    anomaly_predict_rul_path: str = "/predict/rul"

    rag_retrieve_hybrid_path: str = "/retrieve/hybrid"
    rag_retrieve_procedures_path: str = "/retrieve/procedures"
    rag_retrieve_semantic_path: str = "/retrieve/semantic"

    # DB for historical incidents
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
        pass

    return Settings(llm=llm, services=services)