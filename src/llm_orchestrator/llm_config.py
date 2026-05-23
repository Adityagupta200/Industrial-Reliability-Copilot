from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    primary_provider: Literal["openai", "ollama"] = "ollama"
    fallback_provider: Literal["openai", "ollama"] = "ollama"

    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o-mini"

    openai_judge_model: str = "gpt-4o-mini"

    openai_base_url: Optional[str] = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    temperature: float = 0.0
    max_tokens: int = 1000

    # PRODUCTION FIX: Increased from 5.0 to 45.0.
    request_timeout_s: float = 45.0

    max_retries: int = 2
    retry_min_wait_s: float = 0.2
    retry_max_wait_s: float = 1.0

    enable_langchain_tracing: bool = False
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "industrial-reliability-copilot-prod"

    @model_validator(mode="after")
    def route_to_k8s_dns(self) -> "LLMSettings":
        """PRODUCTION FIX: Dynamic K8s DNS Routing for Ollama"""
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            self.ollama_base_url = self.ollama_base_url.replace("localhost", "ollama")
        return self


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    anomaly_service_url: str = Field(default="http://localhost:8001")
    rag_service_url: str = Field(default="http://localhost:8002")

    anomaly_predict_anomaly_path: str = "/predict/anomaly"
    anomaly_predict_rul_path: str = "/predict/rul"
    rag_retrieve_hybrid_path: str = "/retrieve/hybrid"
    rag_retrieve_procedures_path: str = "/retrieve/procedures"
    rag_retrieve_semantic_path: str = "/retrieve/semantic"

    # PRODUCTION FIX: Corrected credentials and database name, matching RAG service
    incidents_db_dsn: str = Field(
        default="postgresql+asyncpg://irc:irc_password@localhost:5432/industrial_maintenance"
    )
    incidents_table: str = Field(default="incidents")

    @model_validator(mode="after")
    def route_to_k8s_dns(self) -> "ServiceSettings":
        """PRODUCTION FIX: Dynamic K8s DNS Routing for Inter-Service Comm & DB"""
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            self.anomaly_service_url = self.anomaly_service_url.replace(
                "localhost", "anomaly-service"
            )
            self.rag_service_url = self.rag_service_url.replace("localhost", "rag-service")
            self.incidents_db_dsn = self.incidents_db_dsn.replace("localhost", "postgres")
        return self


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    services: ServiceSettings


def load_settings() -> Settings:
    llm = LLMSettings()
    services = ServiceSettings()

    if llm.fallback_provider == "openai" and not llm.openai_api_key:
        raise ValueError(
            "CRITICAL: LLM_OPENAI_API_KEY environment variable is required for the fallback provider."
        )

    return Settings(llm=llm, services=services)
