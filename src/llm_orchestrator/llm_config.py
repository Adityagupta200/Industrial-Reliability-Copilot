from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    # Provider routing - Path 1 (Strategic API Compromise)
    primary_provider: Literal["openai", "ollama"] = "openai"
    fallback_provider: Literal["openai", "ollama"] = "ollama"

    # OpenAI (Production Intelligence - Required for complex chains to hit <2s SLA)
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o" 
    openai_base_url: Optional[str] = None

    # Ollama (Local Fallback for simple/background tasks)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Generation controls
    temperature: float = 0.0  # Dropped to 0.0 for maximum determinism in RAG citations
    max_tokens: int = 1000

    # STRICT SLA: Max time allowed for LLM inference to meet 2s end-to-end target
    request_timeout_s: float = 2.5

    # Reliability
    max_retries: int = 2
    retry_min_wait_s: float = 0.2
    retry_max_wait_s: float = 1.0

    # Observability
    enable_langchain_tracing: bool = True
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "industrial-reliability-copilot-prod"

class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    anomaly_service_url: str = Field(default="http://localhost:8001")
    rag_service_url: str = Field(default="http://localhost:8002")
    
    anomaly_predict_anomaly_path: str = "/predict/anomaly"
    anomaly_predict_rul_path: str = "/predict/rul"
    rag_retrieve_hybrid_path: str = "/retrieve/hybrid"
    rag_retrieve_procedures_path: str = "/retrieve/procedures"
    rag_retrieve_semantic_path: str = "/retrieve/semantic"
    
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
    
    # Path 1 Enforcement: Fail fast if the API key is missing rather than falling back 
    # to the 8B model which will violate the latency and quality SLAs.
    if llm.primary_provider == "openai" and not llm.openai_api_key:
        raise ValueError("CRITICAL: LLM_OPENAI_API_KEY environment variable is required to meet production SLA targets.")
        
    return Settings(llm=llm, services=services)