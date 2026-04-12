from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    # PRODUCTION FIX: Enforce local Ollama orchestration by default
    primary_provider: Literal["openai", "ollama"] = "ollama"
    fallback_provider: Literal["openai", "ollama"] = "openai"

    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-5.4-mini" 
    
    openai_judge_model: str = "gpt-5.4-nano"
    
    openai_base_url: Optional[str] = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    temperature: float = 0.0  
    max_tokens: int = 1000
    
    request_timeout_s: float = 5.0

    max_retries: int = 2
    retry_min_wait_s: float = 0.2
    retry_max_wait_s: float = 1.0

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
    
    # Validation remains for fallback SLA targets
    if llm.fallback_provider == "openai" and not llm.openai_api_key:
        raise ValueError("CRITICAL: LLM_OPENAI_API_KEY environment variable is required for the fallback provider.")
        
    return Settings(llm=llm, services=services)