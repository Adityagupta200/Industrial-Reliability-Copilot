from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .llm_config import LLMSettings
from .providers.base import LLMProvider, LLMResult, LLMTransientError, LLMFatalError
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider


class LLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

        if settings.enable_langchain_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            if settings.langsmith_api_key is not None:
                os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()

        # Empty dictionary for lazy loading
        self._providers = {}

    def _build_openai(self, s: LLMSettings) -> LLMProvider:
        api_key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
        return OpenAIProvider(
            api_key=api_key,
            model=s.openai_model,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            timeout_s=s.request_timeout_s,
            base_url=s.openai_base_url,
        )

    def _build_ollama(self, s: LLMSettings) -> LLMProvider:
        return OllamaProvider(
            base_url=s.ollama_base_url,
            model=s.ollama_model,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            timeout_s=s.request_timeout_s,
        )

    def _get_provider(self, name: str) -> LLMProvider:
        """Lazy-loads the requested LLM provider."""
        if name not in self._providers:
            if name == "openai":
                self._providers["openai"] = self._build_openai(self._settings)
            elif name == "ollama":
                self._providers["ollama"] = self._build_ollama(self._settings)
            else:
                raise LLMFatalError(f"Unknown provider '{name}'.")
        return self._providers[name]

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def _invoke_with_retry(self, provider: LLMProvider, prompt: str) -> LLMResult:
        return await provider.invoke(prompt)

    async def invoke(self, prompt: str, *, force_provider: Optional[str] = None) -> LLMResult:
        if force_provider:
            provider = self._get_provider(force_provider)
            return await self._invoke_with_retry(provider, prompt)

        primary = self._get_provider(self._settings.primary_provider)

        try:
            return await self._invoke_with_retry(primary, prompt)
        except LLMTransientError:
            # Fallback only on transient failures (timeouts/rate limits/etc.)
            fallback = self._get_provider(self._settings.fallback_provider)
            return await self._invoke_with_retry(fallback, prompt)
