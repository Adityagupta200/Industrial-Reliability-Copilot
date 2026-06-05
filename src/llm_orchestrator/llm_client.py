from __future__ import annotations

import os
from typing import Optional
import logging

from llm_orchestrator.tracing import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .llm_config import LLMSettings
from .providers.base import LLMProvider, LLMResult, LLMTransientError, LLMFatalError
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


def _trace_llm_inputs(inputs: dict) -> dict:
    prompt = inputs.get("prompt", "")
    prompt_text = str(prompt)
    return {
        "prompt": prompt_text,
        "prompt_chars": len(prompt_text),
        "force_provider": inputs.get("force_provider"),
        "json_mode": inputs.get("json_mode", False),
        "is_judge": inputs.get("is_judge", False),
    }


def _trace_llm_outputs(outputs: LLMResult) -> dict:
    content = outputs.content if isinstance(outputs.content, str) else str(outputs.content)
    return {
        "provider": outputs.provider,
        "model": outputs.model,
        "content_preview": content[:1200],
        "content_chars": len(content),
    }


class LLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

        if settings.enable_langchain_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            if settings.langsmith_api_key is not None:
                os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()

        self._providers = {}

    def _build_openai(self, s: LLMSettings, is_judge: bool = False) -> LLMProvider:
        api_key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
        model_name = s.openai_judge_model if is_judge else s.openai_model

        return OpenAIProvider(
            api_key=api_key,
            model=model_name,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            timeout_s=s.request_timeout_s,
            base_url=s.openai_base_url,
        )

    def _build_ollama(self, s: LLMSettings, is_judge: bool = False) -> LLMProvider:
        return OllamaProvider(
            base_url=s.ollama_base_url,
            model=s.ollama_model,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            timeout_s=120.0,
        )

    def _get_provider(self, name: str, is_judge: bool = False) -> LLMProvider:
        cache_key = f"{name}_judge" if is_judge else name
        if cache_key not in self._providers:
            if name == "openai":
                self._providers[cache_key] = self._build_openai(self._settings, is_judge)
            elif name == "ollama":
                self._providers[cache_key] = self._build_ollama(self._settings, is_judge)
            else:
                raise LLMFatalError(f"Unknown provider '{name}'.")
        return self._providers[cache_key]

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        reraise=True,
    )
    async def _invoke_with_retry(
        self, provider: LLMProvider, prompt: str, json_mode: bool = False
    ) -> LLMResult:
        return await provider.invoke(prompt, json_mode=json_mode)

    @traceable(
        run_type="llm",
        name="Prompt_Model_Call",
        process_inputs=_trace_llm_inputs,
        process_outputs=_trace_llm_outputs,
    )
    async def invoke(
        self,
        prompt: str,
        *,
        force_provider: Optional[str] = None,
        json_mode: bool = False,
        is_judge: bool = False,
    ) -> LLMResult:
        if force_provider:
            provider = self._get_provider(force_provider, is_judge)
            return await self._invoke_with_retry(provider, prompt, json_mode=json_mode)

        primary = self._get_provider(self._settings.primary_provider, is_judge)

        try:
            return await self._invoke_with_retry(primary, prompt, json_mode=json_mode)
        except LLMTransientError as e:
            # PRODUCTION FIX: Implement Graceful Degradation
            logger.warning(
                f"Primary provider '{self._settings.primary_provider}' failed: {e}. "
                f"Initiating failover to fallback provider '{self._settings.fallback_provider}'."
            )

            if self._settings.fallback_provider == self._settings.primary_provider:
                raise LLMFatalError(
                    f"Primary provider '{self._settings.primary_provider}' failed and the "
                    "fallback provider is identical. Configure a distinct fallback provider "
                    "or fix the primary provider/model/API key."
                ) from e

            try:
                fallback = self._get_provider(self._settings.fallback_provider, is_judge)
                return await self._invoke_with_retry(fallback, prompt, json_mode=json_mode)
            except Exception as fallback_err:
                logger.error(f"Fallback provider also failed: {fallback_err}")
                raise LLMFatalError(
                    "Complete LLM orchestration failure: Both primary and fallback APIs are down."
                ) from fallback_err
