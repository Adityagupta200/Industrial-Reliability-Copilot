from __future__ import annotations

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from .base import LLMProvider, LLMResult, LLMTransientError


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        self._model_name = model

        # FIX: Create an explicit httpx Timeout object.
        # langchain_ollama's default timeout parameter sometimes fails to pass through
        # to the underlying async httpx client, causing ReadTimeout crashes.
        timeout_config = httpx.Timeout(timeout_s, connect=10.0, read=timeout_s, write=timeout_s)

        self._client = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=timeout_s,  # Keep for sync fallback compatibility
            client_kwargs={"timeout": timeout_config},  # Force async client timeout
            format="json",  # MLE FIX: Force Constrained Decoding to strictly output valid JSON
        )

    async def invoke(self, prompt: str) -> LLMResult:
        try:
            msg = await self._client.ainvoke([HumanMessage(content=prompt)])
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return LLMResult(content=content, model=self._model_name, provider=self.provider_name)
        except Exception as e:
            raise LLMTransientError(f"Ollama invocation failed: {e}") from e
