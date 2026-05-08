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

        # PRODUCTION FIX: Removed problematic client_kwargs that caused Langchain 
        # to swallow the extended timeout on large context loads. Passed the raw 
        # float directly to the timeout parameter to guarantee the 300s window.
        self._client = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=timeout_s,  
        )

    async def invoke(self, prompt: str, json_mode: bool = False) -> LLMResult:
        try:
            client = self._client.bind(format="json") if json_mode else self._client
            msg = await client.ainvoke([HumanMessage(content=prompt)])
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return LLMResult(content=content, model=self._model_name, provider=self.provider_name)
        except Exception as e:
            # Provide more explicit logging in case of future errors
            error_details = str(e) or repr(e) or "ReadTimeout"
            raise LLMTransientError(f"Ollama invocation failed: {error_details}") from e