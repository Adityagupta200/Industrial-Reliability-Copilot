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
        timeout_config = httpx.Timeout(timeout_s, connect=10.0, read=timeout_s, write=timeout_s)

        self._client = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=timeout_s,  
            client_kwargs={"timeout": timeout_config}, 
            # PRODUCTION FIX: Removed globally hardcoded format="json". 
            # Forcing JSON globally breaks the LLM-as-a-judge which needs to output textual <SCORE> tags.
        )

    # PRODUCTION FIX: Added json_mode parameter to match the caller signature in llm_client.py
    async def invoke(self, prompt: str, json_mode: bool = False) -> LLMResult:
        try:
            # Dynamically apply the JSON constraint ONLY when the specific chain requests it
            client = self._client.bind(format="json") if json_mode else self._client
            msg = await client.ainvoke([HumanMessage(content=prompt)])
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return LLMResult(content=content, model=self._model_name, provider=self.provider_name)
        except Exception as e:
            raise LLMTransientError(f"Ollama invocation failed: {e}") from e