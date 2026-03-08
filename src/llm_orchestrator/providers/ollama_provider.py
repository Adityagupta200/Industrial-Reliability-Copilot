from __future__ import annotations

from langchain_community.chat_models import ChatOllama
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
        self._client = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=timeout_s,
        )

    async def invoke(self, prompt: str) -> LLMResult:
        try:
            msg = await self._client.ainvoke([HumanMessage(content=prompt)])
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return LLMResult(content=content, model=self._model_name, provider=self.provider_name)
        except Exception as e:
            raise LLMTransientError(f"Ollama invocation failed: {e}") from e
