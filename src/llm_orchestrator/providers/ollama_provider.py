from __future__ import annotations

import httpx

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
        self._base_url = base_url.rstrip("/")
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

    async def invoke(self, prompt: str, json_mode: bool = False) -> LLMResult:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(f"{self._base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            detail = str(exc) or repr(exc) or "ReadTimeout"
            raise LLMTransientError(f"Ollama invocation failed: {detail}") from exc

        content = body.get("response", "")
        return LLMResult(content=str(content), model=self._model_name, provider=self.provider_name)
