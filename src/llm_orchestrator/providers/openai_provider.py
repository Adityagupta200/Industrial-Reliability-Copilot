from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI

from .base import LLMProvider, LLMResult, LLMTransientError, LLMFatalError


def _requires_max_completion_tokens(model: str) -> bool:
    normalized = model.lower().strip()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
        base_url: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise LLMFatalError("OpenAI API key is not configured (LLM_OPENAI_API_KEY).")

        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._uses_max_completion_tokens = _requires_max_completion_tokens(model)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )

    async def invoke(self, prompt: str, json_mode: bool = False) -> LLMResult:
        try:
            payload = {
                "model": self._model_name,
                "messages": [await self._build_user_message(prompt, json_mode=json_mode)],
            }

            if self._uses_max_completion_tokens:
                payload["max_completion_tokens"] = self._max_tokens
            else:
                payload["max_tokens"] = self._max_tokens
                payload["temperature"] = self._temperature

            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            response = await self._client.chat.completions.create(**payload)
            message = response.choices[0].message if response.choices else None
            content = message.content if message and isinstance(message.content, str) else ""

            return LLMResult(content=content, model=self._model_name, provider=self.provider_name)

        except Exception as e:
            err_msg = str(e).lower()
            # PRODUCTION FIX: Do not retry on Auth/Billing failures
            if (
                "authentication" in err_msg
                or "401" in err_msg
                or "billing" in err_msg
                or "api_key" in err_msg
                or "model_not_found" in err_msg
                or "does not exist" in err_msg
                or "invalid model" in err_msg
                or "unsupported" in err_msg
                or "bad request" in err_msg
            ):
                raise LLMFatalError(f"OpenAI fatal configuration/API error: {e}") from e

            raise LLMTransientError(f"OpenAI invocation failed: {e}") from e

    async def _build_user_message(self, prompt: str, *, json_mode: bool) -> dict[str, str]:
        if json_mode:
            prompt += (
                "\n\n[SYSTEM]: You MUST output strictly valid JSON format. "
                "No markdown, no conversational text."
            )
        return {"role": "user", "content": prompt}
