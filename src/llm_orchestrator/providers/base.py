from __future__ import annotations

from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


class LLMTransientError(LLMError):
    pass


class LLMFatalError(LLMError):
    pass


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider:
    provider_name: str

    async def invoke(self, prompt: str) -> LLMResult:  # pragma: no cover
        raise NotImplementedError
