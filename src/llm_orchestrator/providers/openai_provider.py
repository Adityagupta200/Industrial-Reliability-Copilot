from __future__ import annotations

from typing import Optional
import httpx

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import LLMProvider, LLMResult, LLMTransientError, LLMFatalError


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
        self._client = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
            base_url=base_url,
        )

    async def invoke(self, prompt: str, json_mode: bool = False) -> LLMResult:
        try:
            # PRODUCTION FIX: Use modern LangChain .bind() to avoid kwargs validation crashes
            client = self._client
            if json_mode:
                client = self._client.bind(response_format={"type": "json_object"})
                prompt += "\n\n[SYSTEM]: You MUST output strictly valid JSON format. No markdown, no conversational text."

            msg = await client.ainvoke([HumanMessage(content=prompt)])
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            
            return LLMResult(content=content, model=self._model_name, provider=self.provider_name)
            
        except Exception as e:  
            err_msg = str(e).lower()
            # PRODUCTION FIX: Do not retry on Auth/Billing failures
            if "authentication" in err_msg or "401" in err_msg or "billing" in err_msg or "api_key" in err_msg:
                raise LLMFatalError(f"API Authentication/Billing Failed: {e}") from e
                
            raise LLMTransientError(f"OpenAI invocation failed: {e}") from e