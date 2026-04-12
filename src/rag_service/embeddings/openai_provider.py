from __future__ import annotations

import httpx
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            
        # PRODUCTION FIX: Use configurable timeouts rather than hardcoded 1.0s.
        # This prevents dropped connections while maintaining a safety ceiling.
        http_client = httpx.Client(timeout=httpx.Timeout(settings.openai_timeout_s))
        
        # PRODUCTION FIX: Enable native OpenAI SDK retries
        self.client = OpenAI(
            api_key=settings.openai_api_key, 
            http_client=http_client,
            max_retries=settings.openai_max_retries
        )
        self.model = settings.openai_embedding_model
        self._dim: int | None = None

    def dim(self) -> int:
        if self._dim is None:
            v = self.embed_texts(["dimension probe"])
            self._dim = len(v[0])
        return self._dim

    # PRODUCTION FIX: Decorate with exponential backoff to handle rate limits (429) 
    # and transient API/network connection errors gracefully during batch ingestion.
    @retry(
        retry=retry_if_exception_type((
            httpx.RequestError, 
            openai.APIConnectionError, 
            openai.RateLimitError, 
            openai.APITimeoutError
        )),
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]