from __future__ import annotations

import httpx
from openai import OpenAI
from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            
        # PRODUCTION FIX: Inject strict HTTP timeout to prevent synchronous thread blocking
        http_client = httpx.Client(timeout=1.0)
        self.client = OpenAI(api_key=settings.openai_api_key, http_client=http_client)
        self.model = settings.openai_embedding_model
        self._dim: int | None = None

    def dim(self) -> int:
        if self._dim is None:
            v = self.embed_texts(["dimension probe"])
            self._dim = len(v[0])
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]