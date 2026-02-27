from __future__ import annotations

from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider
from rag_service.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_service.embeddings.bge_provider import BGEEmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "openai":
        return OpenAIEmbeddingProvider()
    if settings.embedding_provider.lower() == "bge":
        return BGEEmbeddingProvider()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
