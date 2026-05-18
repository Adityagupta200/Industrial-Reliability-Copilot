from __future__ import annotations
import logging

from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# PRODUCTION FIX: Global Singleton Cache
# Prevents reloading 1.5GB model weights into RAM on every FastAPI request.
_PROVIDER_INSTANCE: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """
    Factory to initialize the appropriate embedding model provider based on environment settings.
    Utilizes a Singleton pattern to ensure models are only loaded into memory once.
    """
    global _PROVIDER_INSTANCE
    if _PROVIDER_INSTANCE is not None:
        return _PROVIDER_INSTANCE

    provider_name = settings.embedding_provider.lower().strip()

    if provider_name in ("huggingface", "bge"):
        logger.info(
            f"Initializing local HuggingFace provider: {settings.huggingface_embedding_model}"
        )
        from rag_service.embeddings.bge_provider import BGEEmbeddingProvider

        _PROVIDER_INSTANCE = BGEEmbeddingProvider()

    elif provider_name == "openai":
        logger.info(f"Initializing OpenAI provider: {settings.openai_embedding_model}")
        from rag_service.embeddings.openai_provider import OpenAIEmbeddingProvider

        _PROVIDER_INSTANCE = OpenAIEmbeddingProvider()

    else:
        # This will now safely catch and format any future misconfigurations
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")

    return _PROVIDER_INSTANCE
