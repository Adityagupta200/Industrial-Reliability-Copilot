from __future__ import annotations

import os

from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider
from langchain_huggingface import HuggingFaceEmbeddings

class BGEEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        device = getattr(settings, "bge_device", None) or os.getenv("EMBEDDING_DEVICE", "cpu")
        
        # PRODUCTION FIX: Implemented requested langchain-huggingface provider
        self.model = HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True, 
                'batch_size': settings.embed_batch_size
            }
        )
        self._dim = None

    def dim(self) -> int:
        # Dynamically infer dimension on first call to support swapping between base/large models
        if self._dim is None:
            self._dim = len(self.model.embed_query("dimension_check"))
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # LangChain handles the batch processing and list conversions natively
        return self.model.embed_documents(texts)