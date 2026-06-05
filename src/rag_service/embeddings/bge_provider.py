from __future__ import annotations

import logging
import os

import torch
from sentence_transformers import SentenceTransformer

from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider

# Silence benign HuggingFace architecture warnings for clean service logs.
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class BGEEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        device = getattr(settings, "bge_device", None) or os.getenv("EMBEDDING_DEVICE", "cpu")

        if device == "cpu":
            torch.set_num_threads(2)

        self.model = SentenceTransformer(
            settings.huggingface_embedding_model,
            device=device,
        )
        self._dim: int | None = None

    def dim(self) -> int:
        if self._dim is None:
            dim = self.model.get_sentence_embedding_dimension()
            if dim is None:
                dim = len(self.embed_texts(["dimension_check"])[0])
            self._dim = int(dim)
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(float).tolist()
