from __future__ import annotations

from sentence_transformers import SentenceTransformer
from rag_service.core.config import settings
from rag_service.embeddings.base import EmbeddingProvider


class BGEEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.bge_model_name)

    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]
