from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from .cache import QueryEmbeddingCache
from .filters import build_qdrant_filter
from .qdrant_backend import QdrantBackend, QdrantSettings
from .types import Document, RetrievalFilters


CPU_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GPU_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str
    device: str
    normalize: bool
    query_prefix: str

    @staticmethod
    def from_env() -> "EmbeddingSettings":
        device = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()
        default_model = CPU_DEFAULT_MODEL if device == "cpu" else GPU_DEFAULT_MODEL
        model_name = os.getenv("EMBEDDING_MODEL_NAME", default_model).strip()
        normalize = os.getenv("EMBEDDING_NORMALIZE", "true").strip().lower() == "true"

        query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "").strip()
        if not query_prefix and model_name.lower().startswith("baai/bge"):
            query_prefix = BGE_QUERY_PREFIX

        return EmbeddingSettings(
            model_name=model_name,
            device=device,
            normalize=normalize,
            query_prefix=query_prefix,
        )


class _SentenceTransformerEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        from rag_service.core.model_cache import get_sentence_transformer

        self.model = get_sentence_transformer(settings.model_name, device=settings.device)
        self.normalize = settings.normalize
        self.query_prefix = settings.query_prefix

    def embed_query(self, text: str) -> list[float]:
        query_text = f"{self.query_prefix}{text}" if self.query_prefix else text
        vec = self.model.encode(
            [query_text],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0]
        return [float(x) for x in vec]


class SemanticRetriever:
    def __init__(
        self,
        *,
        qdrant: Optional[QdrantBackend] = None,
        qdrant_settings: Optional[QdrantSettings] = None,
        embedding_settings: Optional[EmbeddingSettings] = None,
        cache: Optional[QueryEmbeddingCache] = None,
    ):
        self.qdrant = qdrant or QdrantBackend(qdrant_settings)
        self.embedder = _SentenceTransformerEmbedder(
            embedding_settings or EmbeddingSettings.from_env()
        )
        self.cache = cache or QueryEmbeddingCache(
            ttl_seconds=int(os.getenv("QUERY_EMBED_CACHE_TTL_SECONDS", "3600"))
        )

    def semantic_search(
        self,
        query: str,
        k: int = 25,
        *,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[Document]:
        t0 = time.perf_counter()

        vec = self.cache.get(query)
        if vec is None:
            vec = self.embedder.embed_query(query)
            self.cache.set(query, vec)

        qfilter = build_qdrant_filter(
            filters,
            equipment_id_key=self.qdrant.settings.payload_equipment_id_key,
            severity_key=self.qdrant.settings.payload_severity_key,
            date_key=self.qdrant.settings.payload_date_key,
        )

        points = self.qdrant.dense_search(query_vector=vec, limit=k, qfilter=qfilter)

        docs: list[Document] = []
        text_key = self.qdrant.settings.payload_text_key
        for p in points:
            payload = dict(p.payload or {})
            docs.append(
                Document(
                    id=str(p.id),
                    text=str(payload.get(text_key, "")),
                    metadata=payload,
                    score=float(p.score),
                    source="semantic",
                )
            )

        _ = time.perf_counter() - t0
        return docs
