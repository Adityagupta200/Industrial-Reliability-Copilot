from __future__ import annotations
import os
import time
from typing import Optional

from .cache import QueryEmbeddingCache
from .filters import build_qdrant_filter
from .qdrant_backend import QdrantBackend, QdrantSettings
from .types import Document, RetrievalFilters

# PRODUCTION FIX: Import the centralized provider instead of local hardcodes
from rag_service.embeddings import get_embedding_provider


class SemanticRetriever:
    def __init__(
        self,
        *,
        qdrant: Optional[QdrantBackend] = None,
        qdrant_settings: Optional[QdrantSettings] = None,
        cache: Optional[QueryEmbeddingCache] = None,
    ):
        self.qdrant = qdrant or QdrantBackend(qdrant_settings)
        # PRODUCTION FIX: Use the configured embedding provider (OpenAI)
        # to guarantee the query vector dimensions match the ingestion vectors.
        self.embedder = get_embedding_provider()
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
            # Use the interface standard embed_texts method
            vec = self.embedder.embed_texts([query])[0]
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
            # Safe parsing regardless of Qdrant client object type returns
            payload = (
                p.get("payload", {}) if isinstance(p, dict) else getattr(p, "payload", {}) or {}
            )
            docs.append(
                Document(
                    id=str(p.get("id", "")) if isinstance(p, dict) else str(getattr(p, "id", "")),
                    text=str(payload.get(text_key, "")),
                    metadata=payload,
                    score=(
                        float(p.get("score", 0.0))
                        if isinstance(p, dict)
                        else float(getattr(p, "score", 0.0))
                    ),
                    source="semantic",
                )
            )

        _ = time.perf_counter() - t0
        return docs
