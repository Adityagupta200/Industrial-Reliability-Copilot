from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rag_service.core.config import settings


@dataclass(frozen=True)
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

    def ensure_collection(self, name: str, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if name in existing:
            # PRODUCTION FIX: Auto-heal dimension mismatches. 
            # If you switch from BGE (384) to OpenAI (1536), wipe and recreate the collection.
            try:
                col_info = self.client.get_collection(name)
                v_config = col_info.config.params.vectors
                current_size = v_config.size if hasattr(v_config, 'size') else getattr(v_config.get('default', {}), 'size', None)
                if current_size != vector_size:
                    self.client.delete_collection(name)
                else:
                    return
            except Exception:
                self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def upsert(self, collection: str, points: Iterable[VectorPoint]) -> None:
        qpoints = [qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
        self.client.upsert(collection_name=collection, points=qpoints, wait=True)

    def count(self, collection: str) -> int:
        return int(self.client.count(collection_name=collection, exact=True).count)