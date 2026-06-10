# src/rag_service/retrieval/qdrant_backend.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse


@dataclass(frozen=True)
class QdrantSettings:
    url: str
    api_key: Optional[str]
    collection: str
    timeout_seconds: float
    payload_text_key: str
    payload_equipment_id_key: str
    payload_severity_key: str
    payload_date_key: str
    vector_name: Optional[str]  # NEW: for named vectors

    @staticmethod
    def from_env() -> "QdrantSettings":
        return QdrantSettings(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection=os.getenv("QDRANT_COLLECTION", "maintenance_docs"),
            timeout_seconds=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "2.0")),
            payload_text_key=os.getenv("QDRANT_TEXT_PAYLOAD_KEY", "text"),
            payload_equipment_id_key=os.getenv("QDRANT_EQUIPMENT_ID_KEY", "equipment_id"),
            payload_severity_key=os.getenv("QDRANT_SEVERITY_KEY", "severity"),
            payload_date_key=os.getenv("QDRANT_DATE_KEY", "date"),
            vector_name=os.getenv("QDRANT_VECTOR_NAME") or None,
        )


class QdrantBackend:
    def __init__(self, settings: Optional[QdrantSettings] = None):
        self.settings = settings or QdrantSettings.from_env()
        self.client = QdrantClient(
            url=self.settings.url,
            api_key=self.settings.api_key,
            timeout=self.settings.timeout_seconds,
        )

    def dense_search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        qfilter: Optional[qmodels.Filter] = None,
    ) -> list[qmodels.ScoredPoint]:
        # Build query object; named vectors are required if your collection uses named vectors.
        query: Any = query_vector
        if self.settings.vector_name:
            query = qmodels.NamedVector(name=self.settings.vector_name, vector=query_vector)

        try:
            # Preferred API (newer clients): query_points
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    collection_name=self.settings.collection,
                    query=query,
                    query_filter=qfilter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                return list(res.points)

            # Backward compatible API (older clients): search
            return self.client.search(
                collection_name=self.settings.collection,
                query_vector=query,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as e:
            # FIX: Handle cold-start/empty environments where the collection hasn't been created yet.
            if getattr(e, "status_code", None) == 404 or "Not found" in str(e):
                return []
            raise

    def scroll_all(
        self,
        *,
        qfilter: Optional[qmodels.Filter] = None,
        batch_size: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Scrolls through all points in the collection (optionally filtered),
        returning a list of {"id": str(id), "payload": dict(payload)}.
        Qdrant scroll iterates page-by-page using an offset. [web:28][web:32]
        """
        out: list[dict[str, Any]] = []
        offset = None

        while True:
            try:
                points, offset = self.client.scroll(
                    collection_name=self.settings.collection,
                    scroll_filter=qfilter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except UnexpectedResponse as e:
                # Handle cold-start/empty environments where the collection hasn't been created yet.
                if getattr(e, "status_code", None) == 404 or "Not found" in str(e):
                    return []
                raise

            for p in points:
                out.append(
                    {
                        "id": str(p.id),
                        "payload": dict(p.payload or {}),
                    }
                )

            if offset is None:
                break

        return out
