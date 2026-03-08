from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from rag_service.retrieval.cache import QueryEmbeddingCache
from rag_service.retrieval.semantic_retriever import SemanticRetriever
from rag_service.retrieval.types import RetrievalFilters


@dataclass
class DummyPoint:
    id: str
    payload: dict[str, Any]
    score: float


class DummyQdrant:
    def __init__(self, *, text_key: str = "text"):
        class _Settings:
            payload_text_key = text_key
            payload_equipment_id_key = "equipment_id"
            payload_severity_key = "severity"
            payload_date_key = "date"

        self.settings = _Settings()
        self.search_calls: list[dict[str, Any]] = []

    def dense_search(self, *, query_vector: list[float], limit: int, qfilter=None):
        self.search_calls.append({"query_vector": query_vector, "limit": limit, "qfilter": qfilter})
        return [
            DummyPoint(
                id="doc1",
                payload={self.settings.payload_text_key: "bearing failure symptoms ..."},
                score=0.91,
            ),
            DummyPoint(
                id="doc2",
                payload={self.settings.payload_text_key: "pump maintenance procedure ..."},
                score=0.87,
            ),
        ]


class DummyEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3]


def test_semantic_search_maps_payload_text_and_preserves_ranking(monkeypatch):
    cache = QueryEmbeddingCache(ttl_seconds=3600)
    retriever = SemanticRetriever()

    dummy_qdrant = DummyQdrant(text_key="content")  # pretend your payload key is "content"
    dummy_embedder = DummyEmbedder()

    retriever.qdrant = dummy_qdrant
    retriever.embedder = dummy_embedder
    retriever.cache = cache

    docs = retriever.semantic_search("bearing failure symptoms", k=2)

    assert [d.id for d in docs] == ["doc1", "doc2"]
    assert docs[0].score >= docs[1].score
    assert docs[0].text.startswith("bearing failure symptoms")
    assert docs[0].metadata["content"].startswith("bearing failure symptoms")


def test_semantic_search_caches_query_embedding(monkeypatch):
    cache = QueryEmbeddingCache(ttl_seconds=3600)
    retriever = SemanticRetriever()

    dummy_qdrant = DummyQdrant(text_key="text")
    dummy_embedder = DummyEmbedder()

    retriever.qdrant = dummy_qdrant
    retriever.embedder = dummy_embedder
    retriever.cache = cache

    _ = retriever.semantic_search("pump maintenance procedure", k=2)
    _ = retriever.semantic_search("pump maintenance procedure", k=2)

    assert dummy_embedder.calls == 1, "Expected only one embedding call due to cache hit"
    assert len(dummy_qdrant.search_calls) == 2, "Qdrant is still queried, only embedding is cached"


def test_semantic_search_applies_metadata_filters_smoke():
    cache = QueryEmbeddingCache(ttl_seconds=3600)
    retriever = SemanticRetriever()

    dummy_qdrant = DummyQdrant(text_key="text")
    dummy_embedder = DummyEmbedder()

    retriever.qdrant = dummy_qdrant
    retriever.embedder = dummy_embedder
    retriever.cache = cache

    filters = RetrievalFilters(equipment_id="P-23", severity="high")
    _ = retriever.semantic_search("bearing replacement", k=2, filters=filters)

    assert dummy_qdrant.search_calls, "Expected Qdrant search to be invoked"
    assert "qfilter" in dummy_qdrant.search_calls[-1]
