from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from rag_service.retrieval.keyword_retriever import BM25KeywordRetriever, _tokenize
from rag_service.retrieval.types import RetrievalFilters


@dataclass
class DummyQdrant:
    """
    Minimal QdrantBackend substitute for BM25KeywordRetriever unit tests.
    We bypass network by mocking scroll_all() results and by providing settings keys.
    """

    class _Settings:
        payload_text_key = "text"
        payload_equipment_id_key = "equipment_id"
        payload_severity_key = "severity"
        payload_date_key = "date"

    settings = _Settings()

    def __init__(self, points: list[dict[str, Any]]):
        self._points = points

    def scroll_all(self, *, qfilter=None, batch_size: int = 256):
        return self._points


def _make_points():
    # Mimic qdrant_backend.scroll_all() output shape:
    # [{"id": "...", "payload": {...}}, ...]
    return [
        {
            "id": "d1",
            "payload": {
                "text": "Pump P-23 maintenance procedure: replace bearing and lubricate.",
                "equipment_id": "P-23",
                "severity": "low",
            },
        },
        {
            "id": "d2",
            "payload": {
                "text": "Alarm: error code E404 indicates sensor timeout on controller.",
                "equipment_id": "CTRL-1",
                "severity": "high",
            },
        },
        {
            "id": "d3",
            "payload": {
                "text": "General overview of pumps and bearing wear symptoms.",
                "equipment_id": "P-99",
                "severity": "medium",
            },
        },
    ]


def test_tokenize_smoke():
    assert _tokenize("pump P-23 error code E404") == ["pump", "p-23", "error", "code", "e404"]


def test_bm25_build_and_keyword_search_returns_specific_matches(tmp_path):
    qdrant = DummyQdrant(points=_make_points())
    idx_path = tmp_path / "bm25.pkl"

    kr = BM25KeywordRetriever(qdrant=qdrant, index_path=str(idx_path))
    kr.build_or_load(force_rebuild=True)

    docs = kr.keyword_search("pump P-23", k=5)
    assert len(docs) > 0
    assert docs[0].id == "d1"
    assert "p-23" in docs[0].text.lower()

    docs2 = kr.keyword_search("error code E404", k=5)
    assert len(docs2) > 0
    assert docs2[0].id == "d2"
    assert "e404" in docs2[0].text.lower()


def test_bm25_equipment_id_filter_limits_candidates(tmp_path):
    qdrant = DummyQdrant(points=_make_points())
    idx_path = tmp_path / "bm25.pkl"

    kr = BM25KeywordRetriever(qdrant=qdrant, index_path=str(idx_path))
    kr.build_or_load(force_rebuild=True)

    # Query that could match multiple docs, but equipment filter should keep only P-23
    filters = RetrievalFilters(equipment_id="P-23")
    docs = kr.keyword_search("bearing", k=10, filters=filters)

    assert len(docs) > 0
    assert all(d.metadata.get("equipment_id") == "P-23" for d in docs)


def test_bm25_results_are_complementary_to_semantic_on_crafted_example(tmp_path):
    qdrant = DummyQdrant(points=_make_points())
    idx_path = tmp_path / "bm25.pkl"

    kr = BM25KeywordRetriever(qdrant=qdrant, index_path=str(idx_path))
    kr.build_or_load(force_rebuild=True)

    docs = kr.keyword_search("E404", k=3)
    assert docs, "Expected BM25 to return at least one result"
    assert docs[0].id == "d2", "Expected exact keyword match to rank highest"
