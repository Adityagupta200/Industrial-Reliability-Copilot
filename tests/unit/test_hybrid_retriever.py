from __future__ import annotations

from typing import Optional


from rag_service.retrieval.hybrid_retriever import HybridRetriever, HybridSettings
from rag_service.retrieval.types import Document, RetrievalFilters


class DummySemantic:
    def __init__(self):
        self.calls = []

    def semantic_search(self, query: str, k: int, *, filters: Optional[RetrievalFilters] = None):
        self.calls.append({"query": query, "k": k, "filters": filters})
        # Include overlap with keyword results: "B" appears in both.
        return [
            Document(
                id="A",
                text="semantic A",
                metadata={"equipment_id": "P-23"},
                score=0.9,
                source="semantic",
            ),
            Document(
                id="B",
                text="semantic B",
                metadata={"equipment_id": "P-23"},
                score=0.8,
                source="semantic",
            ),
            Document(
                id="C",
                text="semantic C",
                metadata={"equipment_id": "P-99"},
                score=0.7,
                source="semantic",
            ),
        ]


class DummyKeyword:
    def __init__(self):
        self.calls = []

    def build_or_load(self, *, force_rebuild: bool = False) -> None:
        return

    def keyword_search(self, query: str, k: int, *, filters: Optional[RetrievalFilters] = None):
        self.calls.append({"query": query, "k": k, "filters": filters})
        return [
            Document(
                id="B",
                text="keyword B",
                metadata={"equipment_id": "P-23"},
                score=12.0,
                source="keyword",
            ),
            Document(
                id="D",
                text="keyword D",
                metadata={"equipment_id": "P-23"},
                score=10.0,
                source="keyword",
            ),
            Document(
                id="E",
                text="keyword E",
                metadata={"equipment_id": "P-23"},
                score=9.0,
                source="keyword",
            ),
        ]


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def test_hybrid_rrf_deduplicates_and_orders_by_rrf():
    sem = DummySemantic()
    kw = DummyKeyword()
    settings = HybridSettings(semantic_k=30, keyword_k=30, rrf_k=60, out_k=15)

    hr = HybridRetriever(semantic=sem, keyword=kw, settings=settings)
    out = hr.hybrid_search("pump maintenance")

    ids = [d.id for d in out]
    assert len(ids) == len(set(ids)), "Expected deduplication by doc id"

    # Compute expected fused scores for the top few:
    # From semantic ranks: A=1, B=2, C=3
    # From keyword ranks:  B=1, D=2, E=3
    score_A = _rrf(1, 60)
    score_B = _rrf(2, 60) + _rrf(1, 60)
    score_C = _rrf(3, 60)
    score_D = _rrf(2, 60)
    score_E = _rrf(3, 60)

    expected_sorted = sorted(
        [("A", score_A), ("B", score_B), ("C", score_C), ("D", score_D), ("E", score_E)],
        key=lambda x: x[1],
        reverse=True,
    )
    # Only compare relative ordering among these ids (the hybrid should include them all).
    expected_order = [x[0] for x in expected_sorted]
    actual_order = [i for i in ids if i in expected_order]

    assert (
        actual_order == expected_order
    ), f"Expected RRF order {expected_order}, got {actual_order}"
    assert all(
        d.source == "hybrid" for d in out[:5]
    ), "Hybrid retriever should label docs as source='hybrid'"


def test_hybrid_passes_filters_to_both_retrievers():
    sem = DummySemantic()
    kw = DummyKeyword()
    hr = HybridRetriever(semantic=sem, keyword=kw)

    f = RetrievalFilters(equipment_id="P-23", severity="high")
    _ = hr.hybrid_search("bearing replacement", filters=f)

    assert sem.calls, "Semantic retriever should be called"
    assert kw.calls, "Keyword retriever should be called"
    assert sem.calls[-1]["filters"] == f, "Expected filters passed to semantic_search"
    assert kw.calls[-1]["filters"] == f, "Expected filters passed to keyword_search"


def test_hybrid_respects_out_k_limit():
    sem = DummySemantic()
    kw = DummyKeyword()
    settings = HybridSettings(out_k=2)
    hr = HybridRetriever(semantic=sem, keyword=kw, settings=settings)

    out = hr.hybrid_search("any query")
    assert len(out) == 2
