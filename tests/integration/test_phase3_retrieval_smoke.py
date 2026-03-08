import os
import time
import statistics

import pytest
from qdrant_client import QdrantClient

from rag_service.retrieval import (
    SemanticRetriever,
    BM25KeywordRetriever,
    HybridRetriever,
    CrossEncoderReranker,
)
from rag_service.retrieval.types import Document


pytestmark = [pytest.mark.integration]


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        pytest.skip(f"Missing required env var: {name}")
    return str(v)


def _p95_ms(samples_ms: list[float]) -> float:
    xs = sorted(samples_ms)
    if not xs:
        return 0.0
    return xs[int(0.95 * (len(xs) - 1))]


def _contains_any_term(docs: list[Document], terms: list[str]) -> bool:
    terms_l = [t.lower() for t in terms]
    for d in docs:
        t = (d.text or "").lower()
        if any(term in t for term in terms_l):
            return True
    return False


def _require_qdrant_has_points(client: QdrantClient, collection: str) -> None:
    try:
        info = client.get_collection(collection_name=collection)
        assert info is not None
    except Exception as e:
        pytest.skip(f"Qdrant collection not accessible: {collection}. Error: {e}")

    try:
        cnt = client.count(collection_name=collection, exact=True).count
    except Exception as e:
        pytest.skip(f"Qdrant count() failed. Error: {e}")

    if cnt <= 0:
        pytest.skip(f"Qdrant collection '{collection}' is empty. Run ingestion (Phase 2) first.")


def test_phase3_retrieval_smoke_end_to_end() -> None:
    """
    Phase 3 acceptance checks (smoke-level):
    - Dense semantic retrieval returns non-empty for sample query
    - BM25 keyword retrieval returns non-empty for keyword queries
    - Hybrid retrieval returns non-empty
    - Optional reranker: reranks top docs and returns top-N
    - Optional strict latency gates (controlled by STRICT_LATENCY_ASSERTS)
    """
    qdrant_url = _env("QDRANT_URL", "http://localhost:6333")
    collection = _env("QDRANT_COLLECTION", "maintenance_docs")

    strict_latency = os.getenv("STRICT_LATENCY_ASSERTS", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    enable_reranker = os.getenv("ENABLE_RERANKER_TESTS", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    client = QdrantClient(url=qdrant_url)
    _require_qdrant_has_points(client, collection)

    # 3.1 Dense semantic retrieval
    sem = SemanticRetriever()
    q1 = "bearing failure symptoms"

    sem.semantic_search(q1, k=25)  # warm-up
    sem_docs = sem.semantic_search(q1, k=25)
    assert len(sem_docs) > 0, "semantic_search returned empty results"

    assert _contains_any_term(sem_docs[:10], ["bearing", "lubrication", "failure", "symptom"]), (
        "semantic_search top docs do not contain expected domain terms; "
        "verify ingestion payload text key and collection are correct"
    )

    hits_before = sem.cache.stats.hits
    sem.semantic_search(q1, k=25)
    hits_after = sem.cache.stats.hits
    assert hits_after > hits_before, "Expected query embedding cache hit on repeat query"

    sem_times: list[float] = []
    for _ in range(15):
        t0 = time.perf_counter()
        sem.semantic_search(q1, k=25)
        sem_times.append((time.perf_counter() - t0) * 1000.0)
    sem_p95 = _p95_ms(sem_times)

    # 3.2 Sparse keyword retrieval (BM25)
    kw = BM25KeywordRetriever()
    kw.build_or_load(force_rebuild=False)

    for q in ["pump P-23", "error code E404"]:
        docs = kw.keyword_search(q, k=10)
        assert len(docs) > 0, f"keyword_search returned empty results for: {q}"

    # 3.3 Hybrid retrieval (RRF)
    hybrid = HybridRetriever(semantic=sem, keyword=kw)
    q2 = "pump maintenance procedure"

    hybrid.hybrid_search(q2)  # warm-up
    hy_docs = hybrid.hybrid_search(q2)
    assert len(hy_docs) > 0, "hybrid_search returned empty results"

    assert _contains_any_term(hy_docs[:10], ["pump", "maintenance", "procedure", "replace"]), (
        "hybrid_search top docs do not contain expected terms; "
        "verify your processed docs are actually ingested"
    )

    hy_times: list[float] = []
    for _ in range(15):
        t0 = time.perf_counter()
        hybrid.hybrid_search(q2)
        hy_times.append((time.perf_counter() - t0) * 1000.0)
    hy_p95 = _p95_ms(hy_times)

    # 3.4 Reranker (optional)
    rr_p95: float | None = None
    final_docs: list[Document] = hy_docs

    reranker: CrossEncoderReranker | None = CrossEncoderReranker() if enable_reranker else None
    if reranker:
        reranker.rerank(q2, hy_docs)  # warm-up once

        rerank_times: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            final_docs = reranker.rerank(q2, hy_docs)
            rerank_times.append((time.perf_counter() - t0) * 1000.0)
        rr_p95 = _p95_ms(rerank_times)

        assert 3 <= len(final_docs) <= 5, "reranker should return top 3–5 docs by config"
        assert all(d.score is not None for d in final_docs), "reranked docs missing scores"

    # End-to-end latency measurement (hybrid + optional rerank) warm
    e2e_times: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        tmp = hybrid.hybrid_search(q2)
        if reranker:
            _ = reranker.rerank(q2, tmp)
        e2e_times.append((time.perf_counter() - t0) * 1000.0)
    e2e_p95 = _p95_ms(e2e_times)

    # Report (useful when passing / debugging)
    sem_p50 = statistics.median(sorted(sem_times))
    hy_p50 = statistics.median(sorted(hy_times))
    e2e_p50 = statistics.median(sorted(e2e_times))
    _ = (sem_p50, hy_p50, e2e_p50)  # keep locals referenced

    # Strict latency gates match Phase 3 acceptance criteria:
    # semantic <200ms, hybrid <300ms, pipeline <400ms, reranker overhead <100ms.
    if strict_latency:
        assert sem_p95 < 200.0, f"semantic p95 latency too high: {sem_p95:.1f}ms"
        assert hy_p95 < 300.0, f"hybrid p95 latency too high: {hy_p95:.1f}ms"
        assert e2e_p95 < 400.0, f"end-to-end p95 latency too high: {e2e_p95:.1f}ms"
        if enable_reranker and rr_p95 is not None:
            assert rr_p95 < 100.0, f"reranker p95 latency too high: {rr_p95:.1f}ms"
