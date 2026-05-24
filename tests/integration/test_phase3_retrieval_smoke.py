import os
import time
import pytest
import requests
from qdrant_client import QdrantClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run integration tests"
    ),
]


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
    Phase 3 -> Phase 9 Upgraded Acceptance Checks:
    Tests the RAG retrieval microservice across the network boundary
    instead of relying on local PyTorch bindings.
    """
    qdrant_url = _env("QDRANT_URL", "http://localhost:6333")
    rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")
    collection = _env("QDRANT_COLLECTION", "maintenance_docs")

    strict_latency = os.getenv("STRICT_LATENCY_ASSERTS", "0").lower() in ("1", "true", "yes")

    client = QdrantClient(url=qdrant_url)
    _require_qdrant_has_points(client, collection)

    # 3.1 Dense semantic retrieval via API
    q1 = "bearing failure symptoms"
    payload_sem = {"query": q1, "k": 25, "filters": {}}

    # Warm-up and fetch
    res_sem = requests.post(f"{rag_url}/retrieve/semantic", json=payload_sem, timeout=60.0)
    assert res_sem.status_code == 200, f"Semantic API failed: {res_sem.text}"

    sem_data = res_sem.json()
    sem_docs = sem_data.get("documents", [])
    assert len(sem_docs) > 0, "semantic_search API returned empty results"

    terms_sem = ["bearing", "lubrication", "failure", "symptom"]
    found_sem = any(any(t in d.get("text", "").lower() for t in terms_sem) for d in sem_docs[:10])
    assert found_sem, "semantic_search top docs do not contain expected domain terms"

    sem_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        requests.post(f"{rag_url}/retrieve/semantic", json=payload_sem, timeout=10.0)
        sem_times.append((time.perf_counter() - t0) * 1000.0)
    sem_p95 = _p95_ms(sem_times)

    # 3.2 Sparse keyword retrieval (BM25) via API
    for q in ["pump P-23", "error code E404"]:
        res_kw = requests.post(
            f"{rag_url}/retrieve/keyword", json={"query": q, "k": 10}, timeout=10.0
        )
        assert res_kw.status_code == 200, f"Keyword API failed: {res_kw.text}"
        docs = res_kw.json().get("documents", [])
        assert len(docs) > 0, f"keyword_search API returned empty results for: {q}"

    # 3.3 Hybrid retrieval (RRF) via API
    q2 = "pump maintenance procedure"
    payload_hy = {"query": q2, "k": 10, "filters": {}}

    res_hy = requests.post(f"{rag_url}/retrieve/hybrid", json=payload_hy, timeout=60.0)
    assert res_hy.status_code == 200, f"Hybrid API failed: {res_hy.text}"

    hy_docs = res_hy.json().get("documents", [])
    assert len(hy_docs) > 0, "hybrid_search API returned empty results"

    terms_hy = ["pump", "maintenance", "procedure", "replace"]
    found_hy = any(any(t in d.get("text", "").lower() for t in terms_hy) for d in hy_docs[:10])
    assert found_hy, "hybrid_search top docs do not contain expected terms"

    hy_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        requests.post(f"{rag_url}/retrieve/hybrid", json=payload_hy, timeout=10.0)
        hy_times.append((time.perf_counter() - t0) * 1000.0)
    hy_p95 = _p95_ms(hy_times)

    # Strict latency gates (loosened slightly for network overhead vs local execution)
    if strict_latency:
        assert sem_p95 < 400.0, f"semantic API p95 latency too high: {sem_p95:.1f}ms"
        assert hy_p95 < 500.0, f"hybrid API p95 latency too high: {hy_p95:.1f}ms"
