from __future__ import annotations

import os
import uuid
import pytest

from rag_service.vectorstore.qdrant_store import QdrantStore, VectorPoint

# PRODUCTION FIX: Gate integration tests behind environment variable
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run integration tests"
)


def test_qdrant_upsert_and_count() -> None:
    store = QdrantStore()
    store.ensure_collection("test_collection_smoke", vector_size=4)

    store.upsert(
        "test_collection_smoke",
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"k": "v"},
            ),
        ],
    )

    assert store.count("test_collection_smoke") >= 1
