from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import List, Tuple

from rag_service.retrieval.reranker import CrossEncoderReranker, RerankerSettings
from rag_service.retrieval.types import Document


@dataclass
class _PredictCall:
    pairs: List[Tuple[str, str]]
    batch_size: int


class DummyCrossEncoder:
    """
    Stand-in for sentence_transformers.CrossEncoder used by CrossEncoderReranker.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.calls: List[_PredictCall] = []

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16, **kwargs):
        self.calls.append(_PredictCall(pairs=pairs, batch_size=batch_size))
        return list(range(len(pairs)))


def _install_dummy_sentence_transformers(monkeypatch) -> None:
    dummy_mod = types.ModuleType("sentence_transformers")
    dummy_mod.CrossEncoder = DummyCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_mod)
    
    # PRODUCTION FIX: Flush the global model cache to prevent test state leakage
    monkeypatch.setattr("rag_service.core.model_cache._cross_models", {})


def _mk_docs(n: int) -> list[Document]:
    return [
        Document(
            id=f"d{i}",
            text=f"doc {i} text",
            metadata={"i": i},
            score=0.0,
            source="hybrid",
        )
        for i in range(n)
    ]


def test_reranker_batches_predict_and_returns_top_n(monkeypatch):
    _install_dummy_sentence_transformers(monkeypatch)

    settings = RerankerSettings(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        max_rerank=10, 
        top_n=5, 
        batch_size=16, 
        device="cpu"
    )
    rr = CrossEncoderReranker(settings=settings)

    docs = _mk_docs(30)
    out = rr.rerank("pump maintenance procedure", docs)

    assert len(out) == 5, "Expected top_n=5 docs returned"
    assert all(d.source == "rerank" for d in out), "Expected docs to be labeled source='rerank'"

    assert out[0].id == "d9"
    assert out[1].id == "d8"
    assert out[2].id == "d7"
    assert out[3].id == "d6"
    assert out[4].id == "d5"

    model = rr._get_model()
    assert isinstance(model, DummyCrossEncoder)
    assert len(model.calls) == 1, "Expected one batched predict() call"
    assert len(model.calls[0].pairs) == 10, "Expected rerank over max_rerank=10 pairs"
    assert model.calls[0].batch_size == 16, "Expected batch_size passed through"


def test_reranker_handles_less_than_max_rerank(monkeypatch):
    _install_dummy_sentence_transformers(monkeypatch)

    settings = RerankerSettings(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        max_rerank=10,
        top_n=5,
        batch_size=8,
        device="cpu"
    )
    rr = CrossEncoderReranker(settings=settings)

    docs = _mk_docs(3) 
    out = rr.rerank("bearing failure symptoms", docs)

    assert len(out) == 3, "Should return all docs if fewer than top_n"
    assert [d.id for d in out] == ["d2", "d1", "d0"], "Expected descending reranker score order"

    model = rr._get_model()
    assert len(model.calls) == 1
    assert len(model.calls[0].pairs) == 3
    assert model.calls[0].batch_size == 8


def test_reranker_scores_are_written_to_documents(monkeypatch):
    _install_dummy_sentence_transformers(monkeypatch)

    settings = RerankerSettings(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        max_rerank=10,
        top_n=5,
        batch_size=4,
        device="cpu"
    )
    rr = CrossEncoderReranker(settings=settings)

    docs = _mk_docs(10)
    out = rr.rerank("error code E404", docs)

    assert out[0].id == "d9"
    assert out[0].score == 9.0
    assert out[-1].score == 5.0