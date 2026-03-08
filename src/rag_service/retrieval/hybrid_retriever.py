from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .keyword_retriever import BM25KeywordRetriever
from .semantic_retriever import SemanticRetriever
from .types import Document, RetrievalFilters


@dataclass(frozen=True)
class HybridSettings:
    semantic_k: int = 30
    keyword_k: int = 30
    rrf_k: int = 60
    out_k: int = 15


class HybridRetriever:
    def __init__(
        self,
        *,
        semantic: Optional[SemanticRetriever] = None,
        keyword: Optional[BM25KeywordRetriever] = None,
        settings: Optional[HybridSettings] = None,
    ):
        self.semantic = semantic or SemanticRetriever()
        self.keyword = keyword or BM25KeywordRetriever()
        self.settings = settings or HybridSettings()

        # Ensure BM25 is ready (can be moved to app startup)
        self.keyword.build_or_load(force_rebuild=False)

    @staticmethod
    def _rrf_score(rank: int, k: int) -> float:
        return 1.0 / (k + rank)

    def hybrid_search(
        self,
        query: str,
        *,
        filters: Optional[RetrievalFilters] = None,
        semantic_k: Optional[int] = None,
        keyword_k: Optional[int] = None,
        out_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> list[Document]:
        sem_k = semantic_k or self.settings.semantic_k
        key_k = keyword_k or self.settings.keyword_k
        out = out_k or self.settings.out_k
        rrf_const = rrf_k or self.settings.rrf_k

        sem_docs = self.semantic.semantic_search(query, k=sem_k, filters=filters)
        key_docs = self.keyword.keyword_search(query, k=key_k, filters=filters)

        fused: dict[str, Document] = {}
        fused_score: dict[str, float] = {}

        for rank, d in enumerate(sem_docs, start=1):
            fused.setdefault(d.id, d)
            fused_score[d.id] = fused_score.get(d.id, 0.0) + self._rrf_score(rank, rrf_const)

        for rank, d in enumerate(key_docs, start=1):
            fused.setdefault(d.id, d)
            fused_score[d.id] = fused_score.get(d.id, 0.0) + self._rrf_score(rank, rrf_const)

        merged = list(fused.values())
        merged.sort(key=lambda d: fused_score.get(d.id, 0.0), reverse=True)

        top = merged[:out]
        for d in top:
            d.source = "hybrid"
            d.score = float(fused_score.get(d.id, 0.0))
        return top
