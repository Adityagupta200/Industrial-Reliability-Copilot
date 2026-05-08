from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from rag_service.core.model_cache import get_cross_encoder
from .types import Document


@dataclass(frozen=True)
class RerankerSettings:
    model_name: str
    max_rerank: int
    top_n: int
    batch_size: int
    device: Optional[str]

    @staticmethod
    def from_env() -> "RerankerSettings":
        return RerankerSettings(
            model_name=os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
            # PRODUCTION FIX: Hard ceiling aligned with HybridSettings out_k
            max_rerank=int(os.getenv("RERANK_MAX_DOCS", "8")),
            top_n=int(os.getenv("RERANK_TOP_N", "3")),
            # PRODUCTION FIX: Process all candidates in exactly one batch
            batch_size=int(os.getenv("RERANK_BATCH_SIZE", "8")),
            device=os.getenv("RERANKER_DEVICE") or None,
        )


class CrossEncoderReranker:
    def __init__(self, *, settings: Optional[RerankerSettings] = None):
        self.settings = settings or RerankerSettings.from_env()
        self._model = None
        
        # PRODUCTION FIX: Prevent CPU Thread Thrashing
        # Clamps PyTorch threads to prevent catastrophic context-switching in containerized 
        # or restricted memory environments.
        if self.settings.device is None or self.settings.device == "cpu":
            import torch
            torch.set_num_threads(2)

    def _get_model(self):
        if self._model is None:
            self._model = get_cross_encoder(
                self.settings.model_name,
                device=self.settings.device,
            )
        return self._model

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        t0 = time.perf_counter()

        if not docs:
            return []

        candidates = docs[: max(0, self.settings.max_rerank)]
        if not candidates:
            return []

        pairs = [(query, (d.text or "")) for d in candidates]

        model = self._get_model()
        # PRODUCTION FIX: Disable progress bar I/O overhead
        scores = model.predict(pairs, batch_size=self.settings.batch_size, show_progress_bar=False)

        rescored: list[Document] = []
        for d, s in zip(candidates, scores):
            rescored.append(
                Document(
                    id=d.id,
                    text=d.text,
                    metadata=d.metadata,
                    score=float(s),
                    source="rerank",
                )
            )

        rescored.sort(key=lambda x: x.score, reverse=True)
        out = rescored[: max(0, min(self.settings.top_n, len(rescored)))]

        _ = time.perf_counter() - t0
        return out