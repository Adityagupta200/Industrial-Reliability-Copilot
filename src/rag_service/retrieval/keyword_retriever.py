from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from .qdrant_backend import QdrantBackend
from .types import Document, RetrievalFilters


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    doc_ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    equipment_to_indices: dict[str, list[int]]


class BM25KeywordRetriever:
    """
    Builds an in-memory BM25 index over Qdrant payload text.

    Note: for production, you typically build this once in a startup job and
    persist to disk/object storage. This implementation supports optional local persistence.
    """

    def __init__(self, *, qdrant: Optional[QdrantBackend] = None, index_path: Optional[str] = None):
        self.qdrant = qdrant or QdrantBackend()
        self.index_path = index_path or os.getenv(
            "BM25_INDEX_PATH", "data/processed/bm25_index.pkl"
        )
        self._lock = threading.RLock()
        self._index: Optional[_BM25Index] = None

    def _load(self) -> Optional[_BM25Index]:
        try:
            import pickle

            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save(self, idx: _BM25Index) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        import pickle

        with open(self.index_path, "wb") as f:
            pickle.dump(idx, f)

    def build_or_load(self, *, force_rebuild: bool = False) -> None:
        with self._lock:
            if self._index is not None and not force_rebuild:
                return

            if not force_rebuild:
                loaded = self._load()
                if loaded is not None:
                    self._index = loaded
                    return

            points = self.qdrant.scroll_all(qfilter=None, batch_size=256)
            text_key = self.qdrant.settings.payload_text_key

            doc_ids: list[str] = []
            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            tokenized: list[list[str]] = []
            equipment_to_indices: dict[str, list[int]] = {}

            for i, p in enumerate(points):
                payload = dict(p["payload"])
                text = str(payload.get(text_key, ""))
                if not text.strip():
                    continue

                doc_ids.append(str(p["id"]))
                texts.append(text)
                metadatas.append(payload)
                tokenized.append(_tokenize(text))

                eq = payload.get(self.qdrant.settings.payload_equipment_id_key)
                if isinstance(eq, str) and eq:
                    equipment_to_indices.setdefault(eq, []).append(i)

            bm25 = BM25Okapi(tokenized)
            idx = _BM25Index(
                bm25=bm25,
                doc_ids=doc_ids,
                texts=texts,
                metadatas=metadatas,
                equipment_to_indices=equipment_to_indices,
            )
            self._index = idx
            self._save(idx)

    def keyword_search(
        self,
        query: str,
        k: int = 25,
        *,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[Document]:
        if self._index is None:
            self.build_or_load(force_rebuild=False)

        assert self._index is not None
        idx = self._index

        q_tokens = _tokenize(query)
        scores = idx.bm25.get_scores(q_tokens)

        candidate_indices = range(len(idx.doc_ids))
        if filters and filters.equipment_id:
            candidate_indices = idx.equipment_to_indices.get(filters.equipment_id, [])

        results: list[tuple[int, float]] = []
        for i in candidate_indices:
            s = float(scores[i])
            if s > 0:
                results.append((i, s))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]

        docs: list[Document] = []
        for i, s in results:
            meta = idx.metadatas[i]
            if filters and filters.severity:
                if meta.get(self.qdrant.settings.payload_severity_key) != filters.severity:
                    continue
            docs.append(
                Document(
                    id=idx.doc_ids[i],
                    text=idx.texts[i],
                    metadata=meta,
                    score=s,
                    source="keyword",
                )
            )
        return docs
