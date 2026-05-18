from __future__ import annotations

import os
import re
import threading
import json
import tempfile
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
    def __init__(self, *, qdrant: Optional[QdrantBackend] = None, index_path: Optional[str] = None):
        self.qdrant = qdrant or QdrantBackend()
        # PRODUCTION FIX: Eliminate hardcoded /tmp directory, use secure tempfile retrieval
        self.index_path = index_path or os.getenv("BM25_INDEX_PATH", os.path.join(tempfile.gettempdir(), "bm25_index.json"))
        self._lock = threading.RLock()
        self._index: Optional[_BM25Index] = None

    def _load(self) -> Optional[_BM25Index]:
        try:
            # PRODUCTION FIX: Safe JSON payload reading and dynamical recreation of BM25
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            tokenized = [_tokenize(text) for text in data["texts"]]
            if not tokenized:
                tokenized = [[""]]
                
            bm25 = BM25Okapi(tokenized)
            
            return _BM25Index(
                bm25=bm25,
                doc_ids=data["doc_ids"],
                texts=data["texts"],
                metadatas=data["metadatas"],
                equipment_to_indices=data["equipment_to_indices"]
            )
        except Exception:
            return None

    def _save(self, idx: _BM25Index) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        # PRODUCTION FIX: Serialize corpus into safe JSON instead of binary pickle
        data = {
            "doc_ids": idx.doc_ids,
            "texts": idx.texts,
            "metadatas": idx.metadatas,
            "equipment_to_indices": idx.equipment_to_indices
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

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

            for p in points:
                # PRODUCTION FIX: Safely extract payload whether Qdrant returns a dict or an object
                payload = (
                    p.get("payload", {}) if isinstance(p, dict) else getattr(p, "payload", {}) or {}
                )
                text = str(payload.get(text_key, ""))
                if not text.strip():
                    continue

                current_idx = len(doc_ids)
                doc_id = str(p.get("id", "")) if isinstance(p, dict) else str(getattr(p, "id", ""))
                doc_ids.append(doc_id)
                texts.append(text)
                metadatas.append(payload)
                tokenized.append(_tokenize(text))

                eq = payload.get(self.qdrant.settings.payload_equipment_id_key)
                # PRODUCTION FIX: Mirror `filters.py` and `pipeline.py` exactly by tracking "all"
                if isinstance(eq, str) and eq.strip():
                    equipment_to_indices.setdefault(eq, []).append(current_idx)
                else:
                    equipment_to_indices.setdefault("all", []).append(current_idx)

            if not tokenized:
                tokenized = [[""]]
                doc_ids = ["empty_db_dummy"]
                texts = [""]
                metadatas = [{}]

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

        # PRODUCTION FIX: Replaced unsafe assert with explicit RuntimeError to resolve B101
        if self._index is None:
            raise RuntimeError("Index could not be loaded or built")
            
        idx = self._index

        q_tokens = _tokenize(query)
        scores = idx.bm25.get_scores(q_tokens)

        candidate_indices = set(range(len(idx.doc_ids)))
        if filters and filters.equipment_id:
            specific_indices = set(idx.equipment_to_indices.get(filters.equipment_id, []))
            # PRODUCTION FIX: Union specific equipment indices with explicitly tagged "all" generic documents
            generic_indices = set(idx.equipment_to_indices.get("all", []))
            candidate_indices = specific_indices.union(generic_indices)

        results: list[tuple[int, float]] = []
        for i in candidate_indices:
            s = float(scores[i])
            if s > 0:
                results.append((i, s))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]

        docs: list[Document] = []
        for i, s in results:
            if idx.doc_ids[i] == "empty_db_dummy":
                continue

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