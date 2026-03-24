from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ..schemas import RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGClient:
    base_url: str
    hybrid_path: str
    procedures_path: str  # Kept for compatibility, but we will route around it
    semantic_path: str
    timeout_s: float = 300.0  # INCREASED default from 60.0s to 300.0s to accommodate heavy ML cold starts

    def _extract_docs(self, response_data: Any) -> list[dict[str, Any]]:
        if isinstance(response_data, dict):
            docs = (
                response_data.get("documents")
                or response_data.get("docs")
                or response_data.get("results")
                or []
            )
            if not isinstance(docs, list):
                logger.warning(f"Expected a list of documents, got {type(docs)}.")
                return []
            return docs
        elif isinstance(response_data, list):
            return response_data

        logger.warning(f"Unexpected RAG response format: {type(response_data)}")
        return []

    def _get_timeout_config(self) -> httpx.Timeout:
        """Returns a robust timeout configuration allowing enough time for ML model inference."""
        return httpx.Timeout(
            self.timeout_s, 
            connect=10.0, 
            read=self.timeout_s, 
            write=self.timeout_s
        )

    async def retrieve_hybrid(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 8
    ) -> list[RetrievedDoc]:
        # FIX: The Hybrid endpoint expects 'out_k' instead of 'k' based on the RAG schema
        payload: dict[str, Any] = {"query": query, "out_k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._get_timeout_config()) as client:
            r = await client.post(self.hybrid_path, json=payload)
            r.raise_for_status()

            raw_docs = self._extract_docs(r.json())
            return [RetrievedDoc.model_validate(d) for d in raw_docs]

    async def retrieve_procedures(
        self, failure_mode: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        # FIX: Route the procedure search to the existing semantic endpoint
        # and map "failure_mode" to a standard context query
        payload: dict[str, Any] = {"query": f"procedure for {failure_mode}", "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._get_timeout_config()) as client:
            # FIX: Send to semantic_path instead of the non-existent procedures_path
            r = await client.post(self.semantic_path, json=payload)
            r.raise_for_status()

            raw_docs = self._extract_docs(r.json())
            return [RetrievedDoc.model_validate(d) for d in raw_docs]

    async def retrieve_semantic(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._get_timeout_config()) as client:
            r = await client.post(self.semantic_path, json=payload)
            r.raise_for_status()

            raw_docs = self._extract_docs(r.json())
            return [RetrievedDoc.model_validate(d) for d in raw_docs]