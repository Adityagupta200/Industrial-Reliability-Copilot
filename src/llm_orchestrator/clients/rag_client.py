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
    procedures_path: str
    semantic_path: str
    timeout_s: float = 1.2  # PRODUCTION FIX: 1.2s SLA budget

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
        """Returns a strict timeout configuration for production SLA."""
        return httpx.Timeout(
            self.timeout_s,
            connect=0.2, # Extremely fast connection drop to prevent deadlocks
            read=self.timeout_s,
            write=self.timeout_s
        )

    async def retrieve_hybrid(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 8
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "out_k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id
            
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._get_timeout_config()) as client:
            try:
                r = await client.post(self.hybrid_path, json=payload)
                r.raise_for_status()
                raw_docs = self._extract_docs(r.json())
                return [RetrievedDoc.model_validate(d) for d in raw_docs]
            except httpx.TimeoutException:
                logger.error("RAG retrieval timed out. SLA budget exceeded. Returning empty context.")
                return []
            except httpx.RequestError as e:
                logger.error(f"RAG retrieval request failed: {e}. Returning empty context.")
                return []

    async def retrieve_procedures(
        self, failure_mode: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": f"procedure for {failure_mode}", "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id
            
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._get_timeout_config()) as client:
            try:
                r = await client.post(self.semantic_path, json=payload)
                r.raise_for_status()
                raw_docs = self._extract_docs(r.json())
                return [RetrievedDoc.model_validate(d) for d in raw_docs]
            except httpx.TimeoutException:
                logger.error("Procedure retrieval timed out. Returning empty context.")
                return []
            except httpx.RequestError as e:
                logger.error(f"Procedure retrieval failed: {e}. Returning empty context.")
                return []

    async def retrieve_semantic(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id
            
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._get_timeout_config()) as client:
            try:
                r = await client.post(self.semantic_path, json=payload)
                r.raise_for_status()
                raw_docs = self._extract_docs(r.json())
                return [RetrievedDoc.model_validate(d) for d in raw_docs]
            except httpx.TimeoutException:
                logger.error("Semantic retrieval timed out. Returning empty context.")
                return []
            except httpx.RequestError as e:
                logger.error(f"Semantic retrieval failed: {e}. Returning empty context.")
                return []