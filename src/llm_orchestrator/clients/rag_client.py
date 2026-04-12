from __future__ import annotations
import logging
from typing import Any, Optional
import httpx

from ..schemas import RetrievedDoc

logger = logging.getLogger(__name__)

class RAGClient:
    def __init__(
        self,
        base_url: str,
        hybrid_path: str,
        procedures_path: str,
        semantic_path: str,
        timeout_s: float = 5.0  
    ):
        self.base_url = base_url
        self.hybrid_path = hybrid_path
        self.procedures_path = procedures_path
        self.semantic_path = semantic_path
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout_config = httpx.Timeout(self.timeout_s, connect=1.0)
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_config)
        return self._client

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

    async def retrieve_hybrid(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 8
    ) -> list[RetrievedDoc]:
        # PRODUCTION FIX: Standardized parameter naming to 'k' to match API contract
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id
            
        client = self._get_client()
        r = await client.post(self.hybrid_path, json=payload)
        r.raise_for_status()
        raw_docs = self._extract_docs(r.json())
        return [RetrievedDoc.model_validate(d) for d in raw_docs]

    async def retrieve_procedures(
        self, failure_mode: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": f"procedure for {failure_mode}", "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id
            
        client = self._get_client()
        r = await client.post(self.procedures_path, json=payload)
        r.raise_for_status()
        raw_docs = self._extract_docs(r.json())
        return [RetrievedDoc.model_validate(d) for d in raw_docs]

    async def retrieve_semantic(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id
            
        client = self._get_client()
        r = await client.post(self.semantic_path, json=payload)
        r.raise_for_status()
        raw_docs = self._extract_docs(r.json())
        return [RetrievedDoc.model_validate(d) for d in raw_docs]