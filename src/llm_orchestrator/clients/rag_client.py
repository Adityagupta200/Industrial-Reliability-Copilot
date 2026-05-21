from __future__ import annotations
import logging
import os
from typing import Any, Optional
import httpx
from langsmith import traceable

from ..schemas import RetrievedDoc

logger = logging.getLogger(__name__)


class RAGClient:
    def __init__(
        self,
        base_url: str,
        hybrid_path: str,
        procedures_path: str,
        semantic_path: str,
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url
        self.hybrid_path = hybrid_path
        self.procedures_path = procedures_path
        self.semantic_path = semantic_path
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout_config = httpx.Timeout(60.0, connect=5.0)
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_config)
        return self._client

    def _extract_docs(self, response_data: Any) -> list[dict[str, Any]]:
        docs = []
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
        elif isinstance(response_data, list):
            docs = response_data
        else:
            logger.warning(f"Unexpected RAG response format: {type(response_data)}")
            return []

        bad_tags = {"hybrid", "semantic", "keyword", "unknown"}
        for d in docs:
            meta = d.get("metadata", {})
            candidate_source = meta.get("file_name") or meta.get("source") or d.get("source")

            if not candidate_source or str(candidate_source).lower() in bad_tags:
                title = meta.get("title", d.get("title", ""))
                if title and title != "Untitled":
                    candidate_source = f"{title.replace(' ', '_').lower()}.pdf"
                else:
                    # PRODUCTION FIX: Safely cast to string before lower() to prevent AttributeError
                    eq_id = meta.get("equipment_id", "equipment")
                    if not eq_id:
                        eq_id = "equipment"
                    candidate_source = f"{str(eq_id).lower()}_reference_manual.pdf"

            d["source"] = os.path.basename(str(candidate_source))
            meta["file_name"] = d["source"]

        return docs

    @traceable(run_type="retriever", name="Qdrant_Hybrid_Search")
    async def retrieve_hybrid(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 8
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "out_k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        client = self._get_client()
        try:
            r = await client.post(self.hybrid_path, json=payload)
            r.raise_for_status()
            raw_docs = self._extract_docs(r.json())
            return [RetrievedDoc.model_validate(d) for d in raw_docs]
        except Exception as e:
            logger.error(f"RAG Service Hybrid Retrieval failed: {type(e).__name__} - {e}")
            return []

    @traceable(run_type="retriever", name="Qdrant_Procedure_Search")
    async def retrieve_procedures(
        self,
        failure_mode: str,
        *,
        equipment_id: Optional[str] = None,
        k: int = 6,
        query: Optional[str] = None,
    ) -> list[RetrievedDoc]:
        retrieval_query = query or f"procedure for {failure_mode}"
        payload: dict[str, Any] = {"query": retrieval_query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        client = self._get_client()
        try:
            r = await client.post(self.procedures_path, json=payload)
            r.raise_for_status()
            raw_docs = self._extract_docs(r.json())
            return [RetrievedDoc.model_validate(d) for d in raw_docs]
        except Exception as e:
            logger.error(f"RAG Service Procedure Retrieval failed: {type(e).__name__} - {e}")
            return []

    @traceable(run_type="retriever", name="Qdrant_Semantic_Search")
    async def retrieve_semantic(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        client = self._get_client()
        try:
            r = await client.post(self.semantic_path, json=payload)
            r.raise_for_status()
            raw_docs = self._extract_docs(r.json())
            return [RetrievedDoc.model_validate(d) for d in raw_docs]
        except Exception as e:
            logger.error(f"RAG Service Semantic Retrieval failed: {type(e).__name__} - {e}")
            return []
