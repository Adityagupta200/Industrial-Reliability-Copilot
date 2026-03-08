from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ..schemas import RetrievedDoc


@dataclass(frozen=True)
class RAGClient:
    base_url: str
    hybrid_path: str
    procedures_path: str
    semantic_path: str
    timeout_s: float = 1.5

    async def retrieve_hybrid(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 8
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            r = await client.post(self.hybrid_path, json=payload)
            r.raise_for_status()
            docs = r.json().get("docs", r.json())
            return [RetrievedDoc.model_validate(d) for d in docs]

    async def retrieve_procedures(
        self, failure_mode: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"failure_mode": failure_mode, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            r = await client.post(self.procedures_path, json=payload)
            r.raise_for_status()
            docs = r.json().get("docs", r.json())
            return [RetrievedDoc.model_validate(d) for d in docs]

    async def retrieve_semantic(
        self, query: str, *, equipment_id: Optional[str] = None, k: int = 6
    ) -> list[RetrievedDoc]:
        payload: dict[str, Any] = {"query": query, "k": k, "filters": {}}
        if equipment_id:
            payload["filters"]["equipment_id"] = equipment_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            r = await client.post(self.semantic_path, json=payload)
            r.raise_for_status()
            docs = r.json().get("docs", r.json())
            return [RetrievedDoc.model_validate(d) for d in docs]
