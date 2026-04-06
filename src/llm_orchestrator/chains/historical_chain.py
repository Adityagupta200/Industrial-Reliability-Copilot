from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..schemas import HistoricalSearchRequest, HistoricalSearchResponse, RetrievedDoc
from ..utils.json_parse import parse_llm_json
from ..clients.rag_client import RAGClient
from ..db.safe_sql import SQLPolicy, validate_readonly_sql
from ..db.incident_repo import IncidentRepo


def _format_docs(docs: list[RetrievedDoc]) -> str:
    parts = []
    for d in docs:
        parts.append(f"[DOC:{d.id}] {d.title or 'Untitled'}\n{d.text}\n")
    return "\n---\n".join(parts)


_TEXT2SQL_PROMPT = """You are a senior data engineer.
Write a single Postgres SELECT query to answer the question using only the allowed table.

Rules:
- ONLY SELECT
- Only table: {table}
- Prefer explicit column names (avoid SELECT *)
- Include a WHERE clause when equipment_id is provided
- Always include ORDER BY timestamp DESC when timestamp exists
- Always include LIMIT {limit}
Return ONLY the SQL string and nothing else.

Question: {question}
equipment_id: {equipment_id}
days_back: {days_back}
"""


@dataclass(frozen=True)
class HistoricalSearchChain:
    llm: LLMClient
    prompts: PromptLoader
    rag_client: RAGClient
    incident_repo: IncidentRepo
    incidents_table: str

    # PRODUCTION FIX: Signature updated
    async def run(self, req: HistoricalSearchRequest) -> tuple[HistoricalSearchResponse, str, str, str]:
        sql_prompt = _TEXT2SQL_PROMPT.format(
            table=self.incidents_table,
            limit=req.limit,
            question=req.user_query,
            equipment_id=req.equipment_id or "",
            days_back=req.days_back,
        )
        sql_result = await self.llm.invoke(sql_prompt)

        policy = SQLPolicy(allowed_tables={self.incidents_table}, max_limit=max(req.limit, 50))
        safe_sql = validate_readonly_sql(sql_result.content.strip(), policy)

        rows = await self.incident_repo.run_query(safe_sql)

        docs = await self.rag_client.retrieve_semantic(
            req.user_query, equipment_id=req.equipment_id, k=6
        )

        formatted_context = _format_docs(docs)
        
        bundle = self.prompts.load("historical_search", req.prompt_version)
        prompt = bundle.template.format(
            user_query=req.user_query,
            sql_rows_json=json.dumps(rows, ensure_ascii=False),
            retrieved_docs=formatted_context,
        )

        final = await self.llm.invoke(prompt)
        parsed = parse_llm_json(final.content, HistoricalSearchResponse)
        return parsed, final.provider, final.model, formatted_context