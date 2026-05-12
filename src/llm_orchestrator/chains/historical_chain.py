from __future__ import annotations

import json
from dataclasses import dataclass
from langsmith import traceable

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


# PRODUCTION FIX: Injected the Data Dictionary and an off-topic fallback rule.
# This strictly grounds the LLM to reality and prevents column hallucination.
_TEXT2SQL_PROMPT = """You are a senior data engineer.
Write a single Postgres SELECT query to answer the question using only the allowed table.

Table: {table}
Allowed Columns:
- id (UUID)
- timestamp (DateTime)
- equipment_id (String)
- sensor_data (JSONB)
- failure_mode (String)
- severity (Enum: low, medium, high, critical)
- actions_taken (Text)
- outcome (Text)
- resolution_time_hours (Float)

Rules:
- ONLY SELECT
- Only use the exact table and columns listed above.
- Prefer explicit column names (avoid SELECT *)
- Include a WHERE clause when equipment_id is provided
- Always include ORDER BY timestamp DESC when timestamp exists
- Always include LIMIT {limit}
- If the question is completely unrelated to the schema (e.g., a joke or casual chat), return exactly: SELECT id FROM {table} LIMIT 0

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

    @traceable(run_type="chain", name="Historical_Chain") 
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