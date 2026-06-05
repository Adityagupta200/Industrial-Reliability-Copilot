from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from llm_orchestrator.tracing import traceable

from ..schemas import HistoricalSearchRequest, HistoricalSearchResponse, RetrievedDoc
from ..schemas import EvidenceItem
from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..clients.rag_client import RAGClient
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


def _query_topics(text: str) -> list[str]:
    lowered = text.lower()
    topics: list[str] = []
    topic_map = {
        "bearing": ["bearing"],
        "cavitation": ["cavitation", "suction", "low flow"],
        "overheating": ["overheating", "temperature", "thermal"],
        "scheduled_maintenance": ["scheduled", "maintenance", "preventive"],
        "sensor": ["sensor", "pressure transducer", "calibration"],
        "lubrication": ["lubrication", "oil"],
    }
    for topic, needles in topic_map.items():
        if any(needle in lowered for needle in needles):
            topics.append(topic)
    return topics


def _historical_sql(req: HistoricalSearchRequest, table: str) -> tuple[str, dict[str, Any]]:
    clauses = ["timestamp >= NOW() - (:days_back * INTERVAL '1 day')"]
    params: dict[str, Any] = {"days_back": req.days_back, "limit": req.limit}

    if req.equipment_id:
        clauses.append("equipment_id = :equipment_id")
        params["equipment_id"] = req.equipment_id

    lowered = req.user_query.lower()
    if "low severity" in lowered or "low-severity" in lowered:
        clauses.append("severity::text = 'low'")
    elif "high or critical" in lowered or "high severity" in lowered or "critical" in lowered:
        clauses.append("severity::text IN ('high', 'critical')")

    topics = _query_topics(req.user_query)
    if topics and not any(word in lowered for word in ["compare", "across all", "all equipment"]):
        topic_clauses: list[str] = []
        for index, topic in enumerate(topics):
            key = f"topic_{index}"
            topic_clauses.append(f"failure_mode ILIKE :{key}")
            params[key] = f"%{topic}%"
        clauses.append("(" + " OR ".join(topic_clauses) + ")")

    order_by = "timestamp DESC"
    if any(term in lowered for term in ["longest", "resolution time", "resolution_time", "mttr"]):
        order_by = "resolution_time_hours DESC, timestamp DESC"

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            id::text AS id,
            timestamp,
            equipment_id,
            failure_mode,
            severity::text AS severity,
            actions_taken,
            outcome,
            resolution_time_hours
        FROM {table}
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT :limit
    """
    return sql, params


def _top_items(values: list[str], *, limit: int = 5) -> list[str]:
    return [name for name, _count in Counter(values).most_common(limit)]


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("failure_mode", "unknown")) for row in rows))


def _severity_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("severity", "unknown")) for row in rows))


def _avg_resolution(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["resolution_time_hours"])
        for row in rows
        if isinstance(row.get("resolution_time_hours"), (int, float))
    ]
    if not values:
        return None
    return round(statistics.fmean(values), 2)


def _human_list(items: list[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _summary_prefix(req: HistoricalSearchRequest) -> str:
    focus = req.equipment_id or "all equipment"
    lowered = req.user_query.lower()
    if "resolution" in lowered:
        return f"Incident resolution summary for {focus}"
    if "low" in lowered and "severity" in lowered:
        return f"Low-severity incident and mitigation summary for {focus}"
    if "scheduled" in lowered or "maintenance" in lowered:
        return f"Scheduled maintenance incident outcome summary for {focus}"
    return f"Incident history summary for {focus}"


def _build_historical_response(
    req: HistoricalSearchRequest,
    rows: list[dict[str, Any]],
) -> HistoricalSearchResponse:
    mode_counts = _mode_counts(rows)
    severity_counts = _severity_counts(rows)
    top_modes = _top_items([str(row.get("failure_mode", "unknown")) for row in rows])
    avg_resolution = _avg_resolution(rows)
    equipment_ids = _top_items([str(row.get("equipment_id", "unknown")) for row in rows], limit=4)
    actions = _top_items([str(row.get("actions_taken", "not recorded")) for row in rows], limit=3)
    outcomes = _top_items([str(row.get("outcome", "not recorded")) for row in rows], limit=3)

    topic_terms = _query_topics(req.user_query)
    topic_text = _human_list(topic_terms) if topic_terms else "reported reliability"
    prefix = _summary_prefix(req)

    if not rows:
        summary = (
            f"{prefix}: no matching incident records were found in the last "
            f"{req.days_back} days for {topic_text} patterns."
        )
        return HistoricalSearchResponse(
            summary=summary,
            key_stats={
                "incident_count": 0,
                "top_failure_modes": [],
                "severity_counts": {},
                "average_resolution_time_hours": None,
            },
            evidence=[EvidenceItem(claim=summary, source="SQL")],
        )

    avg_text = (
        f"Average resolution time is {avg_resolution} hours"
        if avg_resolution is not None
        else "Resolution time was not available"
    )
    summary = (
        f"{prefix}: found {len(rows)} incident records in the last {req.days_back} days. "
        f"Failure patterns include {topic_text}; top failure modes are "
        f"{_human_list(top_modes)}. {avg_text}. Common actions include "
        f"{_human_list(actions)} and outcomes include {_human_list(outcomes)}."
    )

    evidence = [
        EvidenceItem(
            claim=(
                f"{len(rows)} SQL incident rows matched the request for "
                f"{req.equipment_id or 'all equipment'}."
            ),
            source="SQL",
        ),
        EvidenceItem(
            claim=f"Top failure modes: {_human_list(top_modes)}.",
            source="SQL",
        ),
        EvidenceItem(
            claim=f"Severity distribution: {severity_counts}.",
            source="SQL",
        ),
        EvidenceItem(
            claim=f"Typical actions taken: {_human_list(actions)}.",
            source="SQL",
        ),
        EvidenceItem(
            claim=f"Observed outcomes: {_human_list(outcomes)}.",
            source="SQL",
        ),
    ]

    return HistoricalSearchResponse(
        summary=summary,
        key_stats={
            "incident_count": len(rows),
            "top_failure_modes": top_modes,
            "failure_mode_counts": mode_counts,
            "severity_counts": severity_counts,
            "equipment_ids": equipment_ids,
            "average_resolution_time_hours": avg_resolution,
        },
        evidence=evidence,
    )


@dataclass(frozen=True)
class HistoricalSearchChain:
    llm: LLMClient
    prompts: PromptLoader
    rag_client: RAGClient
    incident_repo: IncidentRepo
    incidents_table: str

    @traceable(run_type="chain", name="Historical_Chain")
    async def run(
        self, req: HistoricalSearchRequest
    ) -> tuple[HistoricalSearchResponse, str, str, str]:
        sql, params = _historical_sql(req, self.incidents_table)
        rows = await self.incident_repo.run_query(sql, params)
        response = _build_historical_response(req, rows)
        raw_context = json.dumps(
            {
                "sql_template": re.sub(r"\s+", " ", sql).strip(),
                "params": params,
                "row_count": len(rows),
                "sample_rows": rows[:5],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return response, "rules+sql", "historical-summary-v1", raw_context
