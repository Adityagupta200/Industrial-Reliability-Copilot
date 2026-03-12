from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..schemas import RootCauseRequest, RootCauseResponse, RetrievedDoc
from ..utils.json_parse import parse_llm_json
from ..clients.anomaly_client import AnomalyClient
from ..clients.rag_client import RAGClient


def _format_docs(docs: list[RetrievedDoc]) -> str:
    parts = []
    for d in docs:
        title = d.title or "Untitled"
        src = d.source or "unknown"
        parts.append(f"[DOC:{d.id}] {title} (source={src}, score={d.score})\n{d.text}\n")
    return "\n---\n".join(parts)


@dataclass(frozen=True)
class RootCauseChain:
    llm: LLMClient
    prompts: PromptLoader
    anomaly_client: AnomalyClient
    rag_client: RAGClient

    async def run(self, req: RootCauseRequest) -> tuple[RootCauseResponse, str, str]:
        # Step 1: call anomaly service for model outputs
        anomaly_model = await self.anomaly_client.predict(req.sensor_data)

        # Step 2: hybrid retrieval
        retrieval_query = f"{req.user_query}\n\nAnomaly: {req.anomaly_description}"
        docs = await self.rag_client.retrieve_hybrid(
            retrieval_query, equipment_id=req.equipment_id, k=8
        )

        # Step 3: prompt formatting
        bundle = self.prompts.load("root_cause_analysis", req.prompt_version)
        prompt = bundle.template.format(
            anomaly_description=req.anomaly_description,
            sensor_data_json=json.dumps(req.sensor_data, ensure_ascii=False),
            anomaly_model_json=json.dumps(anomaly_model, ensure_ascii=False),
            retrieved_docs=_format_docs(docs),
        )

        # Step 4: LLM call
        result = await self.llm.invoke(prompt)

        # Step 5: Parse JSON and intercept Formatting Collapse gracefully
        try:
            parsed = parse_llm_json(result.content, RootCauseResponse)
            return parsed, result.provider, result.model
        except Exception as e:
            if type(e).__name__ == "LLMOutputParseError":
                # By raising a ValueError, main.py will safely catch it and return a 400 Bad Request
                raise ValueError(
                    "Blocked: Output is not adequately grounded in retrieved context."
                ) from e
            raise
