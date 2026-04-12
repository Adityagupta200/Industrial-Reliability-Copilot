from __future__ import annotations
import json
import re
import asyncio
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..schemas import RootCauseRequest, RootCauseResponse, RetrievedDoc
from ..utils.json_parse import parse_llm_json
from ..clients.anomaly_client import AnomalyClient
from ..clients.rag_client import RAGClient
from ..guardrails.output_filters import OutputGuardrails

def _format_docs(docs: list[RetrievedDoc]) -> tuple[str, dict[str, str]]:
    if not docs:
        return "NO DOCUMENTATION FOUND.", {}

    parts = []
    mapping = {}
    for i, d in enumerate(docs, start=1):
        title = d.title or "Untitled"
        real_source = d.metadata.get("file_name", d.metadata.get("source", d.id))
        doc_tag = f"DOC_{i}"
        mapping[doc_tag] = real_source
        parts.append(f"[{doc_tag}] {title} (score={d.score})\n{d.text}\n")
        
    return "\n---\n".join(parts), mapping

def _extract_missing_entities(user_query: str, current_eq: str | None, current_anom: str) -> tuple[str | None, str]:
    eq_id = current_eq
    anom = current_anom

    if not eq_id:
        match = re.search(r'([A-Z]-\d+)', user_query, re.IGNORECASE)
        if match:
            base_id = match.group(1).upper()
            q_lower = user_query.lower()
            if "pump" in q_lower: eq_id = f"pump_{base_id}"
            elif "motor" in q_lower: eq_id = f"motor_{base_id}"
            elif "compressor" in q_lower: eq_id = f"compressor_{base_id}"
            elif "turbofan" in q_lower: eq_id = f"turbofan_{base_id}"

    if not anom or anom.strip() == "":
        anom_keywords = []
        q_lower = user_query.lower()
        if "vibration" in q_lower: anom_keywords.append("high vibration")
        if "temperature" in q_lower or "temp" in q_lower or "overheat" in q_lower: anom_keywords.append("overheating")
        if "pressure" in q_lower: anom_keywords.append("pressure anomaly")
        
        anom = " and ".join(anom_keywords) if anom_keywords else "unspecified anomaly"

    return eq_id, anom

@dataclass(frozen=True)
class RootCauseChain:
    llm: LLMClient
    prompts: PromptLoader
    anomaly_client: AnomalyClient
    rag_client: RAGClient

    async def run(self, req: RootCauseRequest) -> tuple[RootCauseResponse, str, str, str]:
        req.equipment_id, req.anomaly_description = _extract_missing_entities(
            req.user_query, req.equipment_id, req.anomaly_description
        )
        
        retrieval_query = f"{req.user_query}\n\nAnomaly: {req.anomaly_description}"
        
        anomaly_task = self.anomaly_client.predict(req.sensor_data)
        rag_task = self.rag_client.retrieve_hybrid(
            retrieval_query, equipment_id=req.equipment_id, k=8
        )
        
        anomaly_model, docs = await asyncio.gather(anomaly_task, rag_task)
        
        docs_text, doc_mapping = _format_docs(docs)
        valid_ids_list = list(doc_mapping.keys())
        valid_doc_ids_str = ", ".join(valid_ids_list) if valid_ids_list else "NO_DOCS"
        
        bundle = self.prompts.load("root_cause_analysis", req.prompt_version)
        prompt = bundle.template.format(
            anomaly_description=req.anomaly_description,
            sensor_data_json=json.dumps(req.sensor_data, ensure_ascii=False),
            anomaly_model_json=json.dumps(anomaly_model, ensure_ascii=False),
            retrieved_docs=docs_text,
            valid_doc_ids=valid_doc_ids_str  
        )
        
        result = await self.llm.invoke(prompt, json_mode=True)

        # PRODUCTION FIX: Ensure the local judge has full visibility into the Anomaly Model
        judge_input = (
            f"User Query: {req.user_query}\n"
            f"Anomaly Description: {req.anomaly_description}\n"
            f"Sensor Data: {json.dumps(req.sensor_data)}\n"
            f"Anomaly Model Output: {json.dumps(anomaly_model)}"
        )

        is_valid, msg = await OutputGuardrails.validate_output(
            llm_client=self.llm,
            context=docs_text,
            answer=result.content,
            initial_input=judge_input
        )
        if not is_valid:
            raise ValueError(msg)
        
        try:
            parsed = parse_llm_json(result.content, RootCauseResponse)
            
            for hyp in parsed.hypotheses:
                if not doc_mapping:
                    hyp.source = "NONE"
                    continue
                    
                text_to_search = f"{hyp.source} {hyp.evidence}"
                match = re.search(r'DOC[_\W]*(\d+)', text_to_search, re.IGNORECASE)
                
                if match:
                    normalized_tag = f"DOC_{match.group(1)}"
                    if normalized_tag in doc_mapping:
                        hyp.source = doc_mapping[normalized_tag]
                    else:
                        raise ValueError(f"Hallucinated citation detected: {normalized_tag}")
                else:
                    raise ValueError("Missing required citation tag in hypothesis.")
                    
            return parsed, result.provider, result.model, docs_text
            
        except Exception as e:
            if type(e).__name__ == "LLMOutputParseError":
                raise ValueError("Blocked: Output JSON was malformed.") from e
            raise ValueError(f"Blocked: Output failed grounding/citation constraints: {e}") from e