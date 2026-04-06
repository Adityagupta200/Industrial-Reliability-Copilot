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

@dataclass(frozen=True)
class RootCauseChain:
    llm: LLMClient
    prompts: PromptLoader
    anomaly_client: AnomalyClient
    rag_client: RAGClient

    async def run(self, req: RootCauseRequest) -> tuple[RootCauseResponse, str, str, str]:
        retrieval_query = f"{req.user_query}\n\nAnomaly: {req.anomaly_description}"
        
        anomaly_task = self.anomaly_client.predict(req.sensor_data)
        rag_task = self.rag_client.retrieve_hybrid(
            retrieval_query, equipment_id=req.equipment_id, k=8
        )
        
        anomaly_model, docs = await asyncio.gather(anomaly_task, rag_task)
        
        docs_text, doc_mapping = _format_docs(docs)
        valid_ids_list = list(doc_mapping.keys())
        valid_doc_ids_str = ", ".join(valid_ids_list) if valid_ids_list else "NONE"
        
        bundle = self.prompts.load("root_cause_analysis", req.prompt_version)
        prompt = bundle.template.format(
            anomaly_description=req.anomaly_description,
            sensor_data_json=json.dumps(req.sensor_data, ensure_ascii=False),
            anomaly_model_json=json.dumps(anomaly_model, ensure_ascii=False),
            retrieved_docs=docs_text,
            valid_doc_ids=valid_doc_ids_str  
        )
        
        result = await self.llm.invoke(prompt, json_mode=True)
        
        try:
            parsed = parse_llm_json(result.content, RootCauseResponse)
            
            for hyp in parsed.hypotheses:
                if str(hyp.source).strip().upper() == "NONE":
                    hyp.source = "NONE"
                    continue
                    
                text_to_search = f"{hyp.source} {hyp.evidence}"
                match = re.search(r'DOC[_\W]*(\d+)', text_to_search, re.IGNORECASE)
                
                if match:
                    normalized_tag = f"DOC_{match.group(1)}"
                    if normalized_tag in doc_mapping:
                        hyp.source = doc_mapping[normalized_tag]
                    else:
                        # PRODUCTION FIX: Graceful degradation for Zero-Context
                        # If the LLM hallucinates DOC_1 but no documents were found, coerce to NONE safely.
                        if not doc_mapping:
                            hyp.source = "NONE"
                        else:
                            raise ValueError(f"Hallucinated citation detected: {normalized_tag}")
                else:
                    # PRODUCTION FIX: Coerce non-matching text to NONE if zero documents exist.
                    if not doc_mapping:
                        hyp.source = "NONE"
                    else:
                        raise ValueError("Missing required citation tag in hypothesis.")
                    
            return parsed, result.provider, result.model, docs_text
            
        except Exception as e:
            if type(e).__name__ == "LLMOutputParseError":
                raise ValueError("Blocked: Output JSON was malformed.") from e
            raise ValueError(f"Blocked: Output failed grounding/citation constraints: {e}") from e