from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..schemas import RemediationRequest, RemediationResponse, RetrievedDoc
from ..utils.json_parse import parse_llm_json
from ..clients.rag_client import RAGClient


def _format_docs(docs: list[RetrievedDoc]) -> tuple[str, dict[str, str]]:
    parts = []
    mapping = {}
    for i, d in enumerate(docs, start=1):
        meta_source = d.metadata.get("source_file") or d.metadata.get("source_id")
        raw_source = meta_source if meta_source else getattr(d, "source", None)

        if not raw_source or str(raw_source).lower() in [
            "hybrid",
            "semantic",
            "keyword",
            "unknown",
        ]:
            raw_source = f"maintenance_procedure_{d.id[:6]}.md"

        real_source = os.path.basename(str(raw_source))
        if not real_source.endswith(".pdf") and not real_source.endswith(".md"):
            real_source += ".md"

        doc_tag = f"DOC_{i}"
        mapping[doc_tag] = real_source

        parts.append(f"[{doc_tag}]\n{d.text}\n")

    return "\n---\n".join(parts), mapping


def _extract_missing_entities(
    user_query: str | None, current_eq: str | None, current_fail: str | None
) -> tuple[str | None, str]:
    eq_id = current_eq
    failure = current_fail or ""

    if user_query:
        if not eq_id:
            match = re.search(r"([A-Z]-\d+)", user_query, re.IGNORECASE)
            if match:
                base_id = match.group(1).upper()
                q_lower = user_query.lower()
                if "pump" in q_lower:
                    eq_id = f"pump_{base_id}"
                elif "motor" in q_lower:
                    eq_id = f"motor_{base_id}"
                elif "compressor" in q_lower:
                    eq_id = f"compressor_{base_id}"
                elif "turbofan" in q_lower:
                    eq_id = f"turbofan_{base_id}"

        if not failure:
            q_lower = user_query.lower()
            if "bearing" in q_lower:
                failure = "bearing_failure"
            elif "cavitation" in q_lower:
                failure = "cavitation"
            elif "recalibrate" in q_lower or "sensor" in q_lower:
                failure = "sensor_calibration"
            else:
                failure = "general_maintenance"

    return eq_id, failure


def _sanitize_list(items: list[str], doc_mapping: dict[str, str]) -> list[str]:
    sanitized = []
    for item in items:
        new_item = item
        found_tags = set(re.findall(r"DOC[_\W]*(\d+)", new_item, re.IGNORECASE))
        for tag_num in found_tags:
            normalized_tag = f"DOC_{tag_num}"
            if normalized_tag in doc_mapping:
                real_source = doc_mapping[normalized_tag]
                new_item = re.sub(
                    rf"\bDOC[_\W]*{tag_num}\b", real_source, new_item, flags=re.IGNORECASE
                )
        sanitized.append(new_item)
    return sanitized


@dataclass(frozen=True)
class RemediationChain:
    llm: LLMClient
    prompts: PromptLoader
    rag_client: RAGClient

    async def run(self, req: RemediationRequest) -> tuple[RemediationResponse, str, str, str]:
        req.equipment_id, req.failure_mode = _extract_missing_entities(
            req.user_query, req.equipment_id, req.failure_mode
        )

        retrieval_query = " ".join(
            part
            for part in [
                req.user_query or "",
                f"failure mode {req.failure_mode}",
                f"equipment {req.equipment_id}" if req.equipment_id else "",
            ]
            if part
        )

        docs = await self.rag_client.retrieve_procedures(
            req.failure_mode,
            equipment_id=req.equipment_id,
            k=6,
            query=retrieval_query,
        )

        if not docs:
            fallback_response = RemediationResponse(
                safety_warnings=[
                    f"No technical documentation was retrieved for {req.failure_mode}, "
                    "so safe repair guidance cannot be provided from the available context."
                ],
                tools_required=[],
                steps=[],
                sources=[],
            )
            return fallback_response, "system", "no_llm_called", "No context retrieved."

        formatted_context, doc_mapping = _format_docs(docs)

        valid_ids_list = list(doc_mapping.keys())
        valid_doc_ids_str = ", ".join(valid_ids_list)

        bundle = self.prompts.load("remediation_guidance", req.prompt_version)
        prompt = bundle.template.format(
            failure_mode=req.failure_mode,
            retrieved_docs=formatted_context,
            valid_doc_ids=valid_doc_ids_str,
        )

        result = await self.llm.invoke(prompt, json_mode=True)

        # PRODUCTION FIX: Wrap LLM parsing in a try-except block to prevent 500 errors
        try:
            parsed = parse_llm_json(result.content, RemediationResponse)
        except Exception as e:
            if type(e).__name__ == "LLMOutputParseError":
                raise ValueError("Blocked: Output JSON was malformed.") from e
            raise ValueError(f"Blocked: Output failed constraints: {e}") from e

        parsed.safety_warnings = _sanitize_list(parsed.safety_warnings, doc_mapping)
        parsed.tools_required = _sanitize_list(parsed.tools_required, doc_mapping)
        parsed.steps = _sanitize_list(parsed.steps, doc_mapping)

        sanitized_sources = set()
        for source in parsed.sources:
            found_tags = re.findall(r"DOC[_\W]*(\d+)", source, re.IGNORECASE)
            if found_tags:
                for tag_num in found_tags:
                    normalized_tag = f"DOC_{tag_num}"
                    if normalized_tag in doc_mapping:
                        sanitized_sources.add(doc_mapping[normalized_tag])
            else:
                sanitized_sources.add(source)

        parsed.sources = list(sanitized_sources)

        return parsed, result.provider, result.model, formatted_context
