from __future__ import annotations

from dataclasses import dataclass

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..schemas import RemediationRequest, RemediationResponse, RetrievedDoc
from ..utils.json_parse import parse_llm_json
from ..clients.rag_client import RAGClient


def _format_docs(docs: list[RetrievedDoc]) -> str:
    parts = []
    for d in docs:
        parts.append(f"[DOC:{d.id}] {d.title or 'Untitled'}\n{d.text}\n")
    return "\n---\n".join(parts)


@dataclass(frozen=True)
class RemediationChain:
    llm: LLMClient
    prompts: PromptLoader
    rag_client: RAGClient

    async def run(self, req: RemediationRequest) -> tuple[RemediationResponse, str, str]:
        # Step 1: retrieve procedure docs filtered by failure_mode
        docs = await self.rag_client.retrieve_procedures(
            req.failure_mode, equipment_id=req.equipment_id, k=6
        )

        # Step 2: format prompt
        bundle = self.prompts.load("remediation_guidance", req.prompt_version)
        prompt = bundle.template.format(
            failure_mode=req.failure_mode,
            retrieved_docs=_format_docs(docs),
        )

        # Step 3: call LLM and parse
        result = await self.llm.invoke(prompt)
        parsed = parse_llm_json(result.content, RemediationResponse)
        return parsed, result.provider, result.model
