from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from ..schemas import RemediationRequest, RemediationResponse, RetrievedDoc
from ..utils.json_parse import parse_llm_json
from ..clients.rag_client import RAGClient

SECTION_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
NUMBERED_LINE_RE = re.compile(r"^\s*\d+\.\s+(?P<text>.+?)\s*$")
BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+(?P<text>.+?)\s*$")
RELEVANCE_TOKEN_RE = re.compile(r"[a-z0-9-]+")
RELEVANCE_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "the",
    "what",
    "when",
    "with",
    "should",
    "follow",
}
FAILURE_RELEVANCE_TERMS: dict[str, tuple[str, ...]] = {
    "bearing_failure": (
        "bearing",
        "lubrication",
        "grease",
        "misalignment",
        "vibration",
        "scoring",
    ),
    "cavitation": (
        "cavitation",
        "npsh",
        "suction",
        "strainer",
        "air ingress",
        "gravel",
        "fluctuating discharge pressure",
        "reduced flow",
    ),
    "overheating": (
        "overheating",
        "temperature",
        "ventilation",
        "cooling",
        "load current",
        "friction",
    ),
    "sensor_calibration": (
        "calibration",
        "recalibration",
        "transducer",
        "sensor",
        "deadweight tester",
        "zero point",
        "span",
    ),
}


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


def _is_unsafe_interlock_request(query: str | None) -> bool:
    text = (query or "").lower()
    unsafe_terms = [
        "bypass interlock",
        "bypass safety",
        "disable interlock",
        "disable safety",
        "override emergency",
        "skip loto",
        "skip lockout",
    ]
    return any(term in text for term in unsafe_terms)


def _section_lines(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            current = section_match.group("name").strip().lower()
            sections.setdefault(current, [])
            continue

        numbered = NUMBERED_LINE_RE.match(line)
        bullet = BULLET_LINE_RE.match(line)
        if numbered:
            sections.setdefault(current, []).append(numbered.group("text").strip())
        elif bullet:
            sections.setdefault(current, []).append(bullet.group("text").strip())

    return sections


def _source_name(doc: RetrievedDoc) -> str:
    meta_source = doc.metadata.get("source_file") or doc.metadata.get("source_id")
    raw_source = meta_source if meta_source else getattr(doc, "source", None)
    if not raw_source or str(raw_source).lower() in {"hybrid", "semantic", "keyword", "unknown"}:
        raw_source = f"maintenance_procedure_{doc.id[:6]}.md"

    source = os.path.basename(str(raw_source))
    if not source.endswith(".pdf") and not source.endswith(".md"):
        source += ".md"
    return source


def _unique(items: list[str], *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        out.append(normalized)
        if limit is not None and len(out) >= limit:
            break
    return out


def _relevance_tokens(text: str) -> set[str]:
    return {
        token
        for token in RELEVANCE_TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in RELEVANCE_STOPWORDS
    }


def _failure_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")


def _remediation_doc_relevance(req: RemediationRequest, doc: RetrievedDoc) -> float:
    query = req.user_query or ""
    failure_key = _failure_key(req.failure_mode)
    source = _source_name(doc)
    metadata_text = " ".join(str(value) for value in doc.metadata.values() if value is not None)
    haystack = f"{source}\n{metadata_text}\n{doc.text}".lower()
    query_lower = query.lower()

    score = float(len(_relevance_tokens(query).intersection(_relevance_tokens(haystack))))

    if failure_key:
        normalized_haystack = _failure_key(haystack)
        if failure_key in normalized_haystack:
            score += 12.0

        for term in FAILURE_RELEVANCE_TERMS.get(failure_key, ()):
            if term in haystack:
                score += 4.0
                if term in query_lower:
                    score += 3.0

    source_lower = source.lower()
    if failure_key and failure_key.replace("_", "-") in source_lower:
        score += 4.0

    doc_equipment = str(doc.metadata.get("equipment_id") or "").lower()
    req_equipment = str(req.equipment_id or "").lower()
    if req_equipment and doc_equipment and doc_equipment == req_equipment:
        score += 2.0

    return score


def _select_relevant_remediation_docs(
    req: RemediationRequest,
    docs: list[RetrievedDoc],
) -> list[RetrievedDoc]:
    scored = [(doc, _remediation_doc_relevance(req, doc)) for doc in docs]
    if not scored:
        return []

    scored.sort(key=lambda item: item[1], reverse=True)
    best_score = scored[0][1]
    if best_score <= 0.0:
        return docs

    selected: list[RetrievedDoc] = []
    for doc, score in scored:
        if score >= best_score - 4.0:
            selected.append(doc)

    # A specific procedure should drive the extractive fast path. Including lower-ranked
    # adjacent procedures can exhaust the response step limit with irrelevant work.
    return selected[:2]


def _build_extractive_remediation(
    req: RemediationRequest,
    docs: list[RetrievedDoc],
) -> RemediationResponse:
    safety_warnings: list[str] = []
    tools_required: list[str] = []
    steps: list[str] = []
    sources: list[str] = []

    for doc in docs:
        source = _source_name(doc)
        sources.append(source)
        sections = _section_lines(doc.text)

        for warning in sections.get("safety", []):
            safety_warnings.append(f"{warning} ({source})")
        for tool in sections.get("tools", []):
            tools_required.append(f"{tool} ({source})")

        procedure_steps = [*sections.get("steps", []), *sections.get("verification", [])]
        for step in procedure_steps:
            steps.append(f"{step} ({source})")

    query_text = (req.user_query or "").lower()
    if "return" in query_text or "restart" in query_text or "closeout" in query_text:
        steps.append(
            "Record the verification result in the maintenance log or CMMS and return "
            "equipment to service only after acceptance checks pass."
        )

    if "bearing" in query_text and not tools_required:
        tools_required.extend(["Bearing puller", "Torque wrench", "Dial indicator", "Grease gun"])

    return RemediationResponse(
        safety_warnings=_unique(safety_warnings, limit=6),
        tools_required=_unique(tools_required, limit=8),
        steps=_unique(steps, limit=10),
        sources=_unique(sources),
    )


@dataclass(frozen=True)
class RemediationChain:
    llm: LLMClient
    prompts: PromptLoader
    rag_client: RAGClient

    async def run(self, req: RemediationRequest) -> tuple[RemediationResponse, str, str, str]:
        req.equipment_id, req.failure_mode = _extract_missing_entities(
            req.user_query, req.equipment_id, req.failure_mode
        )

        if _is_unsafe_interlock_request(req.user_query):
            response = RemediationResponse(
                safety_warnings=[
                    "Safety interlocks, LOTO, and emergency protections must not be bypassed."
                ],
                tools_required=[],
                steps=[
                    "Cannot bypass interlocks or safety controls. Stop the restart, keep the "
                    "asset isolated, and follow the approved procedure with supervisor approval."
                ],
                sources=["NONE"],
            )
            return response, "system", "safety-policy-v1", "NONE"

        retrieval_query = " ".join(
            part
            for part in [
                req.user_query or "",
                f"failure mode {req.failure_mode}",
                f"equipment {req.equipment_id}" if req.equipment_id else "",
            ]
            if part
        )

        docs = await self.rag_client.retrieve_procedures_direct(
            retrieval_query,
            equipment_id=req.equipment_id,
            k=6,
        )
        if not docs:
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

        docs = _select_relevant_remediation_docs(req, docs)
        formatted_context, doc_mapping = _format_docs(docs)

        extractive_response = _build_extractive_remediation(req, docs)
        if extractive_response.steps or extractive_response.safety_warnings:
            return (
                extractive_response,
                "rules+retrieval",
                "remediation-procedure-fast-path-v1",
                formatted_context,
            )

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
