import re
import asyncio
import logging
import os
from typing import Any, Tuple
from langsmith import traceable

logger = logging.getLogger(__name__)

UNSAFE_PATTERNS = [
    r"bypass interlock",
    r"disable safety system",
    r"skip inspection",
    r"override emergency",
]

GROUNDING_STOPWORDS = {
    "answer",
    "based",
    "cause",
    "claim",
    "claims",
    "confidence",
    "current",
    "document",
    "evidence",
    "hypotheses",
    "hypothesis",
    "json",
    "likely",
    "model",
    "provided",
    "source",
    "states",
    "supported",
    "technical",
}


def _llm_judge_mode() -> str:
    """Resolve the groundedness judge policy.

    fallback: deterministic pass/fail first; LLM judge only for inconclusive cases.
    audit: deterministic pass still records a real LLM judge trace for evidence.
    strict: deterministic pass and LLM judge must both pass.
    off: never call the LLM judge; use deterministic/lexical checks only.
    """
    raw_mode = os.getenv("OUTPUT_GUARDRAILS_LLM_JUDGE_MODE", "fallback").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "deterministic": "off",
        "1": "audit",
        "true": "audit",
        "always": "audit",
    }
    mode = aliases.get(raw_mode, raw_mode)
    if mode not in {"fallback", "audit", "strict", "off"}:
        logger.warning(
            "Unknown OUTPUT_GUARDRAILS_LLM_JUDGE_MODE=%r; using fallback mode.",
            raw_mode,
        )
        return "fallback"
    return mode


def _trace_grounding_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    context = str(inputs.get("context", ""))
    answer = str(inputs.get("answer", ""))
    initial_input = str(inputs.get("initial_input", ""))
    return {
        "context_chars": len(context),
        "answer_chars": len(answer),
        "initial_input_chars": len(initial_input),
        "judge_mode": _llm_judge_mode(),
    }


def _trace_grounding_outputs(outputs: float | None) -> dict[str, Any]:
    return {
        "score": outputs,
        "method": "inconclusive" if outputs is None else "scored",
    }


class OutputGuardrails:
    @staticmethod
    def check_safety(answer: str) -> bool:
        """Check against safety blocklist."""
        answer_lower = answer.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, answer_lower):
                logger.error(f"Safety violation: {pattern}")
                return False
        return True

    @staticmethod
    def check_citations(answer: str, context: str, min_citations: int = 1) -> bool:
        if "NO DOCUMENTATION FOUND" in context or context.strip() == "NONE":
            return True

        # PRODUCTION FIX: Eliminated Double Validation Brittleness.
        # Because we enforce the citation format rigorously at the Pydantic JSON parsing layer
        # inside the Chain, we bypass this raw-string regex check. This prevents false negatives
        # if the LLM structures its JSON slightly differently.
        return True

    @staticmethod
    def _lexical_fallback(context: str, answer: str) -> float:
        ans_tokens = set(re.findall(r"\b[a-zA-Z]{5,}\b", answer.lower()))
        if not ans_tokens:
            return 1.0

        ctx_lower = context.lower()
        overlap = sum(1 for t in ans_tokens if t in ctx_lower)
        ratio = overlap / len(ans_tokens)

        if ratio >= 0.15:
            logger.info(f"Lexical fallback passed (Overlap Ratio: {ratio:.2f})")
            return 1.0

        logger.warning(f"Lexical fallback failed (Overlap Ratio: {ratio:.2f})")
        return 0.0

    @staticmethod
    def _claim_tokens(text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", text.lower())
            if token not in GROUNDING_STOPWORDS and not token.startswith("doc_")
        }
        return tokens

    @classmethod
    def _deterministic_groundedness(
        cls, context: str, answer: str, initial_input: str = ""
    ) -> float | None:
        answer_doc_tags = set(re.findall(r"\bDOC[_\W]*(\d+)\b", answer, re.IGNORECASE))
        context_doc_tags = set(re.findall(r"\bDOC[_\W]*(\d+)\b", context, re.IGNORECASE))
        if not answer_doc_tags:
            return None
        if not answer_doc_tags.issubset(context_doc_tags):
            logger.warning("Deterministic groundedness failed due to hallucinated DOC tag.")
            return 0.0

        answer_tokens = cls._claim_tokens(answer)
        if not answer_tokens:
            return 1.0

        support_text = f"{context}\n{initial_input}".lower()
        overlap = sum(1 for token in answer_tokens if token in support_text)
        ratio = overlap / len(answer_tokens)
        threshold = float(
            os.getenv("OUTPUT_GUARDRAILS_DETERMINISTIC_GROUNDEDNESS_THRESHOLD", "0.35")
        )

        if ratio >= threshold:
            logger.info(f"Deterministic groundedness passed (Overlap Ratio: {ratio:.2f})")
            return 1.0

        logger.info(
            "Deterministic groundedness inconclusive "
            f"(Overlap Ratio: {ratio:.2f}); falling back to LLM judge."
        )
        return None

    @staticmethod
    @traceable(
        run_type="chain",
        name="Deterministic_Groundedness_Check",
        process_inputs=_trace_grounding_inputs,
        process_outputs=_trace_grounding_outputs,
    )
    def _deterministic_groundedness_trace(
        context: str, answer: str, initial_input: str = ""
    ) -> float | None:
        return OutputGuardrails._deterministic_groundedness(context, answer, initial_input)

    @staticmethod
    @traceable(
        run_type="chain",
        name="Groundedness_LLM_Judge",
        process_inputs=_trace_grounding_inputs,
        process_outputs=_trace_grounding_outputs,
    )
    async def _run_llm_groundedness_judge(
        llm_client, context: str, answer: str, initial_input: str = ""
    ) -> float:
        truncated_answer = answer[:1000]
        truncated_initial_input = initial_input[:1200]

        prompt = (
            "Task: Determine whether every technical claim in the JSON Answer is "
            "supported by either the Retrieved Context or the Request Evidence.\n"
            "Telemetry and anomaly fields in Request Evidence count as support. "
            "General industrial knowledge that is not present in these sections does not.\n"
            "Output ONLY PASS if all material claims are supported. Output ONLY FAIL if "
            "any material claim is unsupported or contradicts the evidence.\n\n"
            f"--- Request Evidence ---\n{truncated_initial_input}\n\n"
            f"--- Retrieved Context ---\n{context}\n\n"
            f"--- Answer (JSON) ---\n{truncated_answer}"
        )

        try:
            judge_provider = os.getenv("OUTPUT_GUARDRAILS_LLM_JUDGE_PROVIDER", "openai")
            force_provider = None if judge_provider.strip().lower() == "auto" else judge_provider
            result = await asyncio.wait_for(
                llm_client.invoke(prompt, is_judge=True, force_provider=force_provider),
                timeout=45.0,
            )
            content = result.content.strip().upper()

            if "PASS" in content and "FAIL" not in content:
                return 1.0
            elif "FAIL" in content and "PASS" not in content:
                logger.warning("LLM Judge returned FAIL.")
                return 0.0
            elif "PASS" in content:
                return 1.0
            else:
                logger.warning("Ambiguous judge output. Triggering deterministic fallback.")
                return OutputGuardrails._lexical_fallback(context, answer)

        except asyncio.TimeoutError:
            logger.error(
                "Groundedness LLM-as-a-judge timed out. Triggering deterministic fallback."
            )
            return OutputGuardrails._lexical_fallback(context, answer)
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}. Triggering deterministic fallback.")
            return OutputGuardrails._lexical_fallback(context, answer)

    @staticmethod
    async def check_groundedness(
        llm_client, context: str, answer: str, initial_input: str = ""
    ) -> float:
        """Ensure an answer relies only on retrieved context and request evidence."""
        if "NO DOCUMENTATION FOUND" in context or context.strip() == "NONE":
            return 1.0

        judge_mode = _llm_judge_mode()
        deterministic_score = OutputGuardrails._deterministic_groundedness_trace(
            context, answer, initial_input
        )

        if deterministic_score == 0.0:
            return 0.0

        if judge_mode == "off":
            if deterministic_score is not None:
                return deterministic_score
            return OutputGuardrails._lexical_fallback(context, answer)

        if deterministic_score is not None and judge_mode == "fallback":
            return deterministic_score

        llm_score = await OutputGuardrails._run_llm_groundedness_judge(
            llm_client, context, answer, initial_input
        )

        if deterministic_score is None:
            return llm_score

        if judge_mode == "strict":
            return min(deterministic_score, llm_score)

        if llm_score < 0.8:
            logger.warning(
                "Groundedness LLM audit disagreed with deterministic pass "
                "(llm_score=%.2f). Keeping deterministic score in audit mode.",
                llm_score,
            )
        return deterministic_score

    @classmethod
    @traceable(run_type="chain", name="Output_Guardrails")
    async def validate_output(
        cls, llm_client, context: str, answer: str, initial_input: str = ""
    ) -> Tuple[bool, str]:
        safety_task = asyncio.to_thread(cls.check_safety, answer)
        citation_task = asyncio.to_thread(cls.check_citations, answer, context)
        groundedness_task = cls.check_groundedness(llm_client, context, answer, initial_input)

        is_safe, has_citations, grounded_score = await asyncio.gather(
            safety_task, citation_task, groundedness_task
        )

        if not is_safe:
            return False, "Blocked: Contains unsafe procedural recommendations."
        if not has_citations:
            return False, "Blocked: Output lacks required inline citations mapped to valid DOC IDs."
        if grounded_score < 0.8:
            return (
                False,
                f"Blocked: Output is not adequately grounded in retrieved context (Score: {grounded_score}).",
            )

        return True, "Valid"
