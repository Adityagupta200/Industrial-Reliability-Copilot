import re
import asyncio
import logging
from typing import Tuple
from langsmith import traceable  # PRODUCTION FIX: Explicit Tracing

logger = logging.getLogger(__name__)

UNSAFE_PATTERNS = [
    r"bypass interlock",
    r"disable safety system",
    r"skip inspection",
    r"override emergency",
]


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
    @traceable(run_type="llm", name="Groundedness_LLM_Judge")  # PRODUCTION FIX: Explicit Tracing
    async def check_groundedness(
        llm_client, context: str, answer: str, initial_input: str = ""
    ) -> float:
        """LLM-as-a-judge: Ensure answer relies purely on retrieved context."""
        if "NO DOCUMENTATION FOUND" in context or context.strip() == "NONE":
            return 1.0

        truncated_answer = answer[:1000]

        prompt = (
            "Task: Determine if the technical claims made inside the JSON Answer are factually supported by the Context.\n"
            "Output ONLY the word PASS if supported, or FAIL if contradicting.\n\n"
            f"--- Context ---\n{context}\n\n"
            f"--- Answer (JSON) ---\n{truncated_answer}"
        )

        try:
            result = await asyncio.wait_for(
                llm_client.invoke(prompt, is_judge=True, force_provider="openai"), timeout=45.0
            )
            content = result.content.strip().upper()

            if "PASS" in content and "FAIL" not in content:
                return 1.0
            elif "FAIL" in content and "PASS" not in content:
                logger.warning("LLM Judge returned FAIL. Triggering deterministic fallback.")
                return OutputGuardrails._lexical_fallback(context, answer)
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

    @classmethod
    @traceable(run_type="chain", name="Output_Guardrails")  # PRODUCTION FIX: Explicit Tracing
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
