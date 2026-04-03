import re
import asyncio
import logging
from typing import Tuple

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
    def check_citations(answer: str, min_citations: int = 1) -> bool:
        """Ensure minimum number of inline citations exist."""
        # FIXED: Correctly formatted raw string to prevent SyntaxError
        citations = re.findall(r"\", answer, re.IGNORECASE)
        fallback_citations = re.findall(r"\[DOC:.*?\]", answer, re.IGNORECASE)
        
        total_citations = len(citations) + len(fallback_citations)
        if total_citations < min_citations:
            logger.warning(f"Failed citation check: only {total_citations} found.")
            return False
        return True

    @staticmethod
    async def check_groundedness(llm_client, context: str, answer: str) -> float:
        """LLM-as-a-judge: Ensure answer relies purely on retrieved context."""
        prompt = (
            "You are an Auditor. Rate if the Answer is fully supported by the Context.\n"
            "CRITICAL RULE: If the Answer mentions parts or procedures (like crankshafts or engine blocks) "
            "that are NOT explicitly described in the Context for the target equipment, you MUST score 0.0.\n"
            "Reply ONLY with a single float number between 0.0 and 1.0, where 1.0 means fully supported.\n\n"
            f"Context: {context}\n\nAnswer: {answer}"
        )
        try:
            result = await llm_client.invoke(prompt)
            match = re.search(r"(1\.0|0\.\d+|0|1)", result.content)
            return float(match.group()) if match else 0.0
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}")
            return 0.0

    @classmethod
    async def validate_output(cls, llm_client, context: str, answer: str) -> Tuple[bool, str]:
        """Runs all output guardrails in parallel to stay under latency budget."""
        safety_task = asyncio.to_thread(cls.check_safety, answer)
        citation_task = asyncio.to_thread(cls.check_citations, answer)
        groundedness_task = cls.check_groundedness(llm_client, context, answer)

        is_safe, has_citations, grounded_score = await asyncio.gather(
            safety_task, citation_task, groundedness_task
        )

        if not is_safe:
            return False, "Blocked: Contains unsafe procedural recommendations."
        if not has_citations:
            return False, "Blocked: Output lacks required inline citations."
        if grounded_score < 0.8:
            return False, f"Blocked: Output is not adequately grounded in retrieved context (Score: {grounded_score}). Possible hallucination detected."

        return True, "Valid"