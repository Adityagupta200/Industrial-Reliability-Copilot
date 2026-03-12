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
    def check_citations(answer: str, min_citations: int = 2) -> bool:
        """Ensure minimum number of [Doc ID] citations."""
        citations = re.findall(r"\[.*?\]", answer)
        if len(citations) < min_citations:
            logger.warning(f"Failed citation check: only {len(citations)} found.")
            return False
        return True

    @staticmethod
    async def check_groundedness(llm_client, context: str, answer: str) -> float:
        """LLM-as-a-judge: Ensure answer relies purely on retrieved context."""
        prompt = (
            "Is the following answer fully supported by the provided context? "
            "Reply strictly with a single number between 0.0 and 1.0, where 1.0 means fully supported.\n\n"
            f"Context: {context}\n\nAnswer: {answer}"
        )
        try:
            # Assumes your LLMClient has an async generate method.
            # If standard OpenAI, use `await llm_client.client.chat.completions.create(...)`
            result = await llm_client.agenerate(prompt=prompt, temperature=0.0, max_tokens=10)

            # Extract the float from the result
            match = re.search(r"0\.\d+|1\.0|0|1", result)
            return float(match.group()) if match else 0.0
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}")
            return 0.0

    @classmethod
    async def validate_output(cls, llm_client, context: str, answer: str) -> Tuple[bool, str]:
        """Runs all output guardrails in parallel to stay under <500ms budget."""
        safety_task = asyncio.to_thread(cls.check_safety, answer)
        citation_task = asyncio.to_thread(cls.check_citations, answer)
        groundedness_task = cls.check_groundedness(llm_client, context, answer)

        is_safe, has_citations, grounded_score = await asyncio.gather(
            safety_task, citation_task, groundedness_task
        )

        if not is_safe:
            return False, "Blocked: Contains unsafe procedural recommendations."
        if not has_citations:
            return False, "Blocked: Output lacks required citations."
        if grounded_score < 0.8:
            return False, "Blocked: Output is not adequately grounded in retrieved context."

        return True, "Valid"
