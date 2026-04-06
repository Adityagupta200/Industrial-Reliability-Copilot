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
        """Ensure minimum number of valid inline DOC citations exist in the JSON."""
        # PRODUCTION FIX: Regex now accepts "NONE" as a valid explicit citation state 
        # (meaning the LLM correctly acknowledged lack of context).
        citations = re.findall(r'"source"\s*:\s*"(?:DOC_\d+|NONE)"', answer, re.IGNORECASE)
        
        total_citations = len(citations)
        if total_citations < min_citations:
            logger.warning(f"Failed citation check: only {total_citations} valid source citations found.")
            return False
        return True

    @staticmethod
    async def check_groundedness(llm_client, context: str, answer: str, initial_input: str = "") -> float:
        """LLM-as-a-judge: Ensure answer relies purely on retrieved context or initial anomaly inputs."""
        prompt = (
            "You are an Auditor. Rate if the Answer is fully supported by the Context OR the Initial Input.\n"
            "CRITICAL RULE: If the Answer mentions parts or procedures that are NOT explicitly "
            "described in the Context OR the Initial Input for the target equipment, you MUST score 0.0.\n"
            "EXCEPTION: If the Context is 'NONE' (or simply indicates no documents were found), but the Answer provides valid logical hypotheses based purely on the Initial Input without hallucinating unmentioned equipment procedures, score it 1.0.\n"
            "Reply ONLY with a single float number between 0.0 and 1.0.\n\n"
            f"Initial Input: {initial_input}\n\n"
            f"Context: {context}\n\nAnswer: {answer}"
        )
        try:
            # PRODUCTION FIX: Increased timeout to 120.0s. 
            # Local Ollama (Llama 3) running concurrently requires massive latency buffers compared to OpenAI.
            result = await asyncio.wait_for(llm_client.invoke(prompt), timeout=120.0)
            match = re.search(r"(1\.0|0\.\d+|0|1)", result.content)
            
            score = float(match.group()) if match else 0.0
            if score < 0.8:
                logger.warning(f"Groundedness failed (Score: {score}). Context provided: {context[:200]}...")
            return score
            
        except asyncio.TimeoutError:
            logger.error("Groundedness LLM-as-a-judge timed out after 120.0s. Defaulting to 0.0")
            return 0.0
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}")
            return 0.0

    @classmethod
    async def validate_output(cls, llm_client, context: str, answer: str, initial_input: str = "") -> Tuple[bool, str]:
        """Runs all output guardrails in parallel to stay under latency budget."""
        safety_task = asyncio.to_thread(cls.check_safety, answer)
        citation_task = asyncio.to_thread(cls.check_citations, answer)
        groundedness_task = cls.check_groundedness(llm_client, context, answer, initial_input)

        is_safe, has_citations, grounded_score = await asyncio.gather(
            safety_task, citation_task, groundedness_task
        )

        if not is_safe:
            return False, "Blocked: Contains unsafe procedural recommendations."
        if not has_citations:
            return False, "Blocked: Output lacks required inline citations mapped to valid DOC IDs or NONE."
        if grounded_score < 0.8:
            return False, f"Blocked: Output is not adequately grounded in retrieved context (Score: {grounded_score})."
        
        return True, "Valid"