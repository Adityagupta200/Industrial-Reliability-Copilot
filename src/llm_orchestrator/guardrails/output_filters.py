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
    def check_citations(answer: str, context: str, min_citations: int = 1) -> bool:
        if "NO DOCUMENTATION FOUND" in context or context.strip() == "NONE":
            return True
            
        citations = re.findall(r'DOC[_\W]*\d+', answer, re.IGNORECASE)
        
        total_citations = len(citations)
        if total_citations < min_citations:
            logger.warning(f"Failed citation check: only {total_citations} valid source citations found (minimum {min_citations}).")
            return False
        return True

    @staticmethod
    async def check_groundedness(llm_client, context: str, answer: str, initial_input: str = "") -> float:
        """LLM-as-a-judge: Ensure answer relies purely on retrieved context."""
        if "NO DOCUMENTATION FOUND" in context or context.strip() == "NONE":
            return 1.0

        prompt = (
            "You are an expert AI Auditor evaluating a system's output for Groundedness.\n"
            "Task: Determine if the Answer is factually supported by the Context.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST score this as EXACTLY 1.0 (Pass) or 0.0 (Fail). Do not use intermediate decimal values.\n"
            "2. Deduct points (Score 0.0) if the Answer explicitly contradicts the Context or hallucinates fake document quotes.\n"
            "3. The Answer MUST rely on the Context. Logical inferences based ONLY on Initial Input without Context support MUST score 0.0.\n\n"
            "Format your response exactly like this:\n"
            "REASONING: <write 1-2 sentences explaining your logic>\n"
            "<SCORE>1.0</SCORE>\n\n"
            f"--- Initial Input ---\n{initial_input}\n\n"
            f"--- Context ---\n{context}\n\n"
            f"--- Answer ---\n{answer}"
        )
        try:
            # PRODUCTION FIX: Relaxed the hardcoded 3.0s timeout to 120s for local evaluation 
            result = await asyncio.wait_for(llm_client.invoke(prompt, is_judge=True), timeout=120.0)
            
            match = re.search(r"<SCORE>\s*(1\.0|0\.0|1|0)\s*</SCORE>", result.content, re.IGNORECASE)
            if match:
                score = float(match.group(1))
            else:
                fallback = re.search(r"\b(1\.0|0\.0|1|0)\b", result.content)
                score = float(fallback.group(1)) if fallback else 0.0
                
            if score < 0.8:
                logger.warning(f"Groundedness failed (Score: {score}). Judge reasoning:\n{result.content}")
                
            return score
            
        except asyncio.TimeoutError:
            logger.error("Groundedness LLM-as-a-judge timed out. Defaulting to 0.0")
            return 0.0
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}")
            return 0.0

    @classmethod
    async def validate_output(cls, llm_client, context: str, answer: str, initial_input: str = "") -> Tuple[bool, str]:
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
            return False, f"Blocked: Output is not adequately grounded in retrieved context (Score: {grounded_score})."
        
        return True, "Valid"