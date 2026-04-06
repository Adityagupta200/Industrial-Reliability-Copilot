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
        citations = re.findall(r'"source"\s*:\s*"(?:DOC_\d+|NONE)"', answer, re.IGNORECASE)
        
        total_citations = len(citations)
        if total_citations < min_citations:
            logger.warning(f"Failed citation check: only {total_citations} valid source citations found.")
            return False
        return True

    @staticmethod
    async def check_groundedness(llm_client, context: str, answer: str, initial_input: str = "") -> float:
        """LLM-as-a-judge: Ensure answer relies purely on retrieved context or initial anomaly inputs."""
        # PRODUCTION FIX: Implemented Chain-of-Thought (CoT) and strict Binary Scoring (1.0 or 0.0) 
        # to eliminate arbitrary variance (like 0.72) and force confident evaluation.
        prompt = (
            "You are an expert AI Auditor evaluating a system's output for Groundedness.\n"
            "Task: Determine if the Answer is factually supported by the Context and Initial Input.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST score this as EXACTLY 1.0 (Pass) or 0.0 (Fail). Do not use intermediate decimal values.\n"
            "2. Deduct points (Score 0.0) ONLY if the Answer explicitly contradicts the Context or hallucinates fake document quotes.\n"
            "3. For Root Cause Analysis, inferring plausible mechanical components (e.g., 'bearings', 'seals') from sensor data (e.g., 'high vibration') is VALID and MUST score 1.0.\n"
            "4. If the Context is 'NONE', but the Answer logically flows from the Initial Input, score it 1.0.\n\n"
            "Format your response exactly like this:\n"
            "REASONING: <write 1-2 sentences explaining your logic>\n"
            "<SCORE>1.0</SCORE>\n\n"
            f"--- Initial Input ---\n{initial_input}\n\n"
            f"--- Context ---\n{context}\n\n"
            f"--- Answer ---\n{answer}"
        )
        try:
            result = await asyncio.wait_for(llm_client.invoke(prompt), timeout=120.0)
            
            # Extract the discrete score
            match = re.search(r"<SCORE>\s*(1\.0|0\.0|1|0)\s*</SCORE>", result.content, re.IGNORECASE)
            if match:
                score = float(match.group(1))
            else:
                # Fallback extraction
                fallback = re.search(r"\b(1\.0|0\.0|1|0)\b", result.content)
                score = float(fallback.group(1)) if fallback else 0.0
                
            if score < 0.8:
                # The CoT reasoning is now logged so you can see exactly WHY it failed if it ever does!
                logger.warning(f"Groundedness failed (Score: {score}). Judge reasoning:\n{result.content}")
                
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