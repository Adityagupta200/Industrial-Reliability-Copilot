import re
import logging
from typing import Optional
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"you are now a different assistant",
    r"disregard system prompt",
    r"bypass safety",
    r"forget instructions",
]


class InputGuardrails:
    # Class-level variables for the Singleton pattern
    _analyzer: Optional[AnalyzerEngine] = None
    _anonymizer: Optional[AnonymizerEngine] = None

    @classmethod
    def _get_engines(cls) -> tuple[AnalyzerEngine, AnonymizerEngine]:
        """Lazy-loads the heavy NLP models only when an actual safe request arrives."""
        if cls._analyzer is None or cls._anonymizer is None:
            logger.info("Initializing Presidio NLP engines lazily...")

            # --- PHASE 5 Production Optimization: Constrained Edge Deployment ---
            # Configure Presidio to use the lightweight 'sm' model to prevent OOM crashes
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            cls._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            cls._anonymizer = AnonymizerEngine()

        return cls._analyzer, cls._anonymizer

    @staticmethod
    def detect_prompt_injection(text: str, threshold: int = 1) -> bool:
        """Returns True if potential prompt injection is detected based on score."""
        text_lower = text.lower()
        score = sum(1 for pattern in PROMPT_INJECTION_PATTERNS if re.search(pattern, text_lower))
        if score >= threshold:
            logger.warning(f"Prompt injection detected! Score: {score}")
            return True
        return False

    @classmethod
    def redact_pii(cls, text: str) -> str:
        """Triggers lazy loading and redacts sensitive PII using Microsoft Presidio."""
        analyzer, anonymizer = cls._get_engines()
        results = analyzer.analyze(
            text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"], language="en"
        )
        anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_result.text

    @staticmethod
    def check_toxicity(text: str) -> bool:
        """Mock for API integration (e.g., Azure Content Safety)."""
        toxic_keywords = ["harm", "destroy", "sabotage"]
        return any(word in text.lower() for word in toxic_keywords)

    @classmethod
    def process(cls, query: str) -> str:
        """
        Runs guardrails in optimized order.
        O(1) Regex checks run first. If blocked, it safely raises an exception
        without ever utilizing the heavy O(N) NLP PII scanner.
        """
        if cls.detect_prompt_injection(query):
            raise ValueError("Blocked: Potential prompt injection detected.")

        if cls.check_toxicity(query):
            raise ValueError("Blocked: Query violates toxicity policies.")

        return cls.redact_pii(query)
