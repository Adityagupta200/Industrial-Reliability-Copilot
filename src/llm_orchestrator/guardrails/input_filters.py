import re
import logging
import threading
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
    _analyzer: Optional[AnalyzerEngine] = None
    _anonymizer: Optional[AnonymizerEngine] = None
    _is_loading: bool = False

    @classmethod
    def preload_engines(cls) -> None:
        """PRODUCTION FIX: Eagerly loads NLP models in the background at startup to prevent mid-request SLA spikes."""
        if cls._analyzer is not None or cls._is_loading:
            return
            
        cls._is_loading = True
        logger.info("Eagerly loading Presidio NLP engines to prevent latency spikes...")
        try:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            cls._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            cls._anonymizer = AnonymizerEngine()
            logger.info("Presidio NLP engines successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to preload Presidio NLP engines: {e}")
        finally:
            cls._is_loading = False

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
        """Redacts sensitive PII using Microsoft Presidio."""
        # Fallback in case the background thread hasn't finished loading yet during the very first millisecond of boot
        if cls._analyzer is None or cls._anonymizer is None:
            logger.warning("PII Guardrail skipped: NLP models are still loading in the background.")
            return text
            
        results = cls._analyzer.analyze(
            text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"], language="en"
        )
        anonymized_result = cls._anonymizer.anonymize(text=text, analyzer_results=results)
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
        """
        if cls.detect_prompt_injection(query):
            raise ValueError("Blocked: Potential prompt injection detected.")

        if cls.check_toxicity(query):
            raise ValueError("Blocked: Query violates toxicity policies.")

        return cls.redact_pii(query)

# Trigger the background loading immediately upon module import
threading.Thread(target=InputGuardrails.preload_engines, daemon=True).start()