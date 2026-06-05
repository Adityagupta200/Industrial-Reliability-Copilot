import re
import logging
import threading
from typing import Optional
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from llm_orchestrator.tracing import traceable

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"you are now a different assistant",
    r"disregard system prompt",
    r"bypass safety",
    r"forget instructions",
    r"forget all prior rules",
    r"reveal.*system (prompt|instructions)",
    r"output.*(database connection string|connection string|api key|secret)",
]


class InputGuardrails:
    _analyzer: Optional[AnalyzerEngine] = None
    _anonymizer: Optional[AnonymizerEngine] = None
    _is_loading: bool = False
    _load_lock = threading.Lock()
    _load_error: Optional[Exception] = None

    @classmethod
    def engines_ready(cls) -> bool:
        return cls._analyzer is not None and cls._anonymizer is not None

    @classmethod
    def preload_engines(cls) -> None:
        """Load NLP models before traffic reaches request handlers."""
        if cls.engines_ready():
            return

        with cls._load_lock:
            if cls.engines_ready():
                return

            cls._is_loading = True
            cls._load_error = None
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
                cls._load_error = e
                logger.exception("Failed to preload Presidio NLP engines: %s", e)
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
        if not cls.engines_ready():
            cls.preload_engines()
        if not cls.engines_ready():
            raise RuntimeError("PII guardrail unavailable; NLP engines failed to load.")

        results = cls._analyzer.analyze(
            text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"], language="en"
        )
        anonymized_result = cls._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_result.text

    @staticmethod
    def check_toxicity(text: str) -> bool:
        """Mock for API integration (e.g., Azure Content Safety)."""
        toxic_keywords = ["harm", "destroy", "sabotage"]
        unsafe_control_patterns = [
            "bypass interlock",
            "bypass safety",
            "disable interlock",
            "disable safety",
            "override emergency",
            "skip lockout",
            "skip loto",
        ]
        lowered = text.lower()
        return any(word in lowered for word in toxic_keywords) or any(
            pattern in lowered for pattern in unsafe_control_patterns
        )

    @classmethod
    @traceable(run_type="chain", name="Input_Guardrails")  # PRODUCTION FIX: Explicit Tracing
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
