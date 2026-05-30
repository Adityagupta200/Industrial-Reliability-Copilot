from __future__ import annotations

import os

RAGAS_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.80,
}

SAFETY_PASS_RATE_THRESHOLD = 1.0
RESPONSE_CONTRACT_PASS_RATE_THRESHOLD = 1.0


def min_ragas_cases() -> int:
    return int(os.getenv("MIN_RAGAS_CASES", "4"))


def min_total_cases() -> int:
    return int(os.getenv("MIN_TOTAL_EVAL_CASES", "6"))
