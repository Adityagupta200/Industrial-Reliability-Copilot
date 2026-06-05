#!/usr/bin/env python3
import json
import logging
import math
import sys
from pathlib import Path

from evaluation_gate import (
    RAGAS_THRESHOLDS,
    RESPONSE_CONTRACT_PASS_RATE_THRESHOLD,
    SAFETY_PASS_RATE_THRESHOLD,
    min_ragas_cases,
    min_total_cases,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    results_path = Path("ragas_results.json")
    report_path = Path("data/evaluation_results/evaluation_report.json")

    try:
        with results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        logger.error("ragas_results.json not found. Did the evaluation run?")
        sys.exit(1)

    results = payload.get("ragas", payload)
    report = payload if "ragas" in payload else {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

    failed = False
    logger.info("--- Ragas Evaluation Threshold Check ---")

    for metric, threshold in RAGAS_THRESHOLDS.items():
        val = float(results.get(metric, 0.0))
        if math.isnan(val):
            val = 0.0
        if val < threshold:
            logger.error("FAIL %s: %.3f (Threshold: %.2f)", metric, val, threshold)
            failed = True
        else:
            logger.info("PASS %s: %.3f (Threshold: %.2f)", metric, val, threshold)

    safety = report.get("safety", {})
    if safety:
        pass_rate = float(safety.get("pass_rate", 0.0))
        if pass_rate < SAFETY_PASS_RATE_THRESHOLD:
            logger.error(
                "FAIL safety pass rate: %.3f (Threshold: %.2f)",
                pass_rate,
                SAFETY_PASS_RATE_THRESHOLD,
            )
            failed = True
        else:
            logger.info("PASS safety pass rate: %.3f", pass_rate)

    response_contracts = report.get("response_contracts", {})
    if response_contracts:
        pass_rate = float(response_contracts.get("pass_rate", 0.0))
        if pass_rate < RESPONSE_CONTRACT_PASS_RATE_THRESHOLD:
            logger.error(
                "FAIL response contract pass rate: %.3f (Threshold: %.2f)",
                pass_rate,
                RESPONSE_CONTRACT_PASS_RATE_THRESHOLD,
            )
            failed = True
        else:
            logger.info("PASS response contract pass rate: %.3f", pass_rate)

    case_count = report.get("case_count", {})
    ragas_cases = int(case_count.get("ragas", 0) or 0)
    total_cases = int(case_count.get("total", 0) or 0)
    if ragas_cases < min_ragas_cases():
        logger.error("FAIL Ragas case count: %s (Minimum: %s)", ragas_cases, min_ragas_cases())
        failed = True
    else:
        logger.info("PASS Ragas case count: %s", ragas_cases)

    if total_cases < min_total_cases():
        logger.error(
            "FAIL total golden case count: %s (Minimum: %s)", total_cases, min_total_cases()
        )
        failed = True
    else:
        logger.info("PASS total golden case count: %s", total_cases)

    if failed:
        case_metrics = report.get("case_metrics", [])
        if case_metrics:
            logger.error("Per-case Ragas metrics for debugging:")
            for row in case_metrics:
                logger.error(json.dumps(row, sort_keys=True))
        logger.error("Quality gate failed. Metrics are below production thresholds.")
        sys.exit(1)

    logger.info("Quality gate passed. Ready for review.")


if __name__ == "__main__":
    main()
