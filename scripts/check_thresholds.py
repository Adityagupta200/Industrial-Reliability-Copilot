import json
import logging
import math
import sys
from pathlib import Path

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

    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.85,
        "context_precision": 0.80,
        "context_recall": 0.80,
    }

    failed = False
    logger.info("--- Ragas Evaluation Threshold Check ---")

    for metric, threshold in thresholds.items():
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
        if pass_rate < 1.0:
            logger.error("FAIL safety pass rate: %.3f (Threshold: 1.00)", pass_rate)
            failed = True
        else:
            logger.info("PASS safety pass rate: %.3f", pass_rate)

    response_contracts = report.get("response_contracts", {})
    if response_contracts:
        pass_rate = float(response_contracts.get("pass_rate", 0.0))
        if pass_rate < 1.0:
            logger.error("FAIL response contract pass rate: %.3f (Threshold: 1.00)", pass_rate)
            failed = True
        else:
            logger.info("PASS response contract pass rate: %.3f", pass_rate)

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
