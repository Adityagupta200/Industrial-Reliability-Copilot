import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        with open("ragas_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error("ragas_results.json not found. Did the evaluation run?")
        sys.exit(1)

    # Production-level thresholds for 2026 MLE standards
    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.85,
        "context_precision": 0.80,
        "context_recall": 0.80,
    }

    failed = False
    logger.info("--- Ragas Evaluation Threshold Check ---")

    for metric, threshold in thresholds.items():
        val = results.get(metric, 0.0)
        if val < threshold:
            logger.error(f"❌ {metric}: {val:.3f} (Threshold: {threshold})")
            failed = True
        else:
            logger.info(f"✅ {metric}: {val:.3f} (Threshold: {threshold})")

    if failed:
        logger.error("Quality gate failed. Metrics are below production thresholds.")
        sys.exit(1)
    else:
        logger.info("Quality gate passed. Ready for review.")


if __name__ == "__main__":
    main()
