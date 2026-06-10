import json
import sys
import math

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def main():
    print("--- RAG Quality Gate ---")
    try:
        with open("data/evaluation_results/summary.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ Error: summary.json not found. Pipeline failed to run Ragas.")
        sys.exit(1)

    aggregated_scores = {}
    if isinstance(results, list):
        print("ℹ️ Detected row-level evaluation records. Computing aggregates...")
        sums = {k: 0.0 for k in THRESHOLDS.keys()}
        counts = {k: 0 for k in THRESHOLDS.keys()}

        for row in results:
            for metric, score in row.items():
                # Safely capture scores, ignoring NaNs and Nulls
                if metric in sums and isinstance(score, (int, float)) and not math.isnan(score):
                    sums[metric] += float(score)
                    counts[metric] += 1

        for metric in THRESHOLDS.keys():
            # If all evaluations for a metric failed (count = 0), assign it a 0.0 to safely fail the gate
            aggregated_scores[metric] = sums[metric] / counts[metric] if counts[metric] > 0 else 0.0
    elif isinstance(results, dict):
        aggregated_scores = results
    else:
        print("❌ Error: Unrecognized format in summary.json")
        sys.exit(1)

    failed = False
    print("\nEvaluating against thresholds:")

    for metric, threshold in THRESHOLDS.items():
        score = aggregated_scores.get(metric, 0.0)
        if score < threshold:
            print(f"❌ {metric.ljust(20)}: {score:.3f} (Required: {threshold:.3f})")
            failed = True
        else:
            print(f"✅ {metric.ljust(20)}: {score:.3f} (Required: {threshold:.3f})")

    if failed:
        print(
            "\n🚨 Quality Gate FAILED. LLM performance has regressed or local model dropped metrics."
        )
        sys.exit(1)

    print("\n🚀 Quality Gate PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
