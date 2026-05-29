from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evaluation_gate import (
    RAGAS_THRESHOLDS,
    RESPONSE_CONTRACT_PASS_RATE_THRESHOLD,
    SAFETY_PASS_RATE_THRESHOLD,
    min_ragas_cases,
    min_total_cases,
)

DEFAULT_RESULTS_PATH = Path("ragas_results.json")
DEFAULT_GOLDEN_SET_PATH = Path("data/golden_test_set.json")
DEFAULT_OUTPUT_PATH = Path("data/evaluation_results/pr_comment.md")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(parsed) else parsed


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _format_float(value: Any) -> str:
    return f"{_score(value):.3f}"


def _truncate(text: str, limit: int = 96) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _coverage_by_category(cases: list[dict[str, Any]]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for case in cases:
        category = str(case.get("category", "uncategorized"))
        coverage[category] = coverage.get(category, 0) + 1
    return dict(sorted(coverage.items()))


def _case_lookup(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case.get("id")): case for case in cases if case.get("id")}


def _gate_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    ragas = report.get("ragas", {})
    for metric, threshold in RAGAS_THRESHOLDS.items():
        score = _score(ragas.get(metric))
        if score < threshold:
            failures.append(f"{metric} {_format_float(score)} < {threshold:.2f}")

    safety = report.get("safety", {})
    if safety and _score(safety.get("pass_rate")) < SAFETY_PASS_RATE_THRESHOLD:
        failures.append("safety pass rate below 1.00")

    contracts = report.get("response_contracts", {})
    if contracts and _score(contracts.get("pass_rate")) < RESPONSE_CONTRACT_PASS_RATE_THRESHOLD:
        failures.append("response contract pass rate below 1.00")

    case_count = report.get("case_count", {})
    ragas_cases = int(case_count.get("ragas", 0) or 0)
    total_cases = int(case_count.get("total", 0) or 0)
    if ragas_cases < min_ragas_cases():
        failures.append(f"Ragas case count {ragas_cases} < {min_ragas_cases()}")
    if total_cases < min_total_cases():
        failures.append(f"total case count {total_cases} < {min_total_cases()}")

    return failures


def _review_notes(report: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    case_count = report.get("case_count", {})
    ragas_cases = int(case_count.get("ragas", 0) or 0)
    total_cases = int(case_count.get("total", 0) or 0)
    if ragas_cases == min_ragas_cases() or total_cases == min_total_cases():
        notes.append(
            "Golden-set coverage is exactly at the CI minimum; expand it before claiming broad "
            "industrial-domain coverage."
        )

    closest: tuple[float, str, str, float, float] | None = None
    for row in report.get("case_metrics", []):
        case_id = str(row.get("case_id", "unknown"))
        for metric, threshold in RAGAS_THRESHOLDS.items():
            if metric not in row:
                continue
            score = _score(row.get(metric))
            margin = score - threshold
            candidate = (margin, case_id, metric, score, threshold)
            if closest is None or margin < closest[0]:
                closest = candidate

    if closest is not None and closest[0] < 0.05:
        margin, case_id, metric, score, threshold = closest
        relation = "below" if margin < 0 else "above"
        notes.append(
            f"Closest per-case quality margin: `{case_id}` `{metric}` is {score:.3f}, "
            f"{abs(margin):.3f} {relation} the {threshold:.2f} production threshold."
        )

    return notes


def render_markdown(report: dict[str, Any], golden_cases: list[dict[str, Any]]) -> str:
    failures = _gate_failures(report)
    review_notes = _review_notes(report)
    gate_status = "PASS" if not failures else "FAIL"
    case_count = report.get("case_count", {})
    cases_by_id = _case_lookup(golden_cases)
    coverage = _coverage_by_category(golden_cases)

    lines = [
        f"## LLM Quality Gate: {gate_status}",
        "",
        "Production RAG evaluation for Industrial Reliability Copilot. "
        "This PR gate combines Ragas quality metrics, deterministic response contracts, "
        "and adversarial safety checks.",
        "",
        "### Aggregate Metrics",
        "",
        "| Metric | Score | Threshold | Status |",
        "| --- | ---: | ---: | --- |",
    ]

    ragas = report.get("ragas", {})
    for metric, threshold in RAGAS_THRESHOLDS.items():
        score = _score(ragas.get(metric))
        lines.append(
            f"| `{metric}` | {score:.3f} | {threshold:.2f} | {_status(score >= threshold)} |"
        )

    safety = report.get("safety", {})
    if safety:
        pass_rate = _score(safety.get("pass_rate"))
        lines.append(
            "| `safety_pass_rate` | "
            f"{pass_rate:.3f} | {SAFETY_PASS_RATE_THRESHOLD:.2f} | "
            f"{_status(pass_rate >= SAFETY_PASS_RATE_THRESHOLD)} |"
        )

    contracts = report.get("response_contracts", {})
    if contracts:
        pass_rate = _score(contracts.get("pass_rate"))
        lines.append(
            "| `response_contract_pass_rate` | "
            f"{pass_rate:.3f} | {RESPONSE_CONTRACT_PASS_RATE_THRESHOLD:.2f} | "
            f"{_status(pass_rate >= RESPONSE_CONTRACT_PASS_RATE_THRESHOLD)} |"
        )

    lines.extend(
        [
            "",
            "### Coverage",
            "",
            f"- Ragas cases: `{case_count.get('ragas', 0)}` " f"(minimum `{min_ragas_cases()}`)",
            f"- Total golden cases: `{case_count.get('total', 0)}` "
            f"(minimum `{min_total_cases()}`)",
            "- Category mix: " + ", ".join(f"`{name}`={count}" for name, count in coverage.items()),
            "",
            "### Per-Case Ragas Results",
            "",
            "| Case | Category | Query | Faithfulness | Relevancy | Context Precision | Context Recall |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in report.get("case_metrics", []):
        case_id = str(row.get("case_id", "unknown"))
        case = cases_by_id.get(case_id, {})
        category = case.get("category", "unknown")
        query = row.get("question") or case.get("query", "")
        lines.append(
            "| "
            f"`{_escape_cell(case_id)}` | "
            f"{_escape_cell(category)} | "
            f"{_escape_cell(_truncate(query))} | "
            f"{_format_float(row.get('faithfulness'))} | "
            f"{_format_float(row.get('answer_relevancy'))} | "
            f"{_format_float(row.get('context_precision'))} | "
            f"{_format_float(row.get('context_recall'))} |"
        )

    lines.extend(
        [
            "",
            "### Deterministic Checks",
            "",
            "| Suite | Passed | Total | Pass Rate |",
            "| --- | ---: | ---: | ---: |",
            "| Safety | "
            f"{int(safety.get('passed', 0) or 0)} | {int(safety.get('total', 0) or 0)} | "
            f"{_format_float(safety.get('pass_rate'))} |",
            "| Response contracts | "
            f"{int(contracts.get('passed', 0) or 0)} | "
            f"{int(contracts.get('total', 0) or 0)} | "
            f"{_format_float(contracts.get('pass_rate'))} |",
            "",
        ]
    )

    if failures:
        lines.extend(
            [
                "### Required Action",
                "",
                "The quality gate failed for:",
                "",
                *[f"- {failure}" for failure in failures],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Interpretation",
                "",
                "The production CI gate passed. Review the full JSON/CSV artifacts for "
                "trace-level debugging and keep expanding the golden set as new failure "
                "modes or guardrail incidents are discovered.",
                "",
            ]
        )
        if review_notes:
            lines.extend(
                [
                    "### Review Notes",
                    "",
                    *[f"- {note}" for note in review_notes],
                    "",
                ]
            )

    lines.append(
        "*Full artifacts: `ragas_results.json`, `data/evaluation_results/latest_run.csv`, "
        "and `data/evaluation_results/evaluation_report.json`.*"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown Ragas quality-gate report.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = _load_json(args.results)
    golden_cases = _load_json(args.golden_set)
    markdown = render_markdown(report, golden_cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
