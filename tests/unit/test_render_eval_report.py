from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from render_eval_report import render_markdown  # noqa: E402


def test_render_markdown_summarizes_quality_gate() -> None:
    report = {
        "ragas": {
            "faithfulness": 0.93,
            "answer_relevancy": 0.96,
            "context_precision": 1.0,
            "context_recall": 1.0,
        },
        "safety": {"passed": 2, "total": 2, "pass_rate": 1.0},
        "response_contracts": {"passed": 4, "total": 4, "pass_rate": 1.0},
        "case_count": {"ragas": 4, "total": 6},
        "case_metrics": [
            {
                "case_id": "test_001",
                "question": "Why did pump P-23 trigger anomaly at 03:41?",
                "faithfulness": 1.0,
                "answer_relevancy": 0.94,
                "context_precision": 1.0,
                "context_recall": 1.0,
            }
        ],
    }
    golden_cases = [
        {"id": "test_001", "category": "multi-hop"},
        {"id": "test_002", "category": "happy-path"},
        {"id": "test_003", "category": "adversarial"},
        {"id": "test_004", "category": "retrieval-edge"},
        {"id": "test_005", "category": "safety-procedure"},
        {"id": "test_006", "category": "adversarial"},
    ]

    markdown = render_markdown(report, golden_cases)

    assert "## LLM Quality Gate: PASS" in markdown
    assert "| `faithfulness` | 0.930 | 0.85 | PASS |" in markdown
    assert "- Ragas cases: `4` (minimum `4`)" in markdown
    assert "`adversarial`=2" in markdown
    assert "| `test_001` | multi-hop | Why did pump P-23 trigger anomaly at 03:41?" in markdown


def test_render_markdown_flags_small_or_failing_gate() -> None:
    report = {
        "ragas": {
            "faithfulness": 0.72,
            "answer_relevancy": 0.90,
            "context_precision": 0.90,
            "context_recall": 0.90,
        },
        "safety": {"passed": 1, "total": 2, "pass_rate": 0.5},
        "response_contracts": {"passed": 2, "total": 2, "pass_rate": 1.0},
        "case_count": {"ragas": 2, "total": 3},
        "case_metrics": [],
    }

    markdown = render_markdown(report, [])

    assert "## LLM Quality Gate: FAIL" in markdown
    assert "faithfulness 0.720 < 0.85" in markdown
    assert "safety pass rate below 1.00" in markdown
    assert "Ragas case count 2 < 4" in markdown
