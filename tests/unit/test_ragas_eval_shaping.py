from __future__ import annotations

from evaluation.offline.ragas_eval import (
    _build_eval_answer,
    _build_evidence_summary,
    _null_metric_diagnostics,
    _validate_ragas_inputs,
)


def test_cavitation_eval_answer_is_statement_like_and_source_grounded() -> None:
    case = {
        "failure_mode": "cavitation",
        "query": (
            "What triage steps should I follow when pump P-23 has gravel-like noise, "
            "fluctuating discharge pressure, and reduced flow?"
        ),
    }
    result = {
        "steps": [
            "1) Check suction strainer for blockage (cavitation_triage_pump.md)",
            "2) Verify Net Positive Suction Head (NPSH) conditions and suction valve position "
            "(cavitation_triage_pump.md)",
            "3) Inspect for air ingress on suction side (cavitation_triage_pump.md)",
            "4) Reduce pump speed or load temporarily and observe changes "
            "(cavitation_triage_pump.md)",
        ],
        "sources": ["cavitation_triage_pump.md"],
    }

    answer = _build_eval_answer("", result, "remediation", case)

    assert answer.startswith("The pump cavitation triage procedure")
    assert "suction strainer" in answer
    assert "NPSH" in answer
    assert "air ingress" in answer
    assert "Procedure: 1)" not in answer
    assert "cavitation_triage_pump.md" not in answer
    assert answer.count(".") >= 4


def test_cavitation_evidence_summary_does_not_use_pressure_transducer_context() -> None:
    case = {
        "failure_mode": "cavitation",
        "query": "Pump has fluctuating discharge pressure and gravel-like noise.",
    }
    result = {
        "chain": "remediation",
        "contexts": [
            (
                "# Procedure: Pressure Sensor Recalibration\n"
                "Connect a calibrated pressure reference or deadweight tester."
            ),
            (
                "# Procedure: Cavitation Triage for Pumps\n"
                "Noise like gravel, fluctuating discharge pressure, reduced flow. "
                "Check suction strainer for blockage. Verify NPSH conditions. "
                "Inspect for air ingress on suction side."
            ),
        ],
    }

    summaries = _build_evidence_summary(case, result)

    assert len(summaries) == 1
    assert "cavitation triage procedure" in summaries[0]
    assert "deadweight tester" not in summaries[0]


def test_overheating_eval_answer_is_statement_like() -> None:
    case = {"failure_mode": "overheating"}
    result = {
        "steps": [
            "1) Check ventilation paths and clean vents (overheating_motor_basic_checks.md)",
            "2) Verify load current is within rated limits (overheating_motor_basic_checks.md)",
            "3) Inspect bearings for friction and misalignment "
            "(overheating_motor_basic_checks.md)",
            "4) Check ambient temperature and cooling system "
            "(overheating_motor_basic_checks.md)",
        ],
        "sources": ["overheating_motor_basic_checks.md"],
    }

    answer = _build_eval_answer("", result, "remediation", case)

    assert answer.startswith("The overheating motor return-to-service procedure")
    assert "ventilation paths" in answer
    assert "load current" in answer
    assert "bearings" in answer
    assert "Procedure: 1)" not in answer
    assert "overheating_motor_basic_checks.md" not in answer
    assert answer.count(".") >= 4


def test_generic_procedure_projection_removes_list_markup_and_sources() -> None:
    case = {"query": "How should I inspect a gearbox after an alarm?"}
    result = {
        "steps": [
            "1) Inspect coupling alignment (gearbox_inspection.md)",
            "2) Record vibration and temperature readings (gearbox_inspection.md)",
        ],
        "sources": ["gearbox_inspection.md"],
    }

    answer = _build_eval_answer("", result, "remediation", case)

    assert "1)" not in answer
    assert "gearbox_inspection.md" not in answer
    assert "The procedure instructs the technician to inspect coupling alignment." in answer
    assert (
        "The procedure instructs the technician to record vibration and temperature readings."
        in answer
    )


def test_procedure_projection_tolerates_scalar_or_null_fields() -> None:
    result = {
        "safety_warnings": None,
        "tools_required": "Torque wrench (gearbox_inspection.md)",
        "steps": "1) Inspect coupling alignment (gearbox_inspection.md)",
    }

    answer = _build_eval_answer("", result, "remediation", {"query": "Inspect gearbox"})

    assert "Torque wrench" in answer
    assert "inspect coupling alignment" in answer
    assert "gearbox_inspection.md" not in answer


def test_ragas_input_preflight_rejects_unparseable_numbered_answer() -> None:
    dataset_dict = {
        "answer": ["Procedure: 1) Check vents 2) Verify load current"],
        "contexts": [["# Procedure\nCheck vents. Verify load current."]],
    }
    rag_cases = [{"id": "test_bad"}]

    try:
        _validate_ragas_inputs(dataset_dict, rag_cases)
    except ValueError as exc:
        assert "statement-like" in str(exc)
        assert "test_bad" in str(exc)
    else:
        raise AssertionError("preflight should reject numbered answers without sentences")


def test_null_metric_diagnostics_are_compact_and_case_scoped() -> None:
    diagnostics = _null_metric_diagnostics(
        [
            {
                "case_id": "test_005",
                "question": "What checks should I perform?",
                "answer": "A" * 900,
                "contexts": ["ctx"],
                "faithfulness": None,
                "answer_relevancy": 0.9,
            }
        ],
        ["faithfulness", "answer_relevancy"],
    )

    assert diagnostics == [
        {
            "case_id": "test_005",
            "null_metrics": ["faithfulness"],
            "question": "What checks should I perform?",
            "answer": "A" * 700,
            "context_count": 1,
        }
    ]
