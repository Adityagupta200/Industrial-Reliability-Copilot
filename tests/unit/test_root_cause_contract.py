import pytest

from llm_orchestrator.chains.root_cause_chain import (
    _build_supported_bearing_fast_path,
    _dedupe_hypotheses,
    _fast_path_guardrail_answer,
    _format_docs,
    _stabilize_supported_bearing_hypothesis,
    _validate_fast_path_response,
)
from llm_orchestrator.guardrails.output_filters import OutputGuardrails
from llm_orchestrator.schemas import Hypothesis, RetrievedDoc, RootCauseRequest, RootCauseResponse


def test_stabilize_supported_bearing_hypothesis_preserves_grounded_contract() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Mechanical imbalance",
                confidence=0.55,
                evidence="High vibration is discussed in the cited procedure.",
                source="bearing_replacement_pump_P-23.md",
            )
        ]
    )
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description=(
            "Pump P-23 triggered a high-vibration anomaly with no corresponding pressure drop."
        ),
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2, "flow_rate_lpm": 176.0},
    )
    docs_text = (
        "[DOC_1]\nHigh vibration with stable pressure and flow is a common indicator of "
        "bearing wear, insufficient lubrication, contamination, or misalignment."
    )

    stabilized = _stabilize_supported_bearing_hypothesis(
        response,
        request,
        docs_text,
        {"DOC_1": "bearing_replacement_pump_P-23.md"},
    )

    primary = stabilized.hypotheses[0]
    assert primary.cause == "Bearing wear or insufficient lubrication"
    assert "bearing wear" in primary.evidence.lower()
    assert "insufficient lubrication" in primary.evidence.lower()
    assert primary.source == "bearing_replacement_pump_P-23.md"
    assert primary.confidence >= 0.72


def test_stabilize_supported_bearing_hypothesis_does_not_invent_without_context() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Cavitation",
                confidence=0.6,
                evidence="Pressure instability is discussed in the cited procedure.",
                source="cavitation_triage_pump.md",
            )
        ]
    )
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description="Pump P-23 triggered a pressure anomaly.",
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 1.2, "pressure_bar": 0.5},
    )

    stabilized = _stabilize_supported_bearing_hypothesis(
        response,
        request,
        "[DOC_1]\nInspect suction blockage and cavitation.",
    )

    assert stabilized.hypotheses[0].cause == "Cavitation"


def test_dedupe_hypotheses_removes_repeated_bearing_lubrication_topic() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Bearing wear or insufficient lubrication",
                confidence=0.86,
                evidence="High vibration and stable pressure indicate bearing wear.",
                source="bearing_replacement_pump_P-23.md",
            ),
            Hypothesis(
                cause="Insufficient lubrication or contaminated grease",
                confidence=0.74,
                evidence="Grease starvation can cause bearing wear.",
                source="bearing_replacement_pump_P-23.md",
            ),
            Hypothesis(
                cause="Coupling or shaft misalignment",
                confidence=0.58,
                evidence="Alignment checks are required when vibration rises.",
                source="pump_manual2.pdf",
            ),
        ]
    )

    deduped = _dedupe_hypotheses(response)

    assert [h.cause for h in deduped.hypotheses] == [
        "Bearing wear or insufficient lubrication",
        "Coupling or shaft misalignment",
    ]


def test_format_docs_applies_context_budget_without_losing_source_mapping() -> None:
    docs_text = "High vibration with stable pressure indicates bearing wear. " * 20
    formatted, mapping = _format_docs(
        [
            RetrievedDoc(
                id="doc-1",
                text=docs_text,
                metadata={"source_file": "bearing_replacement_pump_P-23.md"},
                score=0.9,
                source="semantic",
            )
        ],
        max_chars_per_doc=160,
    )

    assert mapping == {"DOC_1": "bearing_replacement_pump_P-23.md"}
    assert "[DOC_1]" in formatted
    assert "[TRUNCATED:" in formatted
    assert len(formatted) < len(docs_text)


def test_supported_bearing_fast_path_returns_grounded_response(monkeypatch) -> None:
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "true")
    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description=(
            "Pump P-23 triggered a high-vibration anomaly with no corresponding pressure drop."
        ),
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2, "flow_rate_lpm": 176.0},
    )
    docs_text = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear, insufficient lubrication, contamination, or misalignment. "
        "Perform an alignment check after bearing replacement."
    )

    response = _build_supported_bearing_fast_path(
        request,
        docs_text,
        {"DOC_1": "bearing_replacement_pump_P-23.md"},
    )

    assert response is not None
    assert response.hypotheses[0].cause == "Bearing wear or insufficient lubrication"
    assert response.hypotheses[0].source == "bearing_replacement_pump_P-23.md"
    assert response.hypotheses[0].confidence == 0.9
    assert "vibration RMS 8.4" in response.hypotheses[0].evidence
    assert [hypothesis.cause for hypothesis in response.hypotheses] == [
        "Bearing wear or insufficient lubrication",
        "Pump or coupling misalignment",
    ]


def test_fast_path_guardrail_payload_preserves_doc_tag_grounding() -> None:
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Bearing wear or insufficient lubrication",
                confidence=0.9,
                evidence=(
                    "The bearing_replacement_pump_P-23.md states that high vibration "
                    "with stable pressure and flow supports bearing wear."
                ),
                source="bearing_replacement_pump_P-23.md",
            )
        ]
    )
    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )

    answer = _fast_path_guardrail_answer(
        response,
        {"DOC_1": "bearing_replacement_pump_P-23.md"},
    )

    assert '"source": "DOC_1"' in answer
    assert "DOC_1 supports" in answer
    assert OutputGuardrails._deterministic_groundedness(context, answer) == 1.0


@pytest.mark.asyncio
async def test_fast_path_validation_uses_deterministic_grounding_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTPUT_GUARDRAILS_LLM_JUDGE_MODE", "fallback")

    class FailingLLM:
        async def invoke(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Fast-path fallback validation should not call the LLM judge")

    request = RootCauseRequest(
        user_query="Why did pump P-23 trigger anomaly at 03:41?",
        anomaly_description=(
            "Pump P-23 triggered a high-vibration anomaly with no corresponding pressure drop."
        ),
        equipment_id="pump_P-23",
        sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2, "flow_rate_lpm": 176.0},
    )
    response = RootCauseResponse(
        hypotheses=[
            Hypothesis(
                cause="Bearing wear or insufficient lubrication",
                confidence=0.9,
                evidence=(
                    "The bearing_replacement_pump_P-23.md states that high vibration "
                    "with stable pressure and flow supports bearing wear or insufficient "
                    "lubrication. Sensor readings show vibration RMS 8.4 with stable "
                    "pressure and flow."
                ),
                source="bearing_replacement_pump_P-23.md",
            )
        ]
    )
    context = (
        "[DOC_1]\n"
        "High vibration with stable pressure and flow is a common indicator of "
        "bearing wear and insufficient lubrication."
    )

    await _validate_fast_path_response(
        llm_client=FailingLLM(),
        req=request,
        response=response,
        docs_text=context,
        doc_mapping={"DOC_1": "bearing_replacement_pump_P-23.md"},
        anomaly_model={"anomaly": {"description": "High vibration with stable pressure."}},
    )
