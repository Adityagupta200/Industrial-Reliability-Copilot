from llm_orchestrator.chains.root_cause_chain import _stabilize_supported_bearing_hypothesis
from llm_orchestrator.schemas import Hypothesis, RootCauseRequest, RootCauseResponse


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
