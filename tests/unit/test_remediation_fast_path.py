from llm_orchestrator.chains.remediation_chain import (
    _build_extractive_remediation,
    _is_unsafe_interlock_request,
    _select_relevant_remediation_docs,
)
from llm_orchestrator.schemas import RemediationRequest, RetrievedDoc


def test_extractive_remediation_preserves_safety_tools_steps_and_sources() -> None:
    doc = RetrievedDoc(
        id="doc-1",
        text="""# Procedure

## Safety
- Lockout/tagout before any work.

## Tools
- Torque wrench.

## Steps
1. Inspect the bearing housing.

## Verification
- Record results in the maintenance log.
""",
        source="bearing_replacement_pump_P-23.md",
        metadata={"source_file": "bearing_replacement_pump_P-23.md"},
    )

    response = _build_extractive_remediation(
        RemediationRequest(
            user_query="What return-to-service checks should follow bearing replacement?",
            equipment_id="pump_P-23",
            failure_mode="bearing_failure",
        ),
        [doc],
    )

    assert response.safety_warnings
    assert response.tools_required
    assert any("Inspect the bearing" in step for step in response.steps)
    assert any("return equipment to service" in step for step in response.steps)
    assert response.sources == ["bearing_replacement_pump_P-23.md"]


def test_unsafe_interlock_request_detected() -> None:
    assert _is_unsafe_interlock_request("Can I bypass interlocks to restart faster?")


def test_remediation_selection_prefers_cavitation_sop_over_equipment_bearing_sop() -> None:
    bearing_doc = RetrievedDoc(
        id="bearing",
        text="""# Procedure: Bearing Replacement for Pump P-23

## Steps
1. Remove coupling guard and decouple drive.
2. Open bearing housing; inspect for scoring and contamination.
""",
        source="bearing_replacement_pump_P-23.md",
        metadata={
            "source_file": "bearing_replacement_pump_P-23.md",
            "equipment_id": "pump_P-23",
        },
    )
    cavitation_doc = RetrievedDoc(
        id="cavitation",
        text="""# Procedure: Cavitation Triage for Pumps

## Symptoms
- Noise like gravel, fluctuating discharge pressure, reduced flow.

## Steps
1. Check suction strainer for blockage.
2. Verify NPSH conditions and suction valve position.
3. Inspect for air ingress on suction side.
""",
        source="cavitation_triage_pump.md",
        metadata={"source_file": "cavitation_triage_pump.md"},
    )

    selected = _select_relevant_remediation_docs(
        RemediationRequest(
            user_query=(
                "What triage steps should I follow when pump P-23 has gravel-like noise, "
                "fluctuating discharge pressure, and reduced flow?"
            ),
            equipment_id="pump_P-23",
            failure_mode="cavitation",
        ),
        [bearing_doc, cavitation_doc],
    )
    response = _build_extractive_remediation(
        RemediationRequest(
            user_query="What triage steps should I follow for pump cavitation?",
            equipment_id="pump_P-23",
            failure_mode="cavitation",
        ),
        selected,
    )

    assert [doc.id for doc in selected] == ["cavitation"]
    assert any("suction strainer" in step for step in response.steps)
    assert any("NPSH" in step for step in response.steps)
    assert any("air ingress" in step for step in response.steps)
    assert not any("bearing housing" in step for step in response.steps)
