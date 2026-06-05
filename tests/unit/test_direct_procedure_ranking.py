from rag_service.api.retrieve import _rank_direct_procedures
from rag_service.retrieval.types import Document, RetrievalFilters


def test_direct_procedure_ranking_prefers_fault_specific_generic_sop() -> None:
    bearing_doc = Document(
        id="procedure:bearing_replacement_pump_P-23",
        text="""# Procedure: Bearing Replacement for Pump P-23

## Steps
1. Remove coupling guard and decouple drive.
2. Open bearing housing; inspect for scoring and contamination.
""",
        source="keyword",
        metadata={
            "source_file": "bearing_replacement_pump_P-23.md",
            "equipment_id": "pump_P-23",
        },
    )
    cavitation_doc = Document(
        id="procedure:cavitation_triage_pump",
        text="""# Procedure: Cavitation Triage for Pumps

## Symptoms
- Noise like gravel, fluctuating discharge pressure, reduced flow.

## Steps
1. Check suction strainer for blockage.
2. Verify NPSH conditions and suction valve position.
3. Inspect for air ingress on suction side.
""",
        source="keyword",
        metadata={"source_file": "cavitation_triage_pump.md", "equipment_id": None},
    )

    ranked = _rank_direct_procedures(
        [bearing_doc, cavitation_doc],
        query=(
            "What triage steps should I follow when pump P-23 has gravel-like noise, "
            "fluctuating discharge pressure, and reduced flow? failure mode cavitation "
            "equipment pump_P-23"
        ),
        filters=RetrievalFilters(equipment_id="pump_P-23"),
        k=2,
    )

    assert [doc.id for doc in ranked] == [
        "procedure:cavitation_triage_pump",
        "procedure:bearing_replacement_pump_P-23",
    ]
