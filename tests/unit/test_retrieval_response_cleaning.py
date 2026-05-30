from rag_service.api.retrieve import _to_document_response
from rag_service.retrieval.types import Document


def test_document_response_repairs_mojibake_and_does_not_mark_missing_dates_outdated() -> None:
    response = _to_document_response(
        Document(
            id="doc-1",
            text="Use 7\u00c3\u00b71.2 QBEP. \u00e2\u20ac\u00a2 Inspect bearings at 50\u00c2\u00baC.",
            metadata={"source_file": "pump_manual.pdf", "equipment_id": "pump_P-23"},
            score=0.9,
            source="semantic",
        )
    )

    assert "7/1.2 QBEP" in response.text
    assert "- Inspect bearings at 50 degrees C" in response.text
    assert not response.metadata.get("is_outdated")
    assert not response.text.startswith("(outdated)")
