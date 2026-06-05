from rag_service.api import retrieve
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


def test_document_response_marks_explicit_old_last_updated_as_outdated() -> None:
    response = _to_document_response(
        Document(
            id="doc-old",
            text="Legacy pump seal procedure.",
            metadata={
                "source_file": "Industrial_pump_handbook_GB.pdf",
                "equipment_id": "pump_P-23",
                "last_updated": "2015-01-01T00:00:00+00:00",
            },
            score=0.8,
            source="semantic",
        )
    )

    assert response.metadata["is_outdated"] is True
    assert response.text.startswith("(outdated) Legacy pump seal procedure.")


def test_document_response_infers_pdf_metadata_for_existing_payloads(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieve,
        "_infer_document_source_metadata",
        lambda metadata: {"last_updated": "2016-01-01T00:00:00+00:00"},
    )

    response = _to_document_response(
        Document(
            id="doc-inferred",
            text="Legacy industrial pump handbook guidance.",
            metadata={
                "source_file": "Industrial_pump_handbook_GB.pdf",
                "equipment_id": "pump_P-23",
            },
            score=0.8,
            source="semantic",
        )
    )

    assert response.metadata["last_updated"] == "2016-01-01T00:00:00+00:00"
    assert response.metadata["is_outdated"] is True
    assert response.text.startswith("(outdated)")
