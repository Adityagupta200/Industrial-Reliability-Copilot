from __future__ import annotations

import sys

from rag_service.ingestion.chunking import _get_token_encoding, chunk_text


def test_chunk_text_is_token_bounded_and_dependency_light(monkeypatch):
    monkeypatch.setattr("rag_service.ingestion.chunking.settings.chunk_size_tokens", 24)
    monkeypatch.setattr("rag_service.ingestion.chunking.settings.chunk_overlap_tokens", 4)
    monkeypatch.setattr("rag_service.ingestion.chunking.settings.max_context_chars_per_chunk", 6000)

    text = (
        "Bearing wear often appears as high vibration with stable pressure and flow. "
        "Insufficient lubrication can increase bearing temperature. "
        "Inspect grease condition, contamination, scoring, and shaft alignment."
    )

    chunks = chunk_text(
        text,
        source_id="procedure__bearing_replacement_pump_P-23",
        doc_type="procedure",
        extra_meta={"source_file": "bearing_replacement_pump_P-23.md"},
    )

    enc = _get_token_encoding()
    assert len(chunks) > 1
    assert all(len(enc.encode(chunk.text)) <= 24 for chunk in chunks)
    assert (
        chunks[0].chunk_id
        == chunk_text(
            text,
            source_id="procedure__bearing_replacement_pump_P-23",
            doc_type="procedure",
            extra_meta={"source_file": "bearing_replacement_pump_P-23.md"},
        )[0].chunk_id
    )
    assert chunks[0].metadata["source_id"] == "procedure__bearing_replacement_pump_P-23"
    assert "langchain_text_splitters" not in sys.modules
