from pathlib import Path

from rag_service.api.retrieve import (
    _load_direct_procedure_docs,
    _rank_direct_procedures,
)
from rag_service.retrieval.types import RetrievalFilters


def test_direct_procedure_loader_infers_equipment_id(tmp_path: Path) -> None:
    procedure = tmp_path / "bearing_replacement_pump_P-23.md"
    procedure.write_text(
        "# Procedure: Bearing Replacement for Pump P-23\n\n"
        "High vibration with stable pressure and flow indicates bearing wear.",
        encoding="utf-8",
    )

    docs = _load_direct_procedure_docs(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source_file"] == "bearing_replacement_pump_P-23.md"
    assert docs[0].metadata["equipment_id"] == "pump_P-23"
    assert docs[0].metadata["retrieval_mode"] == "direct_procedure"


def test_direct_procedure_ranking_prefers_exact_equipment(tmp_path: Path) -> None:
    (tmp_path / "bearing_replacement_pump_P-23.md").write_text(
        "# Pump P-23\nHigh vibration with stable pressure and flow indicates bearing wear.",
        encoding="utf-8",
    )
    (tmp_path / "cavitation_triage_pump.md").write_text(
        "# Pump Cavitation\nCheck suction strainer for fluctuating pressure.",
        encoding="utf-8",
    )
    (tmp_path / "overheating_motor_basic_checks.md").write_text(
        "# Motor M-41\nTemperature checks for overheating motors.",
        encoding="utf-8",
    )

    docs = _load_direct_procedure_docs(tmp_path)
    ranked = _rank_direct_procedures(
        docs,
        query="Why did pump P-23 trigger a high vibration anomaly?",
        filters=RetrievalFilters(equipment_id="pump_P-23"),
        k=2,
    )

    assert [doc.metadata["source_file"] for doc in ranked] == ["bearing_replacement_pump_P-23.md"]
    assert ranked[0].score > 0.0
