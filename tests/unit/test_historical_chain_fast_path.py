from llm_orchestrator.chains.historical_chain import (
    _build_historical_response,
    _historical_sql,
)
from llm_orchestrator.schemas import HistoricalSearchRequest


def test_historical_sql_uses_read_only_bounded_query_for_injection_shaped_input() -> None:
    req = HistoricalSearchRequest(
        user_query="Show incidents for pump P-23; DROP TABLE incidents; --",
        equipment_id="pump_P-23",
        days_back=365,
        limit=10,
    )

    sql, params = _historical_sql(req, "incidents")

    assert sql.lstrip().upper().startswith("SELECT")
    assert "DROP TABLE" not in sql
    assert "LIMIT :limit" in sql
    assert params["equipment_id"] == "pump_P-23"


def test_historical_response_summarizes_incidents_without_llm() -> None:
    req = HistoricalSearchRequest(
        user_query="Compare resolution time for cavitation, overheating, and bearing failures.",
        days_back=365,
        limit=50,
    )
    rows = [
        {
            "equipment_id": "pump_P-23",
            "failure_mode": "cavitation",
            "severity": "high",
            "actions_taken": "Cleared suction strainer",
            "outcome": "resolved",
            "resolution_time_hours": 2.0,
        },
        {
            "equipment_id": "motor_M-12",
            "failure_mode": "overheating",
            "severity": "medium",
            "actions_taken": "Cleaned ventilation paths",
            "outcome": "resolved",
            "resolution_time_hours": 4.0,
        },
    ]

    response = _build_historical_response(req, rows)

    assert "cavitation" in response.summary
    assert "overheating" in response.summary
    assert response.key_stats["incident_count"] == 2
    assert response.key_stats["average_resolution_time_hours"] == 3.0
    assert all(item.source == "SQL" for item in response.evidence)
