from scripts.phase2_generate_incidents import generate


def test_generate_incidents_schema() -> None:
    recs = generate(50)
    assert len(recs) == 50
    r = recs[0]
    assert r.timestamp
    assert r.equipment_id
    assert isinstance(r.sensor_data, dict)
    assert r.failure_mode
    assert r.severity in {"low", "medium", "high", "critical"}
    assert r.actions_taken
    assert r.outcome
    assert r.resolution_time_hours > 0
