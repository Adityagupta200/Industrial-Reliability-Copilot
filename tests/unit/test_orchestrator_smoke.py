from fastapi.testclient import TestClient
from llm_orchestrator.main import app
from llm_orchestrator.main import _query_cache_key
from llm_orchestrator.main import _is_billable_llm_provider
from llm_orchestrator.schemas import QueryRequest, RootCauseRequest

# Create a test client for the Orchestrator
client = TestClient(app)


def test_orchestrator_liveness():
    """Verify the orchestrator liveness probe responds successfully."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_orchestrator_metrics_exposed():
    """Verify that Prometheus metrics are correctly exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "orchestrator_requests_total" in response.text
    assert "orchestrator_inference_path_total" in response.text
    assert "retrieved_context_count" in response.text


def test_query_cache_key_includes_sensor_payload():
    base = {
        "chain": "root_cause",
        "root_cause": RootCauseRequest(
            user_query="Why did pump P-23 trigger anomaly?",
            equipment_id="pump_P-23",
            anomaly_description="High vibration anomaly.",
            sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2},
        ),
    }
    changed_sensor = {
        "chain": "root_cause",
        "root_cause": RootCauseRequest(
            user_query="Why did pump P-23 trigger anomaly?",
            equipment_id="pump_P-23",
            anomaly_description="High vibration anomaly.",
            sensor_data={"vibration_rms": 2.1, "pressure_bar": 5.2},
        ),
    }

    assert _query_cache_key(QueryRequest(**base)) != _query_cache_key(
        QueryRequest(**changed_sensor)
    )


def test_query_cache_key_ignores_bypass_cache_control():
    payload = {
        "chain": "root_cause",
        "root_cause": RootCauseRequest(
            user_query="Why did pump P-23 trigger anomaly?",
            equipment_id="pump_P-23",
            anomaly_description="High vibration anomaly.",
            sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2},
        ),
    }

    assert _query_cache_key(QueryRequest(**payload)) == _query_cache_key(
        QueryRequest(**payload, bypass_cache=True)
    )


def test_fast_path_is_not_counted_as_billable_llm_provider():
    assert _is_billable_llm_provider("openai")
    assert _is_billable_llm_provider("ollama")
    assert not _is_billable_llm_provider("rules+retrieval")
    assert not _is_billable_llm_provider("system")
