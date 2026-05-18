import pytest
from fastapi.testclient import TestClient
from llm_orchestrator.main import app

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