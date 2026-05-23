import pytest


@pytest.fixture(autouse=True)
def mock_rul_dependencies(monkeypatch):
    from anomaly_service.core.model_loader import LoadedModels

    dummy_models = LoadedModels(
        anomaly_model="dummy", anomaly_version="1.0", rul_model="dummy", rul_version="1.0"
    )
    monkeypatch.setattr("anomaly_service.main.MODELS", dummy_models)
    monkeypatch.setattr("anomaly_service.main.RUL_SCHEMA", {"dummy": "schema"})

    # Mock inference to return synthetic RUL output
    monkeypatch.setattr("anomaly_service.main.preprocess_rul", lambda vals, sch: "x")
    monkeypatch.setattr("anomaly_service.main.rul_infer", lambda m, x: (120.5, 0.90))


def test_predict_rul_ok(client, sample_rul_request):
    r = client.post("/predict/rul", json=sample_rul_request)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "predicted_rul" in data or "rul" in data
    assert "confidence" in data
    assert "timestamp" in data
