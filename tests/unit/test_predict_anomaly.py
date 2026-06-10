import copy
import pytest


@pytest.fixture(autouse=True)
def mock_anomaly_dependencies(monkeypatch):
    from anomaly_service.core.model_loader import LoadedModels

    dummy_models = LoadedModels(
        anomaly_model="dummy", anomaly_version="1.0", rul_model="dummy", rul_version="1.0"
    )
    monkeypatch.setattr("anomaly_service.main.MODELS", dummy_models)
    monkeypatch.setattr("anomaly_service.main.ANOM_SCHEMA", {"schemas": {"swat": {}}})
    monkeypatch.setattr("anomaly_service.main.ANOM_ARTIFACTS", {"dummy": "artifact"})

    # Mock inference to return synthetic scores
    monkeypatch.setattr(
        "anomaly_service.main.preprocess_anomaly", lambda s, vals, sch, art: ("x", 0)
    )
    monkeypatch.setattr("anomaly_service.main.anomaly_infer", lambda m, x, d: (0.85, 0.95))


def test_predict_anomaly_ok(client, sample_anom_request):
    r = client.post("/predict/anomaly", json=sample_anom_request)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "anomaly_score" in data
    assert "confidence" in data
    assert "timestamp" in data


def test_predict_anomaly_missing_feature_graceful_fallback(
    client, sample_anom_request, monkeypatch
):
    # Instruct our mock to simulate a validation failure
    def mock_preprocess_failure(*args, **kwargs):
        raise ValueError("Missing required features: ['FIT101']")

    monkeypatch.setattr("anomaly_service.main.preprocess_anomaly", mock_preprocess_failure)

    bad = copy.deepcopy(sample_anom_request)

    sv = bad.get("sensor_values", {})
    if not isinstance(sv, dict) or not sv:
        import pytest

        pytest.skip("sample request does not contain a non-empty sensor_values dict")

    k = next(iter(sv.keys()))
    del sv[k]

    r = client.post("/predict/anomaly", json=bad)

    #   Assert the graceful fallback behavior instead of expecting a 422 crash
    assert r.status_code == 200, r.text

    body = r.json()
    assert body.get("schema_id") == "generic"
    assert body.get("model_version") == "heuristic_rules_v1"
    assert "anomaly_score" in body
