from anomaly_service.core.model_loader import LoadedModels


def test_readiness_allows_degraded_fallback(client, monkeypatch):
    monkeypatch.setattr("anomaly_service.main.MODELS", LoadedModels(None, None, None, None))
    monkeypatch.setattr("anomaly_service.main.ANOM_SCHEMA", None)
    monkeypatch.setattr("anomaly_service.main.ANOM_ARTIFACTS", None)
    monkeypatch.setattr("anomaly_service.main.RUL_SCHEMA", None)

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["fallback_active"] is True


def test_dependency_health_fails_when_models_are_degraded(client, monkeypatch):
    monkeypatch.setattr("anomaly_service.main.MODELS", LoadedModels(None, None, None, None))
    monkeypatch.setattr("anomaly_service.main.ANOM_SCHEMA", None)
    monkeypatch.setattr("anomaly_service.main.ANOM_ARTIFACTS", None)
    monkeypatch.setattr("anomaly_service.main.RUL_SCHEMA", None)

    response = client.get("/health/dependencies")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["fallback_active"] is True


def test_dependency_health_passes_when_models_are_loaded(client, monkeypatch):
    monkeypatch.setattr(
        "anomaly_service.main.MODELS",
        LoadedModels(
            anomaly_model="dummy-anomaly",
            anomaly_version="1.0",
            rul_model="dummy-rul",
            rul_version="1.0",
        ),
    )
    monkeypatch.setattr("anomaly_service.main.ANOM_SCHEMA", {"schemas": {"generic": {}}})
    monkeypatch.setattr("anomaly_service.main.ANOM_ARTIFACTS", {"scaler": "dummy"})
    monkeypatch.setattr("anomaly_service.main.RUL_SCHEMA", {"features": []})

    response = client.get("/health/dependencies")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["fallback_active"] is False
