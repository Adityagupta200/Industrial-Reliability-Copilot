def test_predict_rul_ok(client, sample_rul_request, api_prefix):
    r = client.post(f"{api_prefix}/predict/rul", json=sample_rul_request)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "predicted_rul" in data or "rul" in data
    assert "confidence" in data
    assert "timestamp" in data
