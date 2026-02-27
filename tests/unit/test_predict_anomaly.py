import copy


def test_predict_anomaly_ok(client, sample_anom_request, api_prefix):
    r = client.post(f"{api_prefix}/predict/anomaly", json=sample_anom_request)
    assert r.status_code == 200, r.text

    data = r.json()
    # Don’t overfit to exact fields; assert core Phase-1 contract
    assert "anomaly_score" in data
    assert "confidence" in data
    assert "timestamp" in data


def test_predict_anomaly_missing_feature_returns_422(client, sample_anom_request, api_prefix):
    bad = copy.deepcopy(sample_anom_request)

    # remove one sensor key if present
    sv = bad.get("sensor_values", {})
    if not isinstance(sv, dict) or not sv:
        # If your request shape differs, skip with a clear reason.
        # (But based on your curl, sensor_values is a dict.)
        import pytest

        pytest.skip("sample request does not contain a non-empty sensor_values dict")

    k = next(iter(sv.keys()))
    del sv[k]

    r = client.post(f"{api_prefix}/predict/anomaly", json=bad)
    assert r.status_code == 422, r.text

    # Your service currently returns a string detail like:
    # {"detail":"Missing required features: ['FIT101']"}
    body = r.json()
    assert "detail" in body
