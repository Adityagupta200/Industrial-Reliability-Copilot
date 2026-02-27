def test_anomaly_response_contract_stable(client, sample_anom_request, api_prefix):
    r = client.post(f"{api_prefix}/predict/anomaly", json=sample_anom_request)
    assert r.status_code == 200, r.text

    data = r.json()
    # Only assert the contract you actually want to keep stable
    required = {"anomaly_score", "confidence", "timestamp"}
    missing = sorted(list(required - set(data.keys())))
    assert not missing, f"Missing keys: {missing}. Full response: {data}"
