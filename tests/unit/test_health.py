def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    assert "text/plain" in r.headers.get("content-type", "")
