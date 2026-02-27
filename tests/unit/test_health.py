def test_health_ok(client):
    # adjust if your route is different; this is the Phase-1 requirement
    r = client.get("/health")
    assert r.status_code == 200, r.text


def test_metrics_exposed(client):
    # Phase-1 requirement: Prometheus metrics endpoint exists
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    # Prometheus exposition format is plain text
    assert "text/plain" in r.headers.get("content-type", "")
