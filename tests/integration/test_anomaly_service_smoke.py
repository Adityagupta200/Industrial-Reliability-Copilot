import os
import pytest
import requests

pytestmark = pytest.mark.integration


def _enabled() -> bool:
    # Prevent accidental network calls in CI unless explicitly enabled
    return os.getenv("RUN_INTEGRATION", "0") == "1"


def test_smoke_endpoints(base_url, api_prefix, sample_anom_request):
    if not _enabled():
        pytest.skip("Set RUN_INTEGRATION=1 to run integration tests")

    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, r.text

    r = requests.get(f"{base_url}/metrics", timeout=10)
    assert r.status_code == 200, r.text

    # PRODUCTION FIX: Removed api_prefix since the FastAPI app serves at the root level
    r = requests.post(f"{base_url}/predict/anomaly", json=sample_anom_request, timeout=20)
    assert r.status_code == 200, r.text

    r = requests.post(f"{base_url}/predict/rul", json=sample_anom_request, timeout=20)
    assert r.status_code == 200, r.text
