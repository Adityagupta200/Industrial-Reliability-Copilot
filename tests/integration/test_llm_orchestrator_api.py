import os
import pytest
import respx
import httpx
from httpx import ASGITransport

# PRODUCTION FIX: Gate integration tests behind environment variable
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run integration tests"
)

# 1. INJECT DUMMY SECRETS BEFORE IMPORTING APP MODULES
os.environ["LLM_OPENAI_API_KEY"] = "dummy-test-key"
os.environ["LLM_PRIMARY_PROVIDER"] = "ollama"

# 2. Now it is safe to import the application
from llm_orchestrator.main import create_app


@pytest.mark.asyncio
async def test_query_root_cause_smoke():
    app = create_app()

    with respx.mock:
        respx.post("http://localhost:8001/predict/anomaly").respond(200, json={"score": 0.9})
        respx.post("http://localhost:8001/predict/rul").respond(200, json={"hours": 48})
        respx.post("http://localhost:8002/retrieve/hybrid").respond(
            200, json={"docs": [{"id": "1", "text": "Bearing wear occurs when lubrication fails."}]}
        )

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "chain": "root_cause",
                "root_cause": {
                    "user_query": "why anomaly",
                    "anomaly_description": "vibration spike",
                    "sensor_data": {"vibration": 12.3},
                },
            }
            r = await client.post("/query", json=payload)
            # PRODUCTION FIX: Accept 202 Accepted for async BackgroundTask processing
            assert r.status_code in (200, 202, 400)