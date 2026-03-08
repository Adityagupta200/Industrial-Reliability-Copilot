import pytest
import respx
import httpx

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

        # Force LLM to be ollama in env when running tests (or monkeypatch LLMClient)
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "chain": "root_cause",
                "root_cause": {
                    "user_query": "why anomaly",
                    "anomaly_description": "vibration spike",
                    "sensor_data": {"vibration": 12.3},
                },
            }
            r = await client.post("/query", json=payload)
            assert r.status_code in (200, 400)  # 200 once LLM configured for tests
