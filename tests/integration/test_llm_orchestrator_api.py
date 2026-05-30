from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
import respx
from httpx import ASGITransport

# Keep create_app importable in CI without real provider secrets.
os.environ.setdefault("LLM_PRIMARY_PROVIDER", "ollama")
os.environ.setdefault("LLM_FALLBACK_PROVIDER", "ollama")

import llm_orchestrator.main as orchestrator_main
from llm_orchestrator.guardrails.output_filters import OutputGuardrails
from llm_orchestrator.llm_client import LLMClient
from llm_orchestrator.main import create_app
from llm_orchestrator.providers.base import LLMResult


@pytest.mark.asyncio
async def test_query_root_cause_api_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    job_store: dict[str, dict[str, Any]] = {}

    async def create_job_state(job_id: str) -> None:
        job_store[job_id] = {"job_id": job_id, "status": "processing"}

    async def update_job_state(
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        job = job_store.setdefault(job_id, {"job_id": job_id})
        job.update({"status": status, "result": result, "error": error})

    async def get_job_state(job_id: str) -> dict[str, Any] | None:
        return job_store.get(job_id)

    async def log_interaction_async(*_: Any, **__: Any) -> None:
        return None

    async def fake_invoke(
        self: LLMClient,
        prompt: str,
        *,
        force_provider: str | None = None,
        json_mode: bool = False,
        is_judge: bool = False,
    ) -> LLMResult:
        payload = {
            "hypotheses": [
                {
                    "cause": "Bearing wear or insufficient lubrication",
                    "confidence": 0.86,
                    "evidence": (
                        "DOC_1 links high vibration with stable pressure and flow "
                        "to bearing wear and insufficient lubrication."
                    ),
                    "source": "DOC_1",
                }
            ]
        }
        return LLMResult(
            content=json.dumps(payload),
            model="ci-fake-root-cause",
            provider="ci-fake",
        )

    async def validate_output(*_: Any, **__: Any) -> tuple[bool, str]:
        return True, "grounded"

    monkeypatch.setattr(orchestrator_main, "create_job_state", create_job_state)
    monkeypatch.setattr(orchestrator_main, "update_job_state", update_job_state)
    monkeypatch.setattr(orchestrator_main, "get_job_state", get_job_state)
    monkeypatch.setattr(orchestrator_main, "log_interaction_async", log_interaction_async)
    monkeypatch.setattr(LLMClient, "invoke", fake_invoke)
    monkeypatch.setattr(OutputGuardrails, "validate_output", validate_output)
    monkeypatch.setenv("ROOT_CAUSE_FAST_PATH_ENABLED", "false")

    app = create_app()
    transport = ASGITransport(app=app)

    doc = {
        "id": "doc-1",
        "text": (
            "High vibration with stable pressure and flow is a common indicator of "
            "bearing wear, insufficient lubrication, contamination, or misalignment."
        ),
        "score": 0.98,
        "metadata": {
            "source_file": "bearing_replacement_pump_P-23.md",
            "equipment_id": "pump_P-23",
        },
    }

    with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:8001/predict/anomaly").respond(
            200,
            json={
                "is_anomaly": True,
                "confidence": 0.94,
                "description": "High vibration with stable pressure.",
            },
        )
        router.post("http://localhost:8001/predict/rul").respond(
            200,
            json={"remaining_useful_life_days": 12},
        )
        router.post("http://localhost:8002/retrieve/hybrid").respond(
            200,
            json={"documents": [doc]},
        )
        router.post("http://localhost:8002/retrieve/procedures").respond(
            200,
            json={"documents": [doc]},
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            live = await client.get("/health/live")
            assert live.status_code == 200
            assert live.json() == {"status": "alive"}

            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "orchestrator_requests_total" in metrics.text

            payload = {
                "chain": "root_cause",
                "root_cause": {
                    "user_query": "Why did pump P-23 trigger anomaly at 03:41?",
                    "anomaly_description": (
                        "Pump P-23 triggered a high-vibration anomaly with no "
                        "corresponding pressure drop."
                    ),
                    "equipment_id": "pump_P-23",
                    "sensor_data": {
                        "vibration_rms": 8.4,
                        "pressure_bar": 5.2,
                        "flow_rate_lpm": 176.0,
                    },
                },
            }
            response = await client.post(
                "/query",
                json=payload,
                headers={"X-Trace-ID": "ci-root-cause-contract"},
            )
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            status = await client.get(f"/query/{job_id}")
            assert status.status_code == 200

    data = job_store["ci-root-cause-contract"]
    assert data["status"] == "completed"
    result = data["result"]
    assert result["chain"] == "root_cause"
    assert result["model_provider"] == "ci-fake"
    assert "High vibration with stable pressure and flow" in result["raw_context"]
    hypothesis = result["result"]["hypotheses"][0]
    assert hypothesis["source"] == "bearing_replacement_pump_P-23.md"
    assert "bearing wear" in hypothesis["cause"].lower()
