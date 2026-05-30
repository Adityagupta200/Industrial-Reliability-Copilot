from __future__ import annotations

from typing import Any

import httpx
import pytest
from httpx import ASGITransport

import api_gateway.main as gateway_main


class FakeGatewayHTTPClient:
    def __init__(
        self,
        response: httpx.Response,
        *,
        get_response: httpx.Response | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.get_response = get_response or response
        self.get_error = get_error
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response

    async def get(self, url: str, timeout: float) -> httpx.Response:
        self.calls.append({"url": url, "timeout": timeout})
        if self.get_error:
            raise self.get_error
        return self.get_response


@pytest.mark.asyncio
async def test_gateway_feedback_route_forwards_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeGatewayHTTPClient(
        httpx.Response(200, json={"status": "success", "rating": "positive"})
    )
    monkeypatch.setattr(gateway_main, "http_client", fake_client)
    monkeypatch.setattr(gateway_main, "ORCHESTRATOR_URL", "http://orchestrator:8000")

    transport = ASGITransport(app=gateway_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/feedback",
            json={"query_id": "job-123", "score": 5},
            headers={"X-Trace-ID": "trace-123"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "success", "rating": "positive"}
    assert response.headers["x-trace-id"] == "trace-123"
    assert fake_client.calls == [
        {
            "url": "http://orchestrator:8000/feedback",
            "json": {"query_id": "job-123", "score": 5},
            "headers": {"X-Trace-ID": "trace-123", "X-Forwarded-For": "127.0.0.1"},
        }
    ]


@pytest.mark.asyncio
async def test_gateway_feedback_route_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeGatewayHTTPClient(httpx.Response(200, json={"status": "unexpected"}))
    monkeypatch.setattr(gateway_main, "http_client", fake_client)

    transport = ASGITransport(app=gateway_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/feedback",
            content="{",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON payload provided."}
    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_gateway_readiness_checks_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeGatewayHTTPClient(httpx.Response(200, json={"status": "ready"}))
    monkeypatch.setattr(gateway_main, "http_client", fake_client)
    monkeypatch.setattr(gateway_main, "ORCHESTRATOR_URL", "http://orchestrator:8000")
    monkeypatch.setattr(gateway_main, "READINESS_TIMEOUT_S", 0.25)

    transport = ASGITransport(app=gateway_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {"llm_orchestrator": {"status": "ready", "http_status": 200}},
    }
    assert fake_client.calls == [{"url": "http://orchestrator:8000/health/ready", "timeout": 0.25}]


@pytest.mark.asyncio
async def test_gateway_readiness_degrades_when_orchestrator_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeGatewayHTTPClient(httpx.Response(503, json={"status": "degraded"}))
    monkeypatch.setattr(gateway_main, "http_client", fake_client)
    monkeypatch.setattr(gateway_main, "ORCHESTRATOR_URL", "http://orchestrator:8000")

    transport = ASGITransport(app=gateway_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "dependencies": {"llm_orchestrator": {"status": "degraded", "http_status": 503}},
    }


@pytest.mark.asyncio
async def test_gateway_readiness_degrades_on_orchestrator_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "http://orchestrator:8000/health/ready")
    fake_client = FakeGatewayHTTPClient(
        httpx.Response(200),
        get_error=httpx.ConnectError("connection refused", request=request),
    )
    monkeypatch.setattr(gateway_main, "http_client", fake_client)
    monkeypatch.setattr(gateway_main, "ORCHESTRATOR_URL", "http://orchestrator:8000")

    transport = ASGITransport(app=gateway_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "dependencies": {"llm_orchestrator": {"status": "unreachable"}},
    }
