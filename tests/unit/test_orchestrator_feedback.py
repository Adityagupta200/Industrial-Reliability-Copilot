from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

os.environ.setdefault("LLM_PRIMARY_PROVIDER", "ollama")
os.environ.setdefault("LLM_FALLBACK_PROVIDER", "ollama")

import llm_orchestrator.main as orchestrator_main
from llm_orchestrator.main import create_app


class FakeCounter:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self._rating = ""

    def labels(self, *, rating: str) -> "FakeCounter":
        self._rating = rating
        return self

    def inc(self, amount: int = 1) -> None:
        self.events.append((self._rating, amount))


class FakeScalars:
    def __init__(self, log_entry: Any | None) -> None:
        self.log_entry = log_entry

    def first(self) -> Any | None:
        return self.log_entry


class FakeResult:
    def __init__(self, log_entry: Any | None) -> None:
        self.log_entry = log_entry

    def scalars(self) -> FakeScalars:
        return FakeScalars(self.log_entry)


class FakeAsyncSession:
    def __init__(self, log_entry: Any | None) -> None:
        self.log_entry = log_entry
        self.commits = 0
        self.execute_calls = 0

    async def __aenter__(self) -> "FakeAsyncSession":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def execute(self, _: Any) -> FakeResult:
        self.execute_calls += 1
        return FakeResult(self.log_entry)

    async def commit(self) -> None:
        self.commits += 1


class ExistingQueryLog:
    user_feedback_score: int | None = None


async def _allow_rate_limit(**_: Any) -> tuple[bool, int, int]:
    return True, 1, 60


@pytest.mark.asyncio
async def test_orchestrator_feedback_persists_then_updates_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_log = ExistingQueryLog()
    fake_db = FakeAsyncSession(query_log)
    fake_counter = FakeCounter()

    async def get_job_state(_: str) -> dict[str, Any] | None:
        return None

    monkeypatch.setattr(orchestrator_main, "AsyncSessionLocal", lambda: fake_db)
    monkeypatch.setattr(orchestrator_main, "USER_FEEDBACK", fake_counter)
    monkeypatch.setattr(orchestrator_main, "get_job_state", get_job_state)
    monkeypatch.setattr(orchestrator_main, "increment_rate_limit_bucket", _allow_rate_limit)

    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/feedback", json={"query_id": "job-123", "score": 5})

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Feedback recorded",
        "rating": "positive",
    }
    assert query_log.user_feedback_score == 5
    assert fake_db.commits == 1
    assert fake_counter.events == [("positive", 1)]


@pytest.mark.asyncio
async def test_orchestrator_feedback_does_not_increment_metric_for_processing_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession(log_entry=None)
    fake_counter = FakeCounter()

    async def get_job_state(_: str) -> dict[str, Any] | None:
        return {"job_id": "job-123", "status": "processing"}

    monkeypatch.setattr(orchestrator_main, "AsyncSessionLocal", lambda: fake_db)
    monkeypatch.setattr(orchestrator_main, "USER_FEEDBACK", fake_counter)
    monkeypatch.setattr(orchestrator_main, "get_job_state", get_job_state)
    monkeypatch.setattr(orchestrator_main, "increment_rate_limit_bucket", _allow_rate_limit)

    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/feedback", json={"query_id": "job-123", "score": 5})

    assert response.status_code == 409
    assert response.json() == {"detail": "Query is still processing; retry once it has completed."}
    assert fake_db.commits == 0
    assert fake_counter.events == []


@pytest.mark.asyncio
async def test_orchestrator_feedback_validates_score_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession(log_entry=None)
    monkeypatch.setattr(orchestrator_main, "AsyncSessionLocal", lambda: fake_db)

    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/feedback", json={"query_id": "job-123", "score": 6})

    assert response.status_code == 422
    assert fake_db.execute_calls == 0
