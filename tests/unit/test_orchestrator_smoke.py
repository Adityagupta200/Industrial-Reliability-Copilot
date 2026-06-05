import asyncio

import pytest
from fastapi.testclient import TestClient
from llm_orchestrator.main import app
from llm_orchestrator.main import _query_cache_key
from llm_orchestrator.main import _is_billable_llm_provider
from llm_orchestrator.main import _record_query_cache_hit
from llm_orchestrator.main import _strip_raw_context_from_job
from llm_orchestrator.main import _prepare_query_status_response
from llm_orchestrator.main import _cached_or_inflight_response
from llm_orchestrator.main import QUERY_CACHE
from llm_orchestrator.main import QUERY_INFLIGHT
from llm_orchestrator.schemas import (
    QueryRequest,
    QueryResponse,
    RootCauseRequest,
    RootCauseResponse,
)

# Create a test client for the Orchestrator
client = TestClient(app)


def test_orchestrator_liveness():
    """Verify the orchestrator liveness probe responds successfully."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_orchestrator_metrics_exposed():
    """Verify that Prometheus metrics are correctly exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "orchestrator_requests_total" in response.text
    assert "orchestrator_inference_path_total" in response.text
    assert "retrieved_context_count" in response.text


def test_query_cache_key_includes_sensor_payload():
    base = {
        "chain": "root_cause",
        "root_cause": RootCauseRequest(
            user_query="Why did pump P-23 trigger anomaly?",
            equipment_id="pump_P-23",
            anomaly_description="High vibration anomaly.",
            sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2},
        ),
    }
    changed_sensor = {
        "chain": "root_cause",
        "root_cause": RootCauseRequest(
            user_query="Why did pump P-23 trigger anomaly?",
            equipment_id="pump_P-23",
            anomaly_description="High vibration anomaly.",
            sensor_data={"vibration_rms": 2.1, "pressure_bar": 5.2},
        ),
    }

    assert _query_cache_key(QueryRequest(**base)) != _query_cache_key(
        QueryRequest(**changed_sensor)
    )


def test_query_cache_key_ignores_bypass_cache_control():
    payload = {
        "chain": "root_cause",
        "root_cause": RootCauseRequest(
            user_query="Why did pump P-23 trigger anomaly?",
            equipment_id="pump_P-23",
            anomaly_description="High vibration anomaly.",
            sensor_data={"vibration_rms": 8.4, "pressure_bar": 5.2},
        ),
    }

    assert _query_cache_key(QueryRequest(**payload)) == _query_cache_key(
        QueryRequest(**payload, bypass_cache=True)
    )


def test_fast_path_is_not_counted_as_billable_llm_provider():
    assert _is_billable_llm_provider("openai")
    assert _is_billable_llm_provider("ollama")
    assert not _is_billable_llm_provider("rules+retrieval")
    assert not _is_billable_llm_provider("system")


def test_query_cache_hit_trace_payload_is_safe_and_informative():
    response = QueryResponse(
        chain="root_cause",
        result=RootCauseResponse(),
        raw_context="[DOC_1]\nHigh vibration evidence.",
        model_provider="rules+retrieval",
        model_name="root-cause-fast-path-v1",
    )

    payload = _record_query_cache_hit("a" * 64, response)

    assert payload == {
        "cache_key_prefix": "aaaaaaaaaaaa",
        "chain": "root_cause",
        "model_provider": "rules+retrieval",
        "model_name": "root-cause-fast-path-v1",
        "raw_context_chars": 32,
        "raw_context_available": True,
    }


def test_default_status_response_omits_raw_context_but_keeps_evidence_available():
    job = {
        "job_id": "trace-1",
        "status": "completed",
        "result": {
            "chain": "root_cause",
            "raw_context": "[DOC_1]\nHigh vibration evidence.",
            "result": {
                "hypotheses": [
                    {
                        "cause": "Bearing wear",
                        "confidence": 0.9,
                        "evidence": "DOC_1 supports bearing wear.",
                        "source": "bearing_replacement_pump_P-23.md",
                    }
                ]
            },
        },
    }

    sanitized = _strip_raw_context_from_job(job)

    assert sanitized["result"]["raw_context"] == "OMITTED_FROM_DEFAULT_RESPONSE"
    assert sanitized["result"]["raw_context_available"] is True
    assert sanitized["result"]["evidence_summary"] == {
        "raw_context_available": True,
        "raw_context_included": False,
        "context_chars": 32,
        "retrieved_doc_count": 1,
        "retrieved_doc_ids": ["DOC_1"],
        "source_files": ["bearing_replacement_pump_P-23.md"],
        "doc_id_to_source_file": {"DOC_1": "bearing_replacement_pump_P-23.md"},
    }
    assert job["result"]["raw_context"] == "[DOC_1]\nHigh vibration evidence."


def test_include_raw_context_response_adds_structured_evidence_summary():
    job = {
        "job_id": "trace-1",
        "status": "completed",
        "result": {
            "chain": "root_cause",
            "raw_context": "[DOC_1]\nHigh vibration evidence.",
            "result": {
                "hypotheses": [
                    {
                        "cause": "Bearing wear",
                        "confidence": 0.9,
                        "evidence": "DOC_1 supports bearing wear.",
                        "source": "bearing_replacement_pump_P-23.md",
                    }
                ]
            },
        },
    }

    response = _prepare_query_status_response(job, include_raw_context=True)

    assert response["result"]["raw_context"] == "[DOC_1]\nHigh vibration evidence."
    assert response["result"]["raw_context_available"] is True
    assert response["result"]["evidence_summary"]["raw_context_included"] is True
    assert response["result"]["evidence_summary"]["retrieved_doc_ids"] == ["DOC_1"]
    assert response["result"]["evidence_summary"]["source_files"] == [
        "bearing_replacement_pump_P-23.md"
    ]


@pytest.mark.asyncio
async def test_cached_or_inflight_response_coalesces_identical_work() -> None:
    QUERY_CACHE.clear()
    QUERY_INFLIGHT.clear()
    calls = 0

    async def execute() -> QueryResponse:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return QueryResponse(
            chain="root_cause",
            result=RootCauseResponse(),
            raw_context="[DOC_1]\nHigh vibration evidence.",
            model_provider="rules+retrieval",
            model_name="root-cause-fast-path-v1",
        )

    results = await asyncio.gather(
        *[_cached_or_inflight_response("same-cache-key", execute) for _ in range(5)]
    )

    responses = [item[0] for item in results]
    statuses = [item[1] for item in results]
    assert calls == 1
    assert statuses.count("miss") == 1
    assert statuses.count("inflight_join") == 4

    responses[0].trace_id = "mutated-response"
    cached_response, status = await _cached_or_inflight_response("same-cache-key", execute)

    assert status == "hit"
    assert cached_response.trace_id is None
    assert calls == 1
    QUERY_CACHE.clear()
    QUERY_INFLIGHT.clear()
