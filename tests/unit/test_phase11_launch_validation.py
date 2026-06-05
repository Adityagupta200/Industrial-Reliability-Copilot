from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts.phase11_e2e_validation import answer_text, evaluate_case, load_cases
from scripts.phase11_failure_drill import parse_args as parse_failure_drill_args
from scripts.phase11_launch_gate import check_validation_cases
from scripts.phase11_load_test import load_threshold_failures, run_load
from scripts.phase11_security_audit import accepted_vulnerability, pip_audit_cache_dir
from api_gateway.main import _forward_headers, _metrics_endpoint as gateway_metrics_endpoint
from llm_orchestrator.main import (
    _metrics_endpoint as orchestrator_metrics_endpoint,
    _parse_rate_limit,
    _rate_limit_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_phase11_query_set_has_50_cases_across_all_chains() -> None:
    cases = load_cases(REPO_ROOT / "data" / "phase11" / "e2e_queries.json")
    chain_counts: dict[str, int] = {}
    for case in cases:
        chain = case["payload"]["chain"]
        chain_counts[chain] = chain_counts.get(chain, 0) + 1

    assert len(cases) == 50
    assert chain_counts["root_cause"] > 0
    assert chain_counts["remediation"] > 0
    assert chain_counts["historical"] > 0


def test_phase11_operator_scripts_have_python_shebangs() -> None:
    script_names = [
        "phase11_container_scan.py",
        "phase11_e2e_validation.py",
        "phase11_failure_drill.py",
        "phase11_launch_gate.py",
        "phase11_load_test.py",
        "phase11_security_audit.py",
    ]

    for script_name in script_names:
        first_line = (
            (REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8").splitlines()[0]
        )
        assert first_line == "#!/usr/bin/env python"


def test_phase11_e2e_case_evaluation_requires_grounded_context() -> None:
    case = {
        "id": "case-1",
        "purpose": "unit",
        "expect": {
            "answer_contains_any": ["bearing"],
            "forbidden_content": ["replace entire system"],
            "min_sources": 1,
            "max_latency_ms": 2000,
            "require_grounded_context": True,
        },
    }
    status_payload = {
        "status": "completed",
        "result": {
            "latency_ms": 120.0,
            "evidence_summary": {
                "raw_context_available": True,
                "retrieved_doc_count": 1,
                "source_files": ["bearing_replacement_pump_P-23.md"],
            },
            "result": {
                "hypotheses": [
                    {
                        "cause": "Bearing wear from insufficient lubrication.",
                        "evidence": "DOC_1 supports the high vibration pattern.",
                        "source": "bearing_replacement_pump_P-23.md",
                    }
                ]
            },
        },
    }

    result = evaluate_case(case, status_payload)

    assert result["passed"] is True
    assert result["source_count"] == 1


def test_phase11_e2e_case_evaluation_flags_forbidden_terms() -> None:
    case = {
        "id": "case-2",
        "purpose": "unit",
        "expect": {
            "answer_contains_any": ["bearing"],
            "forbidden_content": ["replace entire system"],
            "min_sources": 0,
        },
    }
    status_payload = {
        "status": "completed",
        "result": {
            "latency_ms": 120.0,
            "result": {
                "hypotheses": [
                    {
                        "cause": "Bearing wear; replace entire system.",
                        "evidence": "unsupported",
                        "source": "NONE",
                    }
                ]
            },
        },
    }

    result = evaluate_case(case, status_payload)

    assert result["passed"] is False
    assert any(
        check["name"] == "forbidden_content_absent" and check["passed"] is False
        for check in result["checks"]
    )


def test_phase11_answer_text_includes_remediation_safety_and_tools() -> None:
    text = answer_text(
        {
            "result": {
                "safety_warnings": ["Follow LOTO before work."],
                "tools_required": ["Torque wrench"],
                "steps": ["Inspect the bearing housing."],
            }
        }
    )

    assert "LOTO" in text
    assert "Torque wrench" in text
    assert "Inspect the bearing" in text


def test_launch_gate_static_validation_cases_check_passes() -> None:
    check = check_validation_cases(REPO_ROOT)

    assert check.passed is True
    assert check.evidence["case_count"] == 50


def test_phase11_load_gate_requires_target_qps() -> None:
    report = {
        "success_rate": 1.0,
        "observed_qps": 31.878,
        "latency_ms": {"wall_p95": 1200.0},
        "status_counts": {"completed": 1000},
    }
    args = SimpleNamespace(
        target_qps=50.0,
        min_observed_qps_ratio=0.95,
        min_success_rate=0.99,
        max_wall_p95_ms=2500.0,
        max_5xx=0,
    )

    failures = load_threshold_failures(report, args)

    assert failures == ["observed_qps=31.878 below min_observed_qps=47.500"]


@pytest.mark.asyncio
async def test_phase11_load_harness_submits_open_loop_at_target_rate() -> None:
    jobs: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/query":
            job_id = f"job-{len(jobs) + 1}"
            jobs[job_id] = {"status": "completed"}
            return httpx.Response(202, json={"job_id": job_id, "status": "processing"})

        if request.method == "GET" and request.url.path.startswith("/query/"):
            job_id = request.url.path.rsplit("/", 1)[-1]
            if job_id not in jobs:
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "result": {
                        "latency_ms": 10.0,
                        "result": {
                            "hypotheses": [
                                {
                                    "cause": "Bearing wear",
                                    "evidence": "Grounded evidence.",
                                    "source": "bearing_replacement_pump_P-23.md",
                                }
                            ]
                        },
                    },
                },
            )

        return httpx.Response(404, json={"detail": "unexpected path"})

    args = SimpleNamespace(
        base_url="http://testserver",
        bypass_cache=False,
        request_timeout_seconds=5.0,
        max_in_flight=50,
        poll_worker_count=20,
        target_qps=20.0,
        duration_seconds=0.5,
        client_count=5,
        client_ip_prefix="10.12.0",
        poll_timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        min_success_rate=0.99,
        min_observed_qps_ratio=0.9,
        max_wall_p95_ms=2500.0,
        max_5xx=0,
        transport=httpx.MockTransport(handler),
    )
    cases = [
        {
            "id": "phase11_unit",
            "purpose": "unit",
            "payload": {"chain": "root_cause", "root_cause": {"user_query": "Why?"}},
            "expect": {"min_sources": 0},
        }
    ]

    report = await run_load(args, cases)

    assert report["submitted"] == 10
    assert report["accepted"] == 10
    assert report["passed"] == 10
    assert report["observed_qps"] == 20.0
    assert report["passed_thresholds"] is True


def test_kubernetes_manifest_autoscales_api_gateway() -> None:
    manifest = (REPO_ROOT / "infra" / "kubernetes" / "03-microservices.yaml").read_text()

    assert "name: api-gateway-hpa" in manifest
    assert "name: api-gateway" in manifest
    assert "minReplicas: 3" in manifest
    assert "averageUtilization: 70" in manifest
    assert 'cpu: "1000m"' in manifest


def test_kubernetes_manifest_prescales_orchestrator_for_phase11_load() -> None:
    manifest = (REPO_ROOT / "infra" / "kubernetes" / "03-microservices.yaml").read_text()

    assert "name: orchestrator-hpa" in manifest
    assert "name: llm-orchestrator" in manifest
    assert "replicas: 4 # Phase 11 production baseline for sustained 50 QPS validation" in manifest
    assert "minReplicas: 4" in manifest


def test_terraform_validates_rds_password_before_apply() -> None:
    variables = (REPO_ROOT / "infra" / "terraform" / "variables.tf").read_text()

    assert 'variable "db_password"' in variables
    assert "nullable    = false" in variables
    assert "length(var.db_password) >= 8" in variables
    assert "length(var.db_password) <= 128" in variables
    assert 'regex("^[!-~]+$"' in variables
    assert 'regexall("[/\'\\"@]"' in variables


def test_rate_limit_key_prefers_forwarded_client_ip() -> None:
    request = SimpleNamespace(
        headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.5"},
        client=SimpleNamespace(host="127.0.0.1"),
    )

    assert _rate_limit_key(request) == "203.0.113.10"


def test_rate_limit_parser_supports_production_window_units() -> None:
    assert _parse_rate_limit("60/minute") == (60, 60)
    assert _parse_rate_limit("50 / second") == (50, 1)
    assert _parse_rate_limit("120/hour") == (120, 3600)
    assert _parse_rate_limit("bad-value") == (60, 60)


def test_failure_drill_default_attempts_exercise_current_rate_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["phase11_failure_drill.py"])

    args = parse_failure_drill_args()

    assert args.rate_limit_attempts >= 120


def test_pip_audit_cache_defaults_outside_repo(monkeypatch) -> None:
    monkeypatch.delenv("PHASE11_PIP_AUDIT_CACHE_DIR", raising=False)

    cache_dir = pip_audit_cache_dir(REPO_ROOT)

    assert not cache_dir.resolve().is_relative_to(REPO_ROOT)


def test_pip_audit_cache_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHASE11_PIP_AUDIT_CACHE_DIR", str(tmp_path / "pip-audit-cache"))

    assert pip_audit_cache_dir(REPO_ROOT) == tmp_path / "pip-audit-cache"


def test_vulnerability_acceptance_allows_only_matching_no_fix_advisory() -> None:
    acceptances = [
        {
            "id": "PYSEC-2026-139",
            "aliases": ["CVE-2026-4538"],
            "package": "torch",
            "versions": ["2.9.1"],
            "accepted_until": "2026-09-02",
            "reason": "No fixed release available.",
        }
    ]

    accepted = accepted_vulnerability(
        package="torch",
        version="2.9.1",
        vulnerability={"id": "PYSEC-2026-139", "aliases": [], "fix_versions": []},
        acceptances=acceptances,
        today=date(2026, 6, 2),
    )

    assert accepted is not None


def test_vulnerability_acceptance_rejects_fixable_or_expired_advisory() -> None:
    acceptances = [
        {
            "id": "PYSEC-2026-139",
            "package": "torch",
            "versions": ["2.9.1"],
            "accepted_until": "2026-06-01",
        }
    ]

    fixable = accepted_vulnerability(
        package="torch",
        version="2.9.1",
        vulnerability={"id": "PYSEC-2026-139", "aliases": [], "fix_versions": ["2.9.2"]},
        acceptances=acceptances,
        today=date(2026, 6, 2),
    )
    expired = accepted_vulnerability(
        package="torch",
        version="2.9.1",
        vulnerability={"id": "PYSEC-2026-139", "aliases": [], "fix_versions": []},
        acceptances=acceptances,
        today=date(2026, 6, 2),
    )

    assert fixable is None
    assert expired is None


def test_gateway_forward_headers_appends_client_ip_to_existing_chain() -> None:
    request = SimpleNamespace(
        headers={"X-Forwarded-For": "203.0.113.10"},
        client=SimpleNamespace(host="127.0.0.1"),
    )

    headers = _forward_headers(request, "trace-123")

    assert headers["X-Trace-ID"] == "trace-123"
    assert headers["X-Forwarded-For"] == "203.0.113.10, 127.0.0.1"


def test_metrics_endpoint_uses_stable_query_template_for_job_ids() -> None:
    request = SimpleNamespace(
        scope={},
        url=SimpleNamespace(path="/query/phase11_001-load-000001-uuid"),
    )

    assert gateway_metrics_endpoint(request) == "/query/{job_id}"
    assert orchestrator_metrics_endpoint(request) == "/query/{job_id}"


def test_metrics_endpoint_prefers_fastapi_route_template() -> None:
    request = SimpleNamespace(
        scope={"route": SimpleNamespace(path="/query/{job_id}")},
        url=SimpleNamespace(path="/query/actual-job-id"),
    )

    assert gateway_metrics_endpoint(request) == "/query/{job_id}"
    assert orchestrator_metrics_endpoint(request) == "/query/{job_id}"
