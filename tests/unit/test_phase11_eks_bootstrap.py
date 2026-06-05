from __future__ import annotations

from pathlib import Path

from rag_service.bootstrap_phase11 import _validate_bootstrap
from rag_service.db.ingest_incidents import incident_record_id


def test_incident_record_id_is_stable_for_same_seed_record() -> None:
    record = {
        "timestamp": "2026-01-01T00:00:00Z",
        "equipment_id": "pump_P-23",
        "sensor_data": {"vibration_rms": 8.4},
        "failure_mode": "bearing_failure",
        "severity": "high",
        "actions_taken": "Inspected bearing and lubrication path.",
        "outcome": "Resolved",
        "resolution_time_hours": 3.5,
    }

    assert incident_record_id(record) == incident_record_id(dict(record))


def test_bootstrap_validation_requires_real_corpus_and_incidents(monkeypatch) -> None:
    monkeypatch.setenv("PHASE11_MIN_DOC_POINTS", "50")
    monkeypatch.setenv("PHASE11_MIN_PROCEDURE_POINTS", "5")
    monkeypatch.setenv("PHASE11_MIN_INCIDENT_ROWS", "100")

    failures = _validate_bootstrap(
        {
            "qdrant_counts": {"maintenance_docs": 0, "procedures": 0},
            "incident_rows_total": 0,
        }
    )

    assert len(failures) == 3


def test_bootstrap_validation_passes_with_expected_corpus(monkeypatch) -> None:
    monkeypatch.setenv("PHASE11_MIN_DOC_POINTS", "50")
    monkeypatch.setenv("PHASE11_MIN_PROCEDURE_POINTS", "5")
    monkeypatch.setenv("PHASE11_MIN_INCIDENT_ROWS", "100")

    failures = _validate_bootstrap(
        {
            "qdrant_counts": {"maintenance_docs": 250, "procedures": 5},
            "incident_rows_total": 150,
        }
    )

    assert failures == []


def test_eks_bootstrap_script_waits_before_attaching_logs() -> None:
    script = Path("scripts/phase11_eks_bootstrap.sh").read_text()

    assert "Waiting for pod/${bootstrap_pod} container to become loggable" in script
    assert "{range .items[*]}{.metadata.name}" in script
    assert ".items[0]" not in script
    assert "ContainerCreating" not in script
    assert "LOG_ATTACH_TIMEOUT_SECONDS" in script
    assert "ImagePullBackOff" in script
    assert "phase11_bootstrap_pod_describe_startup_failure.txt" in script


def test_eks_bootstrap_script_uses_pod_logs_for_failure_evidence() -> None:
    script = Path("scripts/phase11_eks_bootstrap.sh").read_text()

    assert 'BOOTSTRAP_TIMEOUT_SECONDS="${BOOTSTRAP_TIMEOUT_SECONDS:-7200}"' in script
    assert "activeDeadlineSeconds: ${BOOTSTRAP_TIMEOUT_SECONDS}" in script
    assert 'logs "pod/${bootstrap_pod}" --all-containers=true' in script
    assert "--pod-running-timeout=300s" in script
    assert "phase11_bootstrap_live_errors.log" in script


def test_eks_bootstrap_job_allows_full_cpu_ingestion_window() -> None:
    manifest = Path("infra/kubernetes/04-phase11-bootstrap-job.yaml").read_text()

    assert "activeDeadlineSeconds: 7200" in manifest
    assert 'memory: "3Gi"' in manifest
    assert 'cpu: "2000m"' in manifest
    assert 'memory: "6Gi"' in manifest
    assert 'cpu: "4000m"' in manifest


def test_eks_bootstrap_script_has_operator_preflights() -> None:
    script = Path("scripts/phase11_eks_bootstrap.sh").read_text()

    assert "require_command aws" in script
    assert "require_command docker" in script
    assert "require_command kubectl" in script
    assert 'kubectl get namespace "${NS}"' in script
    assert "trap cleanup EXIT" in script


def test_bootstrap_evidence_collector_handles_missing_pods() -> None:
    script = Path("scripts/phase11_collect_bootstrap_evidence.sh").read_text()

    assert "#!/usr/bin/env bash" in script
    assert "job-name=${JOB_NAME}" in script
    assert "{range .items[*]}{.metadata.name}" in script
    assert ".items[0]" not in script
    assert "No pods found for selector job-name=${JOB_NAME}" in script
    assert "Also checked controller UID labels when available." in script
    assert "exit 0" in script


def test_bootstrap_evidence_collector_captures_pod_logs_and_events() -> None:
    script = Path("scripts/phase11_collect_bootstrap_evidence.sh").read_text()

    assert 'describe job/"${JOB_NAME}"' in script
    assert "get events --sort-by=.lastTimestamp" in script
    assert "--field-selector" in script
    assert "phase11_bootstrap_status_${SUFFIX}.txt" in script
    assert "batch\\.kubernetes\\.io/controller-uid" in script
    assert "batch.kubernetes.io/controller-uid=${controller_uid}" in script
    assert 'logs "pod/${pod_name}" --all-containers=true' in script
    assert "--previous" in script
    assert "--pod-running-timeout=300s" in script


def test_rag_runtime_image_packages_direct_procedure_corpus() -> None:
    dockerfile = Path("src/rag_service/Dockerfile").read_text()

    assert "data/raw/procedures/" in dockerfile
    assert "/app/data/raw/procedures/" in dockerfile
