from __future__ import annotations

from pathlib import Path

from scripts.phase11_container_scan import (
    SERVICE_SPECS,
    dockerized_trivy_db_preflight_command,
    dockerized_trivy_command,
    extract_json,
    host_trivy_db_preflight_command,
    image_name,
    resolve_scanner,
    safe_cache_segment,
    safe_report_name,
    trivy_db_flags,
    vulnerability_counts,
    write_scan_artifact,
)
from scripts.phase11_security_audit import trivy_filesystem_command


def test_container_scan_image_names_use_service_conventions() -> None:
    image = image_name(SERVICE_SPECS["llm_orchestrator"], prefix="irc", tag="phase11")

    assert image == "irc-llm-orchestrator:phase11"


def test_container_scan_auto_uses_docker_when_host_trivy_missing(monkeypatch) -> None:
    monkeypatch.setattr("scripts.phase11_container_scan.shutil.which", lambda _: None)

    assert resolve_scanner("auto") == "docker"


def test_container_scan_auto_uses_host_when_trivy_exists(monkeypatch) -> None:
    monkeypatch.setattr("scripts.phase11_container_scan.shutil.which", lambda _: "trivy")

    assert resolve_scanner("auto") == "host"


def test_container_scan_counts_high_and_critical_vulnerabilities() -> None:
    payload = {
        "Results": [
            {
                "Vulnerabilities": [
                    {"VulnerabilityID": "CVE-1", "Severity": "HIGH"},
                    {"VulnerabilityID": "CVE-2", "Severity": "CRITICAL"},
                ]
            }
        ]
    }

    total, severity_counts = vulnerability_counts(payload)

    assert total == 2
    assert severity_counts == {"HIGH": 1, "CRITICAL": 1}


def test_extract_json_tolerates_scanner_log_prefix_and_suffix() -> None:
    payload = extract_json(
        "2026-06-02T00:00:00Z INFO scanning image\n"
        '{"Results": [{"Vulnerabilities": []}]}\n'
        "summary written"
    )

    assert payload == {"Results": [{"Vulnerabilities": []}]}


def test_dockerized_trivy_command_scans_saved_image_tar(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PHASE11_TRIVY_CACHE_DIR", str(tmp_path / "cache"))
    tar_path = tmp_path / "image.tar"

    cmd = dockerized_trivy_command(
        tar_path,
        severity="CRITICAL,HIGH",
        ignore_unfixed=True,
        trivy_image="aquasec/trivy:test",
        trivy_timeout="30m",
        trivy_parallel=1,
        db_repositories=["ghcr.io/aquasecurity/trivy-db:2"],
    )

    assert cmd[0:2] == ["docker", "run"]
    assert "aquasec/trivy:test" in cmd
    assert "--timeout" in cmd
    assert "30m" in cmd
    assert "--parallel" in cmd
    assert "1" in cmd
    assert "--input" in cmd
    assert "/work/image.tar" in cmd
    assert "--ignore-unfixed" in cmd
    assert "--db-repository" in cmd
    assert "ghcr.io/aquasecurity/trivy-db:2" in cmd


def test_trivy_db_flags_allow_official_fallback_repositories() -> None:
    flags = trivy_db_flags(
        [
            "ghcr.io/aquasecurity/trivy-db:2",
            "public.ecr.aws/aquasecurity/trivy-db:2",
        ]
    )

    assert flags == [
        "--db-repository",
        "ghcr.io/aquasecurity/trivy-db:2",
        "--db-repository",
        "public.ecr.aws/aquasecurity/trivy-db:2",
    ]


def test_trivy_db_preflight_commands_download_db_only(tmp_path) -> None:
    host_cmd = host_trivy_db_preflight_command(
        trivy_timeout="10m",
        db_repositories=["ghcr.io/aquasecurity/trivy-db:2"],
    )
    docker_cmd = dockerized_trivy_db_preflight_command(
        trivy_image="aquasec/trivy:test",
        trivy_timeout="10m",
        db_repositories=["ghcr.io/aquasecurity/trivy-db:2"],
        cache_dir=tmp_path / "cache",
    )

    assert host_cmd[0:2] == ["trivy", "image"]
    assert docker_cmd[0:2] == ["docker", "run"]
    assert "--download-db-only" in host_cmd
    assert "--download-db-only" in docker_cmd
    assert "--db-repository" in host_cmd
    assert "--db-repository" in docker_cmd


def test_safe_report_name_removes_image_tag_separator() -> None:
    assert safe_report_name("irc-rag-service:phase11") == "irc-rag-service_phase11.json"


def test_safe_cache_segment_removes_path_separators() -> None:
    assert (
        safe_cache_segment("phase11/irc-rag-service:phase11") == "phase11_irc-rag-service_phase11"
    )


def test_write_scan_artifact_removes_stale_failure_log_when_json_exists(tmp_path) -> None:
    report_path = tmp_path / "irc-rag-service_phase11.json"
    stale_log = report_path.with_suffix(".log")
    stale_log.write_text("old cache lock failure", encoding="utf-8")

    artifact = write_scan_artifact(
        report_path=report_path,
        payload={"Results": []},
        output="{}",
    )

    assert artifact == report_path
    assert report_path.exists()
    assert not stale_log.exists()


def test_write_scan_artifact_removes_stale_json_when_only_log_exists(tmp_path) -> None:
    report_path = tmp_path / "irc-rag-service_phase11.json"
    report_path.write_text('{"old": true}', encoding="utf-8")

    artifact = write_scan_artifact(
        report_path=report_path,
        payload=None,
        output="fatal scanner failure",
    )

    assert artifact == report_path.with_suffix(".log")
    assert artifact.read_text(encoding="utf-8") == "fatal scanner failure"
    assert not report_path.exists()


def test_security_audit_trivy_fs_falls_back_to_docker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("scripts.phase11_security_audit.shutil.which", lambda _: None)
    monkeypatch.setenv("PHASE11_TRIVY_CACHE_DIR", str(tmp_path / "cache"))

    cmd = trivy_filesystem_command(Path("C:/repo"))

    assert cmd[0:2] == ["docker", "run"]
    assert "aquasec/trivy:latest" in cmd
    assert "fs" in cmd
    assert "/repo" in cmd
