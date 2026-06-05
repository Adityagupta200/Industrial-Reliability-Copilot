#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("data/phase11/reports/security_audit_report.json")
DEFAULT_VULNERABILITY_ACCEPTANCE_POLICY = Path("security/vulnerability_acceptances.json")
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_secret_key": re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{20,}\b"),
    "aws_secret_access_key": re.compile(
        r"(?i)\baws_secret_access_key\b\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{30,}"
    ),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
SENSITIVE_WORDS = ("API_KEY", "SECRET", "PASSWORD", "POSTGRES_DSN", "INCIDENTS_DB_DSN")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


def run_command(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git_files(repo: Path) -> list[Path]:
    result = run_command(["git", "ls-files", "-z"], cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return [repo / item for item in result.stdout.split("\0") if item]


def scan_tracked_files(repo: Path) -> Check:
    findings: list[dict[str, str]] = []
    for path in git_files(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = str(path.relative_to(repo)).replace("\\", "/")
        for name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "file": rel,
                        "pattern": name,
                        "line": str(text[: match.start()].count("\n") + 1),
                    }
                )

    return Check(
        name="tracked_secret_patterns",
        passed=not findings,
        detail=(
            "No secret-shaped values found in tracked files."
            if not findings
            else f"Found {len(findings)} secret-shaped tracked values."
        ),
        evidence={"findings": findings[:50]},
    )


def check_git_history_markers(repo: Path) -> Check:
    markers = ["sk-", "AWS_SECRET_ACCESS_KEY", "BEGIN PRIVATE KEY"]
    marker_hits: dict[str, list[str]] = {}
    for marker in markers:
        result = run_command(["git", "log", "--all", "--oneline", "-S", marker], cwd=repo)
        if result.returncode == 0 and result.stdout.strip():
            marker_hits[marker] = result.stdout.strip().splitlines()[:20]

    return Check(
        name="git_history_secret_markers",
        passed=True,
        detail=(
            "Raw git-history marker search completed. Marker hits are informational; "
            "secret-shaped regex scan is the blocking check."
        ),
        evidence={"marker_hits": marker_hits},
    )


def check_kubernetes_secret_usage(repo: Path) -> Check:
    manifest_paths = sorted((repo / "infra" / "kubernetes").glob("*.yaml"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in manifest_paths)
    required = ["secretRef:", "name: copilot-secrets", "secretKeyRef:"]
    missing = [item for item in required if item not in text]
    literal_secret_patterns = [
        r"OPENAI_API_KEY:\s+[A-Za-z0-9_-]{8,}",
        r"POSTGRES_PASSWORD:\s+[A-Za-z0-9_!@#$%^&*()-]{8,}",
        r"postgresql\+[^:\s]+://[^*<][^@\s]+@",
    ]
    literal_hits = [
        pattern for pattern in literal_secret_patterns if re.search(pattern, text, re.IGNORECASE)
    ]
    return Check(
        name="kubernetes_secret_refs",
        passed=not missing and not literal_hits,
        detail=(
            "Kubernetes manifests reference Secrets instead of embedding runtime secret values."
            if not missing and not literal_hits
            else "Kubernetes secret wiring is incomplete or contains literal secret-like values."
        ),
        evidence={"missing": missing, "literal_hit_patterns": literal_hits},
    )


def check_sensitive_logging(repo: Path) -> Check:
    findings: list[dict[str, str]] = []
    for root in ["src", "scripts"]:
        for path in (repo / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                lowered = line.lower()
                if ("logger." not in lowered and "print(" not in lowered) or "mask_dsn" in line:
                    continue
                if any(word.lower() in lowered for word in SENSITIVE_WORDS):
                    findings.append(
                        {
                            "file": str(path.relative_to(repo)).replace("\\", "/"),
                            "line": str(line_no),
                            "snippet": line.strip()[:200],
                        }
                    )

    return Check(
        name="sensitive_values_not_logged",
        passed=not findings,
        detail=(
            "No obvious secret-bearing values are logged from src/scripts."
            if not findings
            else f"Found {len(findings)} possible sensitive logging statements."
        ),
        evidence={"findings": findings[:50]},
    )


def check_access_control_tests(repo: Path) -> Check:
    required_patterns = {
        "multi_tenant_filter_test": r"test_retrieval_guardrail_multi_tenancy",
        "pii_redaction_guardrail": r"redact_pii|PII",
        "rate_limiting": r"Limiter|RateLimitExceeded|ORCHESTRATOR_QUERY_RATE_LIMIT",
    }
    haystack = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in [
            repo / "tests" / "integration" / "test_guardrails.py",
            repo / "src" / "llm_orchestrator" / "guardrails" / "input_filters.py",
            repo / "src" / "llm_orchestrator" / "main.py",
        ]
        if path.exists()
    )
    missing = [
        name for name, pattern in required_patterns.items() if not re.search(pattern, haystack)
    ]
    return Check(
        name="access_control_coverage",
        passed=not missing,
        detail=(
            "Tenant filtering, PII redaction, and rate limiting have implementation/test coverage."
            if not missing
            else f"Missing access-control evidence: {', '.join(missing)}."
        ),
        evidence={"missing": missing},
    )


def run_optional_tool(name: str, cmd: list[str], repo: Path, *, required: bool) -> Check:
    result = run_command(cmd, cwd=repo)
    passed = result.returncode == 0 or not required
    evidence: dict[str, Any] = {
        "returncode": result.returncode,
        "output_tail": result.stdout[-5000:],
    }
    if name == "pip_audit":
        json_start = result.stdout.find("{")
        if json_start >= 0:
            try:
                payload = json.loads(result.stdout[json_start:])
                findings = []
                for dependency in payload.get("dependencies", []):
                    vulns = dependency.get("vulns") or []
                    if vulns:
                        findings.append(
                            {
                                "name": dependency.get("name"),
                                "version": dependency.get("version"),
                                "vulnerability_ids": [vuln.get("id") for vuln in vulns],
                            }
                        )
                evidence["vulnerability_count"] = sum(
                    len(item["vulnerability_ids"]) for item in findings
                )
                evidence["vulnerable_packages"] = findings
            except json.JSONDecodeError:
                pass
    return Check(
        name=name,
        passed=passed,
        detail=(
            f"{name} passed."
            if result.returncode == 0
            else f"{name} exited {result.returncode}; {'blocking' if required else 'recorded as warning'}."
        ),
        evidence=evidence,
    )


def extract_json_payload(output: str) -> dict[str, Any] | None:
    json_start = output.find("{")
    if json_start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(output[json_start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def load_vulnerability_acceptances(repo: Path, policy_path: Path) -> list[dict[str, Any]]:
    policy = policy_path if policy_path.is_absolute() else repo / policy_path
    if not policy.exists():
        return []
    payload = json.loads(policy.read_text(encoding="utf-8"))
    acceptances = payload.get("acceptances", [])
    if not isinstance(acceptances, list):
        raise ValueError(f"Invalid vulnerability acceptance policy: {policy}")
    return [item for item in acceptances if isinstance(item, dict)]


def vulnerability_ids(vulnerability: dict[str, Any]) -> set[str]:
    ids = {str(vulnerability.get("id") or "")}
    ids.update(str(alias) for alias in vulnerability.get("aliases") or [])
    return {item for item in ids if item}


def accepted_vulnerability(
    *,
    package: str,
    version: str,
    vulnerability: dict[str, Any],
    acceptances: list[dict[str, Any]],
    today: date,
) -> dict[str, Any] | None:
    if vulnerability.get("fix_versions"):
        return None

    vuln_ids = vulnerability_ids(vulnerability)
    for acceptance in acceptances:
        accepted_until = date.fromisoformat(str(acceptance.get("accepted_until", "")))
        if accepted_until < today:
            continue
        accepted_ids = {str(acceptance.get("id") or "")}
        accepted_ids.update(str(alias) for alias in acceptance.get("aliases") or [])
        accepted_ids = {item for item in accepted_ids if item}

        if str(acceptance.get("package", "")).lower() != package.lower():
            continue
        if version not in {str(item) for item in acceptance.get("versions") or []}:
            continue
        if not vuln_ids.intersection(accepted_ids):
            continue
        return acceptance
    return None


def run_pip_audit_tool(
    cmd: list[str],
    repo: Path,
    *,
    required: bool,
    acceptances: list[dict[str, Any]],
) -> Check:
    result = run_command(cmd, cwd=repo)
    evidence: dict[str, Any] = {
        "returncode": result.returncode,
        "output_tail": result.stdout[-5000:],
    }
    payload = extract_json_payload(result.stdout)
    actionable_findings: list[dict[str, Any]] = []
    accepted_findings: list[dict[str, Any]] = []
    skipped_dependencies: list[dict[str, Any]] = []

    if payload:
        for dependency in payload.get("dependencies", []):
            package = str(dependency.get("name") or "")
            version = str(dependency.get("version") or "")
            if dependency.get("skip_reason"):
                skipped_dependencies.append(
                    {
                        "name": package,
                        "version": version or None,
                        "reason": dependency.get("skip_reason"),
                    }
                )
                continue

            for vulnerability in dependency.get("vulns") or []:
                acceptance = accepted_vulnerability(
                    package=package,
                    version=version,
                    vulnerability=vulnerability,
                    acceptances=acceptances,
                    today=date.today(),
                )
                finding = {
                    "name": package,
                    "version": version,
                    "vulnerability_id": vulnerability.get("id"),
                    "aliases": vulnerability.get("aliases") or [],
                    "fix_versions": vulnerability.get("fix_versions") or [],
                }
                if acceptance:
                    finding["accepted_until"] = acceptance.get("accepted_until")
                    finding["acceptance_reason"] = acceptance.get("reason")
                    accepted_findings.append(finding)
                else:
                    actionable_findings.append(finding)

    evidence["vulnerability_count"] = len(actionable_findings) + len(accepted_findings)
    evidence["actionable_vulnerabilities"] = actionable_findings
    evidence["accepted_no_fix_vulnerabilities"] = accepted_findings
    evidence["skipped_dependencies"] = skipped_dependencies

    passed = result.returncode == 0 and not skipped_dependencies
    if payload and not actionable_findings and not skipped_dependencies:
        passed = True
    if not required:
        passed = True

    if passed and accepted_findings:
        detail = (
            "pip_audit reported only explicitly accepted no-fix vulnerabilities; "
            "review the acceptance policy before expiry."
        )
    elif passed:
        detail = "pip_audit passed."
    elif payload:
        detail = "pip_audit found actionable vulnerabilities or skipped dependencies."
    else:
        detail = (
            f"pip_audit exited {result.returncode}; "
            f"{'blocking' if required else 'recorded as warning'}."
        )

    return Check(
        name="pip_audit",
        passed=passed,
        detail=detail,
        evidence=evidence,
    )


def pip_audit_cache_dir(_repo: Path) -> Path:
    configured = os.getenv("PHASE11_PIP_AUDIT_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    return (
        Path(tempfile.gettempdir()) / "industrial-reliability-copilot" / "phase11-pip-audit-cache"
    )


def trivy_cache_dir() -> Path:
    configured = os.getenv("PHASE11_TRIVY_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "industrial-reliability-copilot" / "trivy-cache"


def trivy_filesystem_command(repo: Path) -> list[str]:
    host_trivy = shutil.which("trivy")
    if host_trivy:
        return [
            host_trivy,
            "fs",
            "--severity",
            "CRITICAL,HIGH",
            "--exit-code",
            "1",
            "--ignore-unfixed",
            "--quiet",
            ".",
        ]

    cache_dir = trivy_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    trivy_image = os.getenv("PHASE11_TRIVY_IMAGE", "aquasec/trivy:latest")
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo.resolve()}:/repo:ro",
        "-v",
        f"{cache_dir.resolve()}:/root/.cache/",
        trivy_image,
        "fs",
        "--severity",
        "CRITICAL,HIGH",
        "--exit-code",
        "1",
        "--ignore-unfixed",
        "--quiet",
        "/repo",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 11 security audit checks.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        help="Requirements file to audit with pip-audit. May be repeated.",
    )
    parser.add_argument("--run-pip-audit", action="store_true")
    parser.add_argument("--run-trivy-fs", action="store_true")
    parser.add_argument("--run-gitleaks", action="store_true")
    parser.add_argument("--strict-optional-tools", action="store_true")
    parser.add_argument(
        "--vulnerability-acceptance-policy",
        type=Path,
        default=DEFAULT_VULNERABILITY_ACCEPTANCE_POLICY,
        help=(
            "JSON policy for explicitly accepted no-fix vulnerabilities. "
            "Fixable vulnerabilities are never accepted by this policy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    checks: list[Check] = [
        scan_tracked_files(repo),
        check_git_history_markers(repo),
        check_kubernetes_secret_usage(repo),
        check_sensitive_logging(repo),
        check_access_control_tests(repo),
    ]

    if args.run_pip_audit:
        requirements = args.requirement or ["requirements.txt"]
        acceptances = load_vulnerability_acceptances(repo, args.vulnerability_acceptance_policy)
        pip_audit_cmd = [
            sys.executable,
            "-m",
            "pip_audit",
            "--cache-dir",
            str(pip_audit_cache_dir(repo)),
            "--progress-spinner",
            "off",
        ]
        for requirement in requirements:
            pip_audit_cmd.extend(["-r", requirement])
        pip_audit_cmd.extend(["--format", "json"])
        checks.append(
            run_pip_audit_tool(
                pip_audit_cmd,
                repo,
                required=args.strict_optional_tools,
                acceptances=acceptances,
            )
        )

    if args.run_trivy_fs:
        checks.append(
            run_optional_tool(
                "trivy_filesystem_scan",
                trivy_filesystem_command(repo),
                repo,
                required=args.strict_optional_tools,
            )
        )

    if args.run_gitleaks:
        checks.append(
            run_optional_tool(
                "gitleaks_history_scan",
                ["gitleaks", "detect", "--no-banner", "--redact", "--source", "."],
                repo,
                required=args.strict_optional_tools,
            )
        )

    report = {
        "checks": [check.__dict__ for check in checks],
        "passed": all(check.passed for check in checks),
        "failed_checks": [check.name for check in checks if not check.passed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"passed": report["passed"], "failed_checks": report["failed_checks"]}, indent=2)
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
