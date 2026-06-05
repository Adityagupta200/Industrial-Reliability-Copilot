#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("data/phase11/reports/container_scan_report.json")
DEFAULT_REPORTS_DIR = Path("data/phase11/reports/container_scans")
DEFAULT_TRIVY_IMAGE = os.getenv("PHASE11_TRIVY_IMAGE", "aquasec/trivy:latest")
DEFAULT_TRIVY_TIMEOUT = os.getenv("PHASE11_TRIVY_TIMEOUT", "30m")
DEFAULT_TRIVY_DB_TIMEOUT = os.getenv("PHASE11_TRIVY_DB_TIMEOUT", "10m")
DEFAULT_TRIVY_DB_REPOSITORIES = tuple(
    repo.strip()
    for repo in os.getenv(
        "PHASE11_TRIVY_DB_REPOSITORIES",
        ",".join(
            [
                "ghcr.io/aquasecurity/trivy-db:2",
                "public.ecr.aws/aquasecurity/trivy-db:2",
                "mirror.gcr.io/aquasec/trivy-db:2",
            ]
        ),
    ).split(",")
    if repo.strip()
)


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    image_component: str
    dockerfile: Path


@dataclass
class ScanResult:
    service: str
    image: str
    passed: bool
    scanner: str
    build_returncode: int
    scan_returncode: int | None = None
    vulnerability_count: int | None = None
    severity_counts: dict[str, int] = field(default_factory=dict)
    scan_report: str | None = None
    detail: str = ""


SERVICE_SPECS: dict[str, ServiceSpec] = {
    "anomaly_service": ServiceSpec(
        name="anomaly_service",
        image_component="anomaly-service",
        dockerfile=Path("src/anomaly_service/Dockerfile"),
    ),
    "api_gateway": ServiceSpec(
        name="api_gateway",
        image_component="api-gateway",
        dockerfile=Path("src/api_gateway/Dockerfile"),
    ),
    "llm_orchestrator": ServiceSpec(
        name="llm_orchestrator",
        image_component="llm-orchestrator",
        dockerfile=Path("src/llm_orchestrator/Dockerfile"),
    ),
    "rag_service": ServiceSpec(
        name="rag_service",
        image_component="rag-service",
        dockerfile=Path("src/rag_service/Dockerfile"),
    ),
}


def run_command(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def progress(message: str) -> None:
    print(f"[phase11-container-scan] {message}", file=sys.stderr, flush=True)


def trivy_cache_dir() -> Path:
    configured = os.getenv("PHASE11_TRIVY_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "industrial-reliability-copilot" / "trivy-cache"


def safe_cache_segment(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe.strip("._-") or "default"


def trivy_db_flags(db_repositories: list[str]) -> list[str]:
    flags: list[str] = []
    for repository in db_repositories:
        flags.extend(["--db-repository", repository])
    return flags


def resolve_scanner(scanner: str) -> str:
    if scanner == "host":
        return "host"
    if scanner == "docker":
        return "docker"
    return "host" if shutil.which("trivy") else "docker"


def image_name(spec: ServiceSpec, *, prefix: str, tag: str) -> str:
    return f"{prefix}-{spec.image_component}:{tag}"


def safe_report_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "_") + ".json"


def extract_json(stdout: str, stderr: str = "") -> dict[str, Any] | None:
    text = "\n".join(part for part in [stdout, stderr] if part)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def vulnerability_counts(payload: dict[str, Any] | None) -> tuple[int | None, dict[str, int]]:
    if not payload:
        return None, {}

    severity_counts: dict[str, int] = {}
    total = 0
    for result in payload.get("Results") or []:
        for vuln in result.get("Vulnerabilities") or []:
            total += 1
            severity = str(vuln.get("Severity") or "UNKNOWN").upper()
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
    return total, severity_counts


def write_scan_artifact(
    *,
    report_path: Path,
    payload: dict[str, Any] | None,
    output: str,
) -> Path:
    log_path = report_path.with_suffix(".log")
    if payload:
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        log_path.unlink(missing_ok=True)
        return report_path

    report_path.unlink(missing_ok=True)
    log_path.write_text(output[-10000:], encoding="utf-8")
    return log_path


def scan_failure_detail(
    *,
    process: subprocess.CompletedProcess[str],
    payload: dict[str, Any] | None,
    vulnerability_count: int | None,
    severity_counts: dict[str, int],
) -> str:
    if payload and vulnerability_count:
        counts = ", ".join(
            f"{severity}={count}" for severity, count in sorted(severity_counts.items())
        )
        return f"Trivy found {vulnerability_count} actionable vulnerabilities ({counts})."
    if cache_lock_failure(process):
        return (
            "Trivy could not acquire its cache lock. The scanner retried with an isolated "
            "cache namespace; inspect the log artifact for the final failure output."
        )
    output = (process.stdout + process.stderr).strip()
    return output[-5000:] or f"Trivy exited with code {process.returncode} without output."


def build_image(repo: Path, spec: ServiceSpec, image: str) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["docker", "build", "-f", str(spec.dockerfile), "-t", image, "."],
        cwd=repo,
    )


def host_trivy_command(
    image: str,
    *,
    severity: str,
    ignore_unfixed: bool,
    trivy_timeout: str,
    trivy_parallel: int | None,
    db_repositories: list[str],
) -> list[str]:
    cmd = [
        "trivy",
        "image",
        *trivy_db_flags(db_repositories),
        "--timeout",
        trivy_timeout,
        "--severity",
        severity,
        "--exit-code",
        "1",
        "--format",
        "json",
        "--quiet",
        image,
    ]
    if trivy_parallel is not None:
        cmd[2:2] = ["--parallel", str(trivy_parallel)]
    if ignore_unfixed:
        cmd.insert(-1, "--ignore-unfixed")
    return cmd


def dockerized_trivy_command(
    tar_path: Path,
    *,
    severity: str,
    ignore_unfixed: bool,
    trivy_image: str,
    trivy_timeout: str,
    trivy_parallel: int | None,
    db_repositories: list[str],
    cache_dir: Path | None = None,
) -> list[str]:
    cache_dir = cache_dir or trivy_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{tar_path.parent.resolve()}:/work:ro",
        "-v",
        f"{cache_dir.resolve()}:/root/.cache/",
        trivy_image,
        "image",
        *trivy_db_flags(db_repositories),
        "--timeout",
        trivy_timeout,
        "--input",
        f"/work/{tar_path.name}",
        "--severity",
        severity,
        "--exit-code",
        "1",
        "--format",
        "json",
        "--quiet",
    ]
    if trivy_parallel is not None:
        image_index = cmd.index("image")
        cmd[image_index + 1 : image_index + 1] = ["--parallel", str(trivy_parallel)]
    if ignore_unfixed:
        cmd.append("--ignore-unfixed")
    return cmd


def host_trivy_db_preflight_command(
    *,
    trivy_timeout: str,
    db_repositories: list[str],
) -> list[str]:
    return [
        "trivy",
        "image",
        *trivy_db_flags(db_repositories),
        "--timeout",
        trivy_timeout,
        "--download-db-only",
    ]


def dockerized_trivy_db_preflight_command(
    *,
    trivy_image: str,
    trivy_timeout: str,
    db_repositories: list[str],
    cache_dir: Path,
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{cache_dir.resolve()}:/root/.cache/",
        trivy_image,
        "image",
        *trivy_db_flags(db_repositories),
        "--timeout",
        trivy_timeout,
        "--download-db-only",
    ]


def cache_lock_failure(process: subprocess.CompletedProcess[str]) -> bool:
    output = f"{process.stdout}\n{process.stderr}".lower()
    return "cache may be in use" in output or "unable to initialize fs cache" in output


def run_trivy_scan(
    repo: Path,
    image: str,
    *,
    scanner: str,
    severity: str,
    ignore_unfixed: bool,
    trivy_image: str,
    trivy_timeout: str,
    trivy_parallel: int | None,
    trivy_cache_namespace: str,
    db_repositories: list[str],
) -> subprocess.CompletedProcess[str]:
    if scanner == "host":
        return run_command(
            host_trivy_command(
                image,
                severity=severity,
                ignore_unfixed=ignore_unfixed,
                trivy_timeout=trivy_timeout,
                trivy_parallel=trivy_parallel,
                db_repositories=db_repositories,
            ),
            cwd=repo,
        )

    with tempfile.TemporaryDirectory(prefix="phase11-image-scan-") as tmp:
        tar_path = Path(tmp) / "image.tar"
        save = run_command(["docker", "save", "-o", str(tar_path), image], cwd=repo)
        if save.returncode != 0:
            return save
        cache_root = trivy_cache_dir()
        cache_dir = cache_root / safe_cache_segment(trivy_cache_namespace)
        scan = run_command(
            dockerized_trivy_command(
                tar_path,
                severity=severity,
                ignore_unfixed=ignore_unfixed,
                trivy_image=trivy_image,
                trivy_timeout=trivy_timeout,
                trivy_parallel=trivy_parallel,
                db_repositories=db_repositories,
                cache_dir=cache_dir,
            ),
            cwd=repo,
        )
        if scan.returncode == 0 or not cache_lock_failure(scan):
            return scan

        image_cache_name = safe_cache_segment(
            f"{trivy_cache_namespace}-{safe_report_name(image).removesuffix('.json')}"
        )
        retry_cache_name = safe_cache_segment(f"{image_cache_name}-retry-{os.getpid()}")
        return run_command(
            dockerized_trivy_command(
                tar_path,
                severity=severity,
                ignore_unfixed=ignore_unfixed,
                trivy_image=trivy_image,
                trivy_timeout=trivy_timeout,
                trivy_parallel=trivy_parallel,
                db_repositories=db_repositories,
                cache_dir=cache_root / retry_cache_name,
            ),
            cwd=repo,
        )


def run_trivy_db_preflight(
    repo: Path,
    *,
    scanner: str,
    trivy_image: str,
    trivy_timeout: str,
    trivy_cache_namespace: str,
    db_repositories: list[str],
) -> subprocess.CompletedProcess[str]:
    if scanner == "host":
        return run_command(
            host_trivy_db_preflight_command(
                trivy_timeout=trivy_timeout,
                db_repositories=db_repositories,
            ),
            cwd=repo,
        )

    return run_command(
        dockerized_trivy_db_preflight_command(
            trivy_image=trivy_image,
            trivy_timeout=trivy_timeout,
            db_repositories=db_repositories,
            cache_dir=trivy_cache_dir() / safe_cache_segment(trivy_cache_namespace),
        ),
        cwd=repo,
    )


def scan_service(
    repo: Path, spec: ServiceSpec, args: argparse.Namespace, scanner: str
) -> ScanResult:
    image = image_name(spec, prefix=args.image_prefix, tag=args.tag)
    progress(f"Scanning service={spec.name} image={image}")
    build_returncode = 0
    if not args.skip_build:
        progress(f"Building {image}")
        build = build_image(repo, spec, image)
        build_returncode = build.returncode
        if build.returncode != 0:
            progress(f"Build failed for {image}")
            return ScanResult(
                service=spec.name,
                image=image,
                passed=False,
                scanner=scanner,
                build_returncode=build.returncode,
                detail=(build.stdout + build.stderr)[-5000:],
            )

    progress(f"Running Trivy for {image}")
    scan = run_trivy_scan(
        repo,
        image,
        scanner=scanner,
        severity=args.severity,
        ignore_unfixed=not args.include_unfixed,
        trivy_image=args.trivy_image,
        trivy_timeout=args.trivy_timeout,
        trivy_parallel=args.trivy_parallel,
        trivy_cache_namespace=args.trivy_cache_namespace,
        db_repositories=args.db_repository,
    )
    payload = extract_json(scan.stdout, scan.stderr)
    vulnerability_count, severity_counts = vulnerability_counts(payload)

    report_path = args.reports_dir / safe_report_name(image)
    scan_artifact = write_scan_artifact(
        report_path=report_path,
        payload=payload,
        output=scan.stdout + scan.stderr,
    )

    result = ScanResult(
        service=spec.name,
        image=image,
        passed=scan.returncode == 0,
        scanner=scanner,
        build_returncode=build_returncode,
        scan_returncode=scan.returncode,
        vulnerability_count=vulnerability_count,
        severity_counts=severity_counts,
        scan_report=str(scan_artifact),
        detail=(
            "No actionable HIGH/CRITICAL vulnerabilities found."
            if scan.returncode == 0
            else scan_failure_detail(
                process=scan,
                payload=payload,
                vulnerability_count=vulnerability_count,
                severity_counts=severity_counts,
            )
        ),
    )
    progress(f"Finished {image}; passed={result.passed}; scan_report={result.scan_report}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and scan Phase 11 service images with Trivy. Uses host Trivy "
            "when installed, otherwise falls back to the official Trivy Docker image."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--service", action="append", choices=sorted(SERVICE_SPECS), default=[])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--scanner", choices=["auto", "host", "docker"], default="auto")
    parser.add_argument("--trivy-image", default=DEFAULT_TRIVY_IMAGE)
    parser.add_argument("--severity", default="CRITICAL,HIGH")
    parser.add_argument("--include-unfixed", action="store_true")
    parser.add_argument("--image-prefix", default="irc")
    parser.add_argument("--tag", default="phase11")
    parser.add_argument(
        "--db-repository",
        action="append",
        default=[],
        help=(
            "Trivy vulnerability DB repository. Can be repeated. Defaults to "
            "official GHCR, public ECR, and GCR mirror fallbacks."
        ),
    )
    parser.add_argument(
        "--trivy-timeout",
        default=DEFAULT_TRIVY_TIMEOUT,
        help=(
            "Trivy analysis timeout. Large ML images contain PyTorch shared "
            "objects that can exceed Trivy's short default timeout on laptops."
        ),
    )
    parser.add_argument(
        "--trivy-db-timeout",
        default=DEFAULT_TRIVY_DB_TIMEOUT,
        help="Fail-fast timeout for the one-time Trivy vulnerability DB preflight.",
    )
    parser.add_argument(
        "--trivy-parallel",
        type=int,
        default=None,
        help="Optional Trivy parallelism override. Use 1 on very memory-constrained machines.",
    )
    parser.add_argument(
        "--trivy-cache-namespace",
        default=os.getenv("PHASE11_TRIVY_CACHE_NAMESPACE", "").strip(),
        help=(
            "Namespace for Dockerized Trivy cache directories. Defaults to a "
            "unique value per invocation to avoid stale local cache locks."
        ),
    )
    args = parser.parse_args()
    if args.trivy_parallel is not None and args.trivy_parallel < 1:
        parser.error("--trivy-parallel must be a positive integer when provided")
    if not args.db_repository:
        args.db_repository = list(DEFAULT_TRIVY_DB_REPOSITORIES)
    if not args.trivy_cache_namespace:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        args.trivy_cache_namespace = f"phase11-{timestamp}-{os.getpid()}"
    return args


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    args.output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    args.reports_dir = (
        (repo / args.reports_dir).resolve()
        if not args.reports_dir.is_absolute()
        else args.reports_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    scanner = resolve_scanner(args.scanner)
    selected = args.service or list(SERVICE_SPECS)

    progress(
        "Preflighting Trivy vulnerability DB "
        f"with scanner={scanner}, timeout={args.trivy_db_timeout}, "
        f"repositories={','.join(args.db_repository)}"
    )
    db_preflight = run_trivy_db_preflight(
        repo,
        scanner=scanner,
        trivy_image=args.trivy_image,
        trivy_timeout=args.trivy_db_timeout,
        trivy_cache_namespace=args.trivy_cache_namespace,
        db_repositories=args.db_repository,
    )
    db_preflight_log = args.reports_dir / "trivy_db_preflight.log"
    db_preflight_log.write_text(
        (db_preflight.stdout + db_preflight.stderr)[-10000:], encoding="utf-8"
    )
    if db_preflight.returncode != 0:
        report = {
            "passed": False,
            "failed_images": [SERVICE_SPECS[name].image_component for name in selected],
            "scanner": scanner,
            "trivy_image": args.trivy_image if scanner == "docker" else None,
            "trivy_timeout": args.trivy_timeout,
            "trivy_db_timeout": args.trivy_db_timeout,
            "trivy_db_repositories": args.db_repository,
            "trivy_cache_namespace": args.trivy_cache_namespace if scanner == "docker" else None,
            "severity": args.severity,
            "ignore_unfixed": not args.include_unfixed,
            "skip_build": args.skip_build,
            "db_preflight": {
                "passed": False,
                "returncode": db_preflight.returncode,
                "log": str(db_preflight_log),
                "detail": (db_preflight.stdout + db_preflight.stderr).strip()[-2000:],
            },
            "results": [],
        }
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "passed": False,
                    "failed_images": report["failed_images"],
                    "scanner": scanner,
                    "report": str(args.output),
                    "failure_details": [
                        {
                            "service": "trivy_db_preflight",
                            "detail": report["db_preflight"]["detail"][:500],
                            "scan_report": str(db_preflight_log),
                        }
                    ],
                },
                indent=2,
            )
        )
        return 1

    results = [scan_service(repo, SERVICE_SPECS[name], args, scanner) for name in selected]
    failed = [result.image for result in results if not result.passed]
    report = {
        "passed": not failed,
        "failed_images": failed,
        "scanner": scanner,
        "trivy_image": args.trivy_image if scanner == "docker" else None,
        "trivy_timeout": args.trivy_timeout,
        "trivy_db_timeout": args.trivy_db_timeout,
        "trivy_db_repositories": args.db_repository,
        "trivy_parallel": args.trivy_parallel,
        "trivy_cache_namespace": args.trivy_cache_namespace if scanner == "docker" else None,
        "severity": args.severity,
        "ignore_unfixed": not args.include_unfixed,
        "skip_build": args.skip_build,
        "db_preflight": {
            "passed": True,
            "returncode": db_preflight.returncode,
            "log": str(db_preflight_log),
        },
        "results": [result.__dict__ for result in results],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failure_details = [
        {
            "image": result.image,
            "service": result.service,
            "scan_returncode": result.scan_returncode,
            "detail": result.detail[:500],
            "scan_report": result.scan_report,
        }
        for result in results
        if not result.passed
    ]
    summary = {
        "passed": report["passed"],
        "failed_images": failed,
        "scanner": scanner,
        "report": str(args.output),
    }
    if failure_details:
        summary["failure_details"] = failure_details
    print(json.dumps(summary, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
