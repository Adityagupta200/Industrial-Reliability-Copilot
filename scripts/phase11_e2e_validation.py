#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CASES = Path("data/phase11/e2e_queries.json")
DEFAULT_REPORT = Path("data/phase11/reports/e2e_validation_report.json")


def load_cases(
    path: Path, *, load_safe_only: bool = False, case_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a JSON list")

    filtered: list[dict[str, Any]] = []
    for case in cases:
        if case_ids and case.get("id") not in case_ids:
            continue
        if load_safe_only and not case.get("load_safe", False):
            continue
        filtered.append(case)
    return filtered


def answer_text(result: dict[str, Any]) -> str:
    payload = result.get("result") or {}
    if "hypotheses" in payload:
        parts = []
        for item in payload.get("hypotheses") or []:
            parts.extend([str(item.get("cause", "")), str(item.get("evidence", ""))])
        return " ".join(parts)
    if "steps" in payload:
        parts: list[str] = []
        parts.extend(str(item) for item in payload.get("safety_warnings") or [])
        parts.extend(str(item) for item in payload.get("tools_required") or [])
        parts.extend(str(step) for step in payload.get("steps") or [])
        return " ".join(parts)
    if "summary" in payload:
        evidence = " ".join(str(item.get("claim", "")) for item in payload.get("evidence") or [])
        return f"{payload.get('summary', '')} {evidence}"
    return json.dumps(payload, sort_keys=True)


def source_values(result: dict[str, Any]) -> list[str]:
    payload = result.get("result") or {}
    sources: list[str] = []
    for item in payload.get("hypotheses") or []:
        source = str(item.get("source", "")).strip()
        if source and source.upper() != "NONE":
            sources.append(source)
    for source in payload.get("sources") or []:
        source = str(source).strip()
        if source and source.upper() != "NONE":
            sources.append(source)
    for item in payload.get("evidence") or []:
        source = str(item.get("source", "")).strip()
        if source and source.upper() != "NONE":
            sources.append(source)

    evidence_summary = result.get("evidence_summary") or {}
    for source in evidence_summary.get("source_files") or []:
        source = str(source).strip()
        if source and source.upper() != "NONE":
            sources.append(source)

    return sorted(set(sources))


def _contains_any(text: str, expected: list[str]) -> bool:
    if not expected:
        return True
    lowered = text.lower()
    return any(term.lower() in lowered for term in expected)


def evaluate_case(case: dict[str, Any], status_payload: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expect = case.get("expect") or {}
    job_status = status_payload.get("status")
    result = status_payload.get("result") or {}
    text = answer_text(result)
    sources = source_values(result)
    latency_ms = result.get("latency_ms")
    evidence_summary = result.get("evidence_summary") or {}

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    add_check("job_completed", job_status == "completed", f"status={job_status!r}")

    contains_any = [str(item) for item in expect.get("answer_contains_any", [])]
    add_check(
        "expected_answer_terms",
        _contains_any(text, contains_any),
        f"expected_any={contains_any}",
    )

    forbidden = [str(item) for item in expect.get("forbidden_content", [])]
    forbidden_hits = [term for term in forbidden if term.lower() in text.lower()]
    add_check("forbidden_content_absent", not forbidden_hits, f"hits={forbidden_hits}")

    min_sources = int(expect.get("min_sources", 0))
    add_check(
        "minimum_sources",
        len(sources) >= min_sources,
        f"source_count={len(sources)}, min_sources={min_sources}, sources={sources[:8]}",
    )

    max_latency_ms = expect.get("max_latency_ms")
    if max_latency_ms is not None:
        latency_ok = isinstance(latency_ms, (int, float)) and latency_ms <= float(max_latency_ms)
        add_check(
            "latency_slo",
            latency_ok,
            f"latency_ms={latency_ms}, max_latency_ms={max_latency_ms}",
        )

    if bool(expect.get("require_grounded_context", False)):
        raw_available = bool(evidence_summary.get("raw_context_available"))
        retrieved_count = int(evidence_summary.get("retrieved_doc_count") or 0)
        add_check(
            "grounded_context_available",
            raw_available and retrieved_count > 0,
            f"raw_available={raw_available}, retrieved_doc_count={retrieved_count}",
        )

    passed = all(check["passed"] for check in checks)
    return {
        "case_id": case.get("id"),
        "purpose": case.get("purpose"),
        "passed": passed,
        "checks": checks,
        "status": job_status,
        "latency_ms": latency_ms,
        "model_provider": result.get("model_provider"),
        "model_name": result.get("model_name"),
        "guardrails_applied": result.get("guardrails_applied") or [],
        "source_count": len(sources),
        "sources": sources,
        "answer_preview": " ".join(text.split())[:500],
    }


async def submit_and_poll_case(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    case: dict[str, Any],
    poll_timeout_seconds: float,
    poll_interval_seconds: float,
    client_ip: str | None,
) -> dict[str, Any]:
    trace_id = f"{case['id']}-{uuid.uuid4()}"
    headers = {"X-Trace-ID": trace_id}
    if client_ip:
        headers["X-Forwarded-For"] = client_ip

    started = time.perf_counter()
    try:
        submit = await client.post(
            f"{base_url.rstrip('/')}/query",
            json=case["payload"],
            headers=headers,
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        return {
            "case_id": case.get("id"),
            "purpose": case.get("purpose"),
            "passed": False,
            "status": "submit_transport_error",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "wall_clock_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "checks": [
                {
                    "name": "submit_transport_ok",
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            ],
        }
    if submit.status_code != 202:
        return {
            "case_id": case.get("id"),
            "purpose": case.get("purpose"),
            "passed": False,
            "status": "submit_failed",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "checks": [
                {
                    "name": "submit_accepted",
                    "passed": False,
                    "detail": f"HTTP {submit.status_code}: {submit.text[:500]}",
                }
            ],
        }

    job_id = submit.json().get("job_id")
    if not job_id:
        return {
            "case_id": case.get("id"),
            "purpose": case.get("purpose"),
            "passed": False,
            "status": "missing_job_id",
            "checks": [{"name": "job_id_present", "passed": False, "detail": submit.text[:500]}],
        }

    deadline = time.perf_counter() + poll_timeout_seconds
    last_payload: dict[str, Any] | None = None
    transport_errors = 0
    while time.perf_counter() < deadline:
        try:
            response = await client.get(
                f"{base_url.rstrip('/')}/query/{job_id}",
                params={"include_raw_context": "true"},
                headers=headers,
            )
        except (httpx.HTTPError, RuntimeError):
            transport_errors += 1
            await asyncio.sleep(poll_interval_seconds)
            continue
        if response.status_code != 200:
            await asyncio.sleep(poll_interval_seconds)
            continue
        last_payload = response.json()
        if last_payload.get("status") != "processing":
            evaluated = evaluate_case(case, last_payload)
            evaluated["wall_clock_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            evaluated["job_id"] = job_id
            return evaluated
        await asyncio.sleep(poll_interval_seconds)

    return {
        "case_id": case.get("id"),
        "purpose": case.get("purpose"),
        "passed": False,
        "status": "poll_timeout",
        "job_id": job_id,
        "last_payload": last_payload,
        "wall_clock_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "checks": [
            {
                "name": "poll_completed",
                "passed": False,
                "detail": (
                    f"Timed out after {poll_timeout_seconds}s; "
                    f"poll_transport_errors={transport_errors}"
                ),
            }
        ],
    }


async def run_cases(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.request_timeout_seconds, connect=5.0)
    limits = httpx.Limits(max_connections=max(args.concurrency * 3, 20))
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def run_one(index: int, case: dict[str, Any]) -> None:
            async with semaphore:
                client_ip = (
                    f"{args.client_ip_prefix}.{(index % 250) + 1}"
                    if args.client_ip_prefix
                    else None
                )
                result = await submit_and_poll_case(
                    client,
                    base_url=args.base_url,
                    case=case,
                    poll_timeout_seconds=args.poll_timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                    client_ip=client_ip,
                )
                results.append(result)

        await asyncio.gather(*(run_one(index, case) for index, case in enumerate(cases)))

    results.sort(key=lambda item: str(item.get("case_id")))
    latencies = [
        float(item["latency_ms"])
        for item in results
        if item.get("passed") and isinstance(item.get("latency_ms"), (int, float))
    ]
    p95 = (
        statistics.quantiles(latencies, n=20)[-1]
        if len(latencies) >= 20
        else (max(latencies) if latencies else None)
    )
    p50 = statistics.median(latencies) if latencies else None
    passed = sum(1 for item in results if item.get("passed"))

    return {
        "base_url": args.base_url,
        "case_file": str(args.cases),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "latency_ms": {"p50": p50, "p95": p95},
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 11 end-to-end validation queries against a live deployment."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--poll-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--require-count", type=int, default=50)
    parser.add_argument("--load-safe-only", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--client-ip-prefix",
        default="10.11.0",
        help="Optional prefix for synthetic X-Forwarded-For addresses.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(
        args.cases,
        load_safe_only=args.load_safe_only,
        case_ids=set(args.case_id) if args.case_id else None,
    )
    if not cases:
        print("No Phase 11 cases selected.", file=sys.stderr)
        return 2
    if args.require_count and len(cases) < args.require_count:
        print(
            f"Selected {len(cases)} cases, below required count {args.require_count}.",
            file=sys.stderr,
        )
        return 2

    report = asyncio.run(run_cases(args, cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed_cases = []
    for result in report["results"]:
        if result.get("passed"):
            continue
        failed_cases.append(
            {
                "case_id": result.get("case_id"),
                "status": result.get("status"),
                "model_provider": result.get("model_provider"),
                "model_name": result.get("model_name"),
                "failed_checks": [
                    {
                        "name": check.get("name"),
                        "detail": check.get("detail"),
                    }
                    for check in result.get("checks", [])
                    if not check.get("passed", False)
                ],
            }
        )

    print(
        json.dumps(
            {
                **{k: report[k] for k in ["total", "passed", "failed", "pass_rate", "latency_ms"]},
                "failed_case_examples": failed_cases[:5],
            },
            indent=2,
        )
    )
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
