#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

try:
    from scripts.phase11_e2e_validation import DEFAULT_CASES, evaluate_case, load_cases
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from phase11_e2e_validation import DEFAULT_CASES, evaluate_case, load_cases


DEFAULT_REPORT = Path("data/phase11/reports/load_test_report.json")


def _prepare_load_case(case: dict[str, Any], *, bypass_cache: bool) -> dict[str, Any]:
    cloned = copy.deepcopy(case)
    cloned["payload"]["bypass_cache"] = bypass_cache
    return cloned


def _classify_submit_error(result: dict[str, Any]) -> int | None:
    checks = result.get("checks") or []
    for check in checks:
        detail = str(check.get("detail", ""))
        if "HTTP 429" in detail:
            return 429
        if "HTTP 5" in detail:
            return 500
    return None


def _task_failure_result(
    case_id: str,
    *,
    purpose: str | None,
    exc: BaseException,
    started_at: float,
) -> dict[str, Any]:
    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    return {
        "case_id": case_id,
        "purpose": purpose,
        "passed": False,
        "status": "client_task_error",
        "latency_ms": elapsed_ms,
        "wall_clock_ms": elapsed_ms,
        "checks": [
            {
                "name": "client_task_completed",
                "passed": False,
                "detail": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        ],
    }


def _pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) < 20:
        return max(values)
    return statistics.quantiles(values, n=100)[p - 1]


def _synthetic_client_ip(args: argparse.Namespace, request_index: int) -> str | None:
    if not args.client_ip_prefix:
        return None
    client_octet = (request_index % args.client_count) + 1
    return f"{args.client_ip_prefix}.{client_octet}"


def load_threshold_failures(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    success_rate = float(report.get("success_rate") or 0.0)
    observed_qps = float(report.get("observed_qps") or 0.0)
    min_observed_qps = float(args.target_qps) * float(args.min_observed_qps_ratio)
    wall_p95 = (report.get("latency_ms") or {}).get("wall_p95")
    status_counts = report.get("status_counts") or {}
    five_xx = sum(count for status, count in status_counts.items() if str(status).startswith("5"))

    if success_rate < args.min_success_rate:
        failures.append(
            f"success_rate={success_rate:.4f} below min_success_rate={args.min_success_rate:.4f}"
        )
    if observed_qps < min_observed_qps:
        failures.append(
            f"observed_qps={observed_qps:.3f} below min_observed_qps={min_observed_qps:.3f}"
        )
    if wall_p95 is not None and float(wall_p95) > args.max_wall_p95_ms:
        failures.append(f"wall_p95_ms={float(wall_p95):.2f} above max={args.max_wall_p95_ms:.2f}")
    if five_xx > args.max_5xx:
        failures.append(f"5xx_count={five_xx} above max_5xx={args.max_5xx}")

    return failures


async def run_load(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    prepared_cases = [_prepare_load_case(case, bypass_cache=args.bypass_cache) for case in cases]
    timeout = httpx.Timeout(args.request_timeout_seconds, connect=5.0)
    limits = httpx.Limits(
        max_connections=max(args.max_in_flight, args.poll_worker_count + args.client_count, 100),
        max_keepalive_connections=max(args.poll_worker_count, args.client_count, 50),
    )
    http_semaphore = asyncio.Semaphore(args.max_in_flight)
    interval_seconds = 1.0 / args.target_qps if args.target_qps > 0 else 0.0
    results: list[dict[str, Any]] = []
    submit_latencies: list[float] = []
    job_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def submit_one(
        client: httpx.AsyncClient,
        *,
        case_index: int,
        request_index: int,
    ) -> dict[str, Any]:
        case = copy.deepcopy(prepared_cases[case_index % len(prepared_cases)])
        case["id"] = f"{case['id']}-load-{request_index:06d}"
        trace_id = f"{case['id']}-{uuid.uuid4()}"
        headers = {"X-Trace-ID": trace_id}
        client_ip = _synthetic_client_ip(args, request_index)
        if client_ip:
            headers["X-Forwarded-For"] = client_ip

        started_at = time.perf_counter()
        try:
            async with http_semaphore:
                response = await client.post(
                    f"{args.base_url.rstrip('/')}/query",
                    json=case["payload"],
                    headers=headers,
                )
        except (httpx.HTTPError, RuntimeError) as exc:
            return {
                "accepted": False,
                "result": _task_failure_result(
                    case["id"],
                    purpose=case.get("purpose"),
                    exc=exc,
                    started_at=started_at,
                )
                | {
                    "status": "submit_transport_error",
                    "checks": [
                        {
                            "name": "submit_transport_ok",
                            "passed": False,
                            "detail": f"{type(exc).__name__}: {str(exc)[:500]}",
                        }
                    ],
                },
            }

        submit_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        if response.status_code != 202:
            return {
                "accepted": False,
                "result": {
                    "case_id": case.get("id"),
                    "purpose": case.get("purpose"),
                    "passed": False,
                    "status": "submit_failed",
                    "latency_ms": submit_latency_ms,
                    "wall_clock_ms": submit_latency_ms,
                    "checks": [
                        {
                            "name": "submit_accepted",
                            "passed": False,
                            "detail": f"HTTP {response.status_code}: {response.text[:500]}",
                        }
                    ],
                },
            }

        job_id = response.json().get("job_id")
        if not job_id:
            return {
                "accepted": False,
                "result": {
                    "case_id": case.get("id"),
                    "purpose": case.get("purpose"),
                    "passed": False,
                    "status": "missing_job_id",
                    "latency_ms": submit_latency_ms,
                    "wall_clock_ms": submit_latency_ms,
                    "checks": [
                        {
                            "name": "job_id_present",
                            "passed": False,
                            "detail": response.text[:500],
                        }
                    ],
                },
            }

        return {
            "accepted": True,
            "submit_latency_ms": submit_latency_ms,
            "job": {
                "case": case,
                "job_id": job_id,
                "headers": headers,
                "submitted_at": started_at,
            },
        }

    async def poll_job(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
        case = job["case"]
        job_id = job["job_id"]
        deadline = job["submitted_at"] + args.poll_timeout_seconds
        last_payload: dict[str, Any] | None = None
        transport_errors = 0

        while time.perf_counter() < deadline:
            try:
                async with http_semaphore:
                    response = await client.get(
                        f"{args.base_url.rstrip('/')}/query/{job_id}",
                        params={"include_raw_context": "true"},
                        headers=job["headers"],
                    )
            except (httpx.HTTPError, RuntimeError):
                transport_errors += 1
                await asyncio.sleep(args.poll_interval_seconds)
                continue

            if response.status_code != 200:
                await asyncio.sleep(args.poll_interval_seconds)
                continue

            last_payload = response.json()
            if last_payload.get("status") != "processing":
                evaluated = evaluate_case(case, last_payload)
                evaluated["wall_clock_ms"] = round(
                    (time.perf_counter() - job["submitted_at"]) * 1000.0,
                    2,
                )
                evaluated["job_id"] = job_id
                return evaluated

            await asyncio.sleep(args.poll_interval_seconds)

        return {
            "case_id": case.get("id"),
            "purpose": case.get("purpose"),
            "passed": False,
            "status": "poll_timeout",
            "job_id": job_id,
            "last_payload": last_payload,
            "wall_clock_ms": round((time.perf_counter() - job["submitted_at"]) * 1000.0, 2),
            "checks": [
                {
                    "name": "poll_completed",
                    "passed": False,
                    "detail": (
                        f"Timed out after {args.poll_timeout_seconds}s; "
                        f"poll_transport_errors={transport_errors}"
                    ),
                }
            ],
        }

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, transport=getattr(args, "transport", None)
    ) as client:
        submit_start = time.perf_counter()
        target_submissions = max(1, int(round(args.duration_seconds * args.target_qps)))
        next_submit_at = submit_start
        submitted = 0
        accepted = 0
        submit_tasks: set[asyncio.Task[dict[str, Any]]] = set()

        async def poll_worker() -> None:
            while True:
                try:
                    job = await job_queue.get()
                    try:
                        results.append(await poll_job(client, job))
                    finally:
                        job_queue.task_done()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive load harness safety
                    results.append(
                        _task_failure_result(
                            "unknown-poll-task",
                            purpose=None,
                            exc=exc,
                            started_at=submit_start,
                        )
                    )

        workers = [asyncio.create_task(poll_worker()) for _ in range(args.poll_worker_count)]

        async def drain_submit_tasks(*, wait: bool) -> None:
            nonlocal accepted
            if not submit_tasks:
                return

            done: set[asyncio.Task[dict[str, Any]]]
            if wait:
                done, pending = await asyncio.wait(
                    submit_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                submit_tasks.clear()
                submit_tasks.update(pending)
            else:
                done = {task for task in submit_tasks if task.done()}
                submit_tasks.difference_update(done)

            for task in done:
                result = task.result()
                if not result.get("accepted"):
                    results.append(result["result"])
                    continue
                accepted += 1
                submit_latencies.append(float(result["submit_latency_ms"]))
                await job_queue.put(result["job"])

        try:
            while submitted < target_submissions:
                await drain_submit_tasks(wait=False)

                if len(submit_tasks) >= args.max_in_flight:
                    await drain_submit_tasks(wait=True)
                    continue

                now = time.perf_counter()
                if now < next_submit_at:
                    await asyncio.sleep(min(0.01, next_submit_at - now))
                    continue

                while (
                    submitted < target_submissions
                    and time.perf_counter() >= next_submit_at
                    and len(submit_tasks) < args.max_in_flight
                ):
                    task = asyncio.create_task(
                        submit_one(
                            client,
                            case_index=submitted % len(prepared_cases),
                            request_index=submitted,
                        )
                    )
                    submit_tasks.add(task)
                    submitted += 1
                    next_submit_at += interval_seconds

            while submit_tasks:
                await drain_submit_tasks(wait=True)

            await job_queue.join()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    total_elapsed = time.perf_counter() - submit_start
    active_elapsed = max(args.duration_seconds, 0.001)
    passed = sum(1 for item in results if item.get("passed"))
    status_counts: dict[str, int] = {}
    for item in results:
        key = str(_classify_submit_error(item) or item.get("status") or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1

    wall_latencies = [
        float(item["wall_clock_ms"])
        for item in results
        if item.get("passed") and isinstance(item.get("wall_clock_ms"), (int, float))
    ]
    service_latencies = [
        float(item["latency_ms"])
        for item in results
        if item.get("passed") and isinstance(item.get("latency_ms"), (int, float))
    ]

    success_rate = passed / submitted if submitted else 0.0
    report = {
        "base_url": args.base_url,
        "duration_seconds": args.duration_seconds,
        "target_qps": args.target_qps,
        "client_count": args.client_count,
        "submitted": submitted,
        "accepted": accepted,
        "completed": len(results),
        "passed": passed,
        "failed": submitted - passed,
        "success_rate": round(success_rate, 4),
        "observed_qps": round(accepted / active_elapsed, 3),
        "submitted_qps": round(submitted / active_elapsed, 3),
        "completion_qps": round(len(results) / total_elapsed, 3) if total_elapsed > 0 else 0.0,
        "timing": {
            "active_submit_seconds": round(active_elapsed, 3),
            "total_elapsed_seconds": round(total_elapsed, 3),
            "drain_seconds": round(max(total_elapsed - active_elapsed, 0.0), 3),
        },
        "status_counts": status_counts,
        "latency_ms": {
            "submit_p50": statistics.median(submit_latencies) if submit_latencies else None,
            "submit_p95": _pct(submit_latencies, 95),
            "service_p50": statistics.median(service_latencies) if service_latencies else None,
            "service_p95": _pct(service_latencies, 95),
            "wall_p50": statistics.median(wall_latencies) if wall_latencies else None,
            "wall_p95": _pct(wall_latencies, 95),
        },
        "thresholds": {
            "min_success_rate": args.min_success_rate,
            "min_observed_qps": round(args.target_qps * args.min_observed_qps_ratio, 3),
            "min_observed_qps_ratio": args.min_observed_qps_ratio,
            "max_wall_p95_ms": args.max_wall_p95_ms,
            "max_5xx": args.max_5xx,
        },
        "sample_failures": [item for item in results if not item.get("passed")][:10],
    }

    failure_reasons = load_threshold_failures(report, args)
    report["failure_reasons"] = failure_reasons
    report["passed_thresholds"] = not failure_reasons
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 11 sustained load test against a live API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--target-qps", type=float, default=50.0)
    parser.add_argument("--client-count", type=int, default=50)
    parser.add_argument("--max-in-flight", type=int, default=500)
    parser.add_argument("--poll-worker-count", type=int, default=250)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    parser.add_argument("--client-ip-prefix", default="10.12.0")
    parser.add_argument("--case-id", action="append", default=["phase11_001", "phase11_050"])
    parser.add_argument("--bypass-cache", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--min-observed-qps-ratio", type=float, default=0.95)
    parser.add_argument("--max-wall-p95-ms", type=float, default=2500.0)
    parser.add_argument("--max-5xx", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases, case_ids=set(args.case_id) if args.case_id else None)
    if not cases:
        print("No load cases selected.", file=sys.stderr)
        return 2

    report = asyncio.run(run_load(args, cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "submitted": report["submitted"],
                "accepted": report["accepted"],
                "completed": report["completed"],
                "success_rate": report["success_rate"],
                "observed_qps": report["observed_qps"],
                "submitted_qps": report["submitted_qps"],
                "completion_qps": report["completion_qps"],
                "latency_ms": report["latency_ms"],
                "timing": report["timing"],
                "passed_thresholds": report["passed_thresholds"],
                "failure_reasons": report.get("failure_reasons", []),
            },
            indent=2,
        )
    )
    return 0 if report["passed_thresholds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
