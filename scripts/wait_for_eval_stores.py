from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _infer_chain(case: dict[str, Any]) -> str:
    query = case.get("query", "").lower()
    if case.get("chain"):
        return str(case["chain"])
    if any(k in query for k in ["calibrate", "procedure", "maintenance", "steps", "repair"]):
        return "remediation"
    return "root_cause"


def _failure_terms(case: dict[str, Any]) -> list[str]:
    text = f"{case.get('query', '')} {case.get('anomaly_description', '')}".lower()
    sensor_data = case.get("sensor_data") or {}
    terms: list[str] = []

    vibration = sensor_data.get("vibration_rms")
    pressure = sensor_data.get("pressure_bar")
    if "bearing" in text or "lubric" in text or "vibration" in text:
        terms.extend(["bearing failure", "bearing wear", "lubrication", "relubrication"])
    if isinstance(vibration, (int, float)) and vibration >= 4.0:
        terms.extend(["high vibration", "bearing failure", "bearing wear", "lubrication"])
    if "cavitation" in text or (isinstance(pressure, (int, float)) and pressure < 1.0):
        terms.extend(["cavitation", "suction blockage", "fluctuating discharge pressure"])
    if "sensor" in text or "transducer" in text:
        terms.extend(["sensor malfunction", "pressure transducer", "calibration"])
    return list(dict.fromkeys(terms))


def _root_cause_query(case: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(case.get("query", "")),
            f"Equipment: {case.get('equipment_id') or 'unknown'}",
            f"Anomaly: {case.get('anomaly_description') or case.get('query', '')}",
            "Observed signals: " + ", ".join(_failure_terms(case)),
        ]
    )


def _document_name(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    candidates = [
        metadata.get("source_file"),
        metadata.get("file_name"),
        metadata.get("source"),
        doc.get("title"),
        doc.get("source"),
        doc.get("id"),
    ]
    for candidate in candidates:
        if candidate:
            return os.path.basename(str(candidate))
    return "unknown"


def _expected_vector_docs(case: dict[str, Any]) -> list[str]:
    return [
        name
        for name in case.get("expected_retrieved_docs", [])
        if Path(name).suffix.lower() in {".pdf", ".md", ".markdown"}
    ]


def _matches_expected(doc_names: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    lowered = " ".join(doc_names).lower()
    return any(name.lower() in lowered for name in expected)


def _probe_payloads(case: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    chain = _infer_chain(case)
    filters: dict[str, Any] = {}
    if case.get("equipment_id"):
        filters["equipment_id"] = case["equipment_id"]

    if chain == "remediation":
        query = case.get("query", "")
        return [
            (
                "/retrieve/procedures",
                {
                    "query": query,
                    "k": 8,
                    "filters": filters,
                },
            )
        ]

    query = _root_cause_query(case)
    return [
        (
            "/retrieve/hybrid",
            {
                "query": query,
                "out_k": 8,
                "filters": filters,
            },
        ),
        (
            "/retrieve/procedures",
            {
                "query": query,
                "k": 6,
                "filters": filters,
            },
        ),
    ]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        case
        for case in cases
        if case.get("expected_retrieved_docs") and case.get("category") != "adversarial"
    ]


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query = re.sub(r"\s+", " ", str(payload.get("query", ""))).strip()
    return {
        "query": query[:160],
        "filters": payload.get("filters", {}),
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to {url}: {exc}") from exc

    return json.loads(body)


def probe_once(base_url: str, cases: list[dict[str, Any]]) -> tuple[bool, Any]:
    results = []
    all_ready = True

    for case in cases:
        expected = _expected_vector_docs(case)
        case_doc_names: list[str] = []
        probe_results = []

        for path, payload in _probe_payloads(case):
            body = _post_json(f"{base_url.rstrip('/')}{path}", payload)
            docs = body.get("documents") or []
            names = [_document_name(doc) for doc in docs]
            case_doc_names.extend(names)
            probe_results.append(
                {
                    "path": path,
                    "count": body.get("count", len(docs)),
                    "sources": names[:8],
                    "payload": _summarize_payload(payload),
                }
            )

        case_ready = bool(case_doc_names) and _matches_expected(case_doc_names, expected)
        all_ready = all_ready and case_ready
        results.append(
            {
                "case_id": case.get("id"),
                "ready": case_ready,
                "expected_vector_docs": expected,
                "probes": probe_results,
            }
        )

    return all_ready, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait until the live RAG service can retrieve golden-set evidence."
    )
    parser.add_argument("--base-url", default=os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--cases", default="data/golden_test_set.json")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if not cases:
        print("No golden retrieval cases found.", flush=True)
        return 1

    deadline = time.monotonic() + args.timeout_seconds
    last_results: Any = None
    while time.monotonic() < deadline:
        try:
            ready, results = probe_once(args.base_url, cases)
            last_results = results
            print(json.dumps({"rag_eval_store_probe": results}, indent=2), flush=True)
            if ready:
                print("Golden retrieval stores are queryable.", flush=True)
                return 0
        except Exception as exc:
            last_results = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"rag_eval_store_probe_error": last_results}), flush=True)

        time.sleep(args.interval_seconds)

    print(
        "Evaluation stores did not become ready before timeout:\n"
        + json.dumps(last_results, indent=2),
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
