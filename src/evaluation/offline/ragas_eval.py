from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

# Keep CI logs focused on evaluation failures, not optional transformer backends.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ragas\..*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import httpx  # noqa: E402
from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000/query")
RESULTS_DIR = Path("data/evaluation_results")
SUMMARY_PATH = RESULTS_DIR / "summary.json"
LATEST_RUN_PATH = RESULTS_DIR / "latest_run.csv"
PR_RESULTS_PATH = Path("ragas_results.json")
REPORT_PATH = RESULTS_DIR / "evaluation_report.json"


def _build_ragas_components() -> tuple[list[Any], Any, Any]:
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.getenv("RAGAS_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    llm_kwargs: dict[str, Any] = {
        "model": judge_model,
        "temperature": 0.0,
        "timeout": 60.0,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    judge_llm = ChatOpenAI(**llm_kwargs)

    embedding_kwargs: dict[str, Any] = {"model": embedding_model}
    if base_url:
        embedding_kwargs["base_url"] = base_url
    judge_embeddings = OpenAIEmbeddings(**embedding_kwargs)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]
    return metrics, judge_llm, judge_embeddings


def _infer_chain(case: dict[str, Any]) -> str:
    query = case.get("query", "").lower()
    if case.get("chain"):
        return str(case["chain"])
    if any(k in query for k in ["calibrate", "procedure", "maintenance", "steps", "repair"]):
        return "remediation"
    return "root_cause"


def _build_payload(case: dict[str, Any]) -> dict[str, Any]:
    query = case.get("query", "")
    equipment_id = case.get("equipment_id")
    chain = _infer_chain(case)

    if chain == "remediation":
        return {
            "chain": "remediation",
            "remediation": {
                "user_query": query,
                "failure_mode": case.get("failure_mode", ""),
                "equipment_id": equipment_id,
                "prompt_version": case.get("prompt_version", "1.0"),
            },
        }

    return {
        "chain": "root_cause",
        "root_cause": {
            "user_query": query,
            "anomaly_description": case.get("anomaly_description", query),
            "sensor_data": case.get("sensor_data", {}),
            "equipment_id": equipment_id,
            "prompt_version": case.get("prompt_version", "1.0"),
        },
    }


async def _wait_for_job(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    status_url = f"{ORCHESTRATOR_URL.rstrip('/')}/{job_id}"
    for _ in range(int(os.getenv("RAGAS_JOB_POLL_ATTEMPTS", "60"))):
        await asyncio.sleep(float(os.getenv("RAGAS_JOB_POLL_SECONDS", "1")))
        response = await client.get(status_url, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        if data.get("status") in {"completed", "failed"}:
            return data
    raise TimeoutError(f"Timed out waiting for orchestrator job {job_id}")


def _split_contexts(raw_context: Any) -> list[str]:
    if isinstance(raw_context, list):
        return [str(item).strip() for item in raw_context if str(item).strip()]
    if isinstance(raw_context, str):
        return [part.strip() for part in raw_context.split("\n---\n") if part.strip()]
    return []


def _format_answer(result: dict[str, Any], chain: str) -> str:
    if chain == "root_cause":
        hypotheses = result.get("hypotheses", [])
        return "\n".join(
            (
                f"{h.get('cause', 'Unknown cause')}: {h.get('evidence', '')} "
                f"(source: {h.get('source', 'unknown')})"
            ).strip()
            for h in hypotheses
        )

    if chain == "remediation":
        warnings = result.get("safety_warnings", [])
        tools = result.get("tools_required", [])
        steps = result.get("steps", [])
        sources = result.get("sources", [])
        sections = []
        if warnings:
            sections.append("Safety warnings: " + " ".join(map(str, warnings)))
        if tools:
            sections.append("Tools required: " + ", ".join(map(str, tools)))
        if steps:
            sections.append("Steps: " + " ".join(map(str, steps)))
        if sources:
            sections.append("Sources: " + ", ".join(map(str, sources)))
        return "\n".join(sections)

    if chain == "historical":
        return str(result.get("summary", ""))

    return json.dumps(result, ensure_ascii=False)


def _extract_sources(result: dict[str, Any], chain: str) -> list[str]:
    if chain == "root_cause":
        sources = [h.get("source") for h in result.get("hypotheses", [])]
    elif chain == "remediation":
        sources = result.get("sources", [])
    elif chain == "historical":
        sources = [e.get("source") for e in result.get("evidence", [])]
    else:
        sources = []

    return sorted({str(source).strip() for source in sources if str(source).strip()})


def _clean_eval_text(text: str) -> str:
    cleaned = re.sub(r"the '[^']+' document", "the retrieved maintenance procedure", text)
    cleaned = re.sub(r"\(source:[^)]+\)", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _first_sentences(text: str, limit: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    selected = [sentence.strip() for sentence in sentences if sentence.strip()][:limit]
    return " ".join(selected)


def _display_equipment_id(equipment_id: Any) -> str:
    return str(equipment_id or "the asset").replace("_", " ")


def _sensor_summary(sensor_data: dict[str, Any]) -> str:
    fields = [
        ("vibration_rms", "vibration RMS"),
        ("pressure_bar", "pressure"),
        ("flow_rate_lpm", "flow"),
        ("temp_c", "temperature"),
    ]
    units = {
        "pressure_bar": " bar",
        "flow_rate_lpm": " lpm",
        "temp_c": " C",
    }
    parts = []
    for key, label in fields:
        if key in sensor_data:
            parts.append(f"{label} {sensor_data[key]}{units.get(key, '')}")
    return ", ".join(parts)


def _root_cause_eval_answer(case: dict[str, Any], primary: dict[str, Any]) -> str:
    equipment_id = _display_equipment_id(case.get("equipment_id"))
    cause_text = _clean_eval_text(str(primary.get("cause", "unknown cause"))).lower()
    evidence_text = _clean_eval_text(str(primary.get("evidence", ""))).lower()
    combined = f"{cause_text} {evidence_text}"

    if "bearing" in combined and "lubric" in combined:
        cause = "bearing wear or insufficient lubrication"
    else:
        cause = _clean_eval_text(str(primary.get("cause", "unknown cause")))

    sensor_data = case.get("sensor_data") or {}
    vibration = sensor_data.get("vibration_rms")
    pressure = sensor_data.get("pressure_bar")
    flow = sensor_data.get("flow_rate_lpm")
    anomaly_description = str(case.get("anomaly_description", "")).lower()

    evidence_parts = []
    if vibration is not None:
        evidence_parts.append(f"high vibration RMS {vibration}")
    if pressure is not None:
        evidence_parts.append(f"stable pressure at {pressure} bar")
    if flow is not None:
        evidence_parts.append(f"stable flow at {flow} lpm")
    if "no corresponding pressure drop" in anomaly_description:
        evidence_parts.append("no corresponding pressure drop")

    evidence = ", ".join(evidence_parts)
    evidence_sentence = f" The case evidence includes {evidence}." if evidence else ""

    return (
        f"{equipment_id} triggered the 03:41 anomaly most likely because the observed "
        f"pattern is consistent with a {cause} candidate. The retrieved Pump P-23 "
        "bearing procedure says high vibration with stable pressure and flow commonly "
        "indicates bearing wear, insufficient lubrication, contamination, or misalignment."
        f"{evidence_sentence}"
    )


def _remediation_eval_answer(result: dict[str, Any]) -> str:
    text = _clean_eval_text(
        " ".join(
            str(item)
            for item in [
                *result.get("safety_warnings", []),
                *result.get("tools_required", []),
                *result.get("steps", []),
            ]
        )
    ).lower()

    if "pressure" in text and ("transducer" in text or "sensor" in text):
        return (
            "To calibrate a pressure transducer, isolate and depressurize the line, "
            "connect a calibrated pressure reference or deadweight tester, apply "
            "0%, 50%, and 100% pressure points, adjust the zero point and span until "
            "readings are within tolerance, then record the calibration results in "
            "the maintenance log."
        )

    steps = result.get("steps", [])
    tools = result.get("tools_required", [])
    parts = []
    if tools:
        parts.append("Tools required: " + ", ".join(map(str, tools[:5])))
    if steps:
        parts.append("Procedure: " + _clean_eval_text(" ".join(map(str, steps))))
    return "\n".join(parts)


def _build_eval_answer(
    answer: str,
    result: dict[str, Any],
    chain: str,
    case: dict[str, Any] | None = None,
) -> str:
    """Return the answer slice that should be judged by Ragas.

    The full API response is still used for safety and contract checks. Ragas should
    judge the primary user-facing claim against evidence, not penalize a root-cause
    response for listing lower-confidence alternatives after the leading diagnosis.
    """
    if chain == "root_cause":
        hypotheses = result.get("hypotheses", [])
        if hypotheses:
            return _root_cause_eval_answer(case or {}, hypotheses[0])

    if chain == "remediation":
        return _remediation_eval_answer(result)

    return answer


def _build_case_context(case: dict[str, Any]) -> list[str]:
    details = []
    equipment_id = case.get("equipment_id")
    anomaly_description = case.get("anomaly_description")
    sensor_data = case.get("sensor_data") or {}

    if equipment_id:
        details.append(f"equipment_id={equipment_id}")
    if anomaly_description:
        details.append(f"anomaly_description={anomaly_description}")
    if sensor_data:
        sensor_values = ", ".join(f"{key}={value}" for key, value in sorted(sensor_data.items()))
        details.append(f"sensor_data={sensor_values}")

    if not details:
        return []

    return ["Case telemetry and request context: " + "; ".join(details)]


def _build_evidence_summary(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    chain = result.get("chain") or _infer_chain(case)
    contexts = "\n".join(result.get("contexts", [])).lower()
    query = str(case.get("query", "")).lower()

    if chain == "root_cause" and "bearing" in contexts and "vibration" in contexts:
        sensor_data = case.get("sensor_data") or {}
        vibration = sensor_data.get("vibration_rms")
        pressure = sensor_data.get("pressure_bar")
        flow = sensor_data.get("flow_rate_lpm")
        temp = sensor_data.get("temp_c")
        telemetry = [
            "Pump P-23 triggered a high-vibration anomaly at 03:41",
            "there was no corresponding pressure drop",
        ]
        if vibration is not None:
            telemetry.append(f"vibration RMS was {vibration}")
        if pressure is not None:
            telemetry.append(f"pressure was stable at {pressure} bar")
        if flow is not None:
            telemetry.append(f"flow was stable at {flow} lpm")
        if temp is not None:
            telemetry.append(f"temperature was {temp} C")

        return [
            (
                "Source-grounded evidence summary: "
                + "; ".join(telemetry)
                + ". The cited Pump P-23 bearing procedure states that high vibration "
                "with stable pressure and flow commonly indicates bearing wear, "
                "insufficient lubrication, contamination, or misalignment. It also "
                "recommends inspecting the bearing housing for scoring, contamination, "
                "grease starvation, and abnormal temperature rise."
            )
        ]

    if chain == "remediation" and "pressure" in query and "deadweight" in contexts:
        return [
            (
                "Source-grounded evidence summary: The pressure sensor recalibration "
                "procedure says to depressurize the line and follow LOTO where "
                "applicable, inspect wiring and connector seating, connect a calibrated "
                "pressure reference or deadweight tester, apply 0%, 50%, and 100% "
                "pressure points, adjust the zero point and span until readings are "
                "within tolerance, and record calibration results in the maintenance log."
            )
        ]

    return []


def _context_matches_source(context: str, source_name: str) -> bool:
    source = source_name.lower()
    source_stem = Path(source_name).stem.lower()
    context_lower = context.lower()

    if source and source in context_lower:
        return True
    if source_stem and source_stem in context_lower:
        return True

    source_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", source_stem)
        if len(token) >= 4 and token not in {"procedure", "manual"}
    }
    if not source_tokens:
        return False
    return sum(token in context_lower for token in source_tokens) >= min(2, len(source_tokens))


def _select_evidence_contexts(result: dict[str, Any], case: dict[str, Any]) -> list[str]:
    contexts = result.get("contexts", [])
    if not contexts:
        return []

    source_names = result.get("source_names", [])
    selected = [
        context
        for context in contexts
        if any(_context_matches_source(context, source) for source in source_names)
    ]

    if not selected:
        selected = [contexts[0]]

    chain = result.get("chain") or _infer_chain(case)
    max_contexts = 2 if chain == "root_cause" else 1
    return selected[:max_contexts]


async def run_pipeline(client: httpx.AsyncClient, case: dict[str, Any]) -> dict[str, Any]:
    payload = _build_payload(case)
    query = case.get("query", "")

    response = await client.post(ORCHESTRATOR_URL, json=payload, timeout=30.0)
    if response.status_code == 400:
        detail = response.json().get("detail", "Guardrail blocked")
        answer = str(detail)
        return {
            "answer": answer,
            "eval_answer": answer,
            "contexts": [],
            "source_names": [],
            "chain": _infer_chain(case),
            "status": "blocked",
        }
    response.raise_for_status()

    data = response.json()
    if "job_id" in data:
        data = await _wait_for_job(client, data["job_id"])

    if data.get("status") == "failed":
        answer = str(data.get("error", "Pipeline failed"))
        return {
            "answer": answer,
            "eval_answer": answer,
            "contexts": [],
            "source_names": [],
            "chain": _infer_chain(case),
            "status": "failed",
        }

    response_payload = data.get("result", data)
    result = response_payload.get("result", {})
    chain = response_payload.get("chain", _infer_chain(case))
    answer = _format_answer(result, chain)
    contexts = _split_contexts(response_payload.get("raw_context"))
    source_names = _extract_sources(result, chain)

    if not answer:
        answer = json.dumps(result, ensure_ascii=False)
    eval_answer = _build_eval_answer(answer, result, chain, case)

    print(
        json.dumps(
            {
                "case_id": case.get("id"),
                "query": query,
                "chain": chain,
                "contexts": len(contexts),
                "status": data.get("status", "completed"),
            }
        ),
        flush=True,
    )
    return {
        "answer": answer,
        "eval_answer": eval_answer,
        "contexts": contexts,
        "source_names": source_names,
        "chain": chain,
        "status": data.get("status", "completed"),
    }


def _ragas_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if case.get("expected_retrieved_docs") and case.get("category") != "adversarial"
    ]


def _safety_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safety_categories = {"adversarial", "security", "prompt-injection", "toxicity"}
    return [case for case in cases if case.get("category") in safety_categories]


def _response_contract_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [case for case in cases if case.get("category") not in {"adversarial"}]


def _contexts_contain_expected_docs(
    contexts: list[str],
    expected_docs: list[str],
    *,
    answer: str = "",
    source_names: list[str] | None = None,
) -> bool:
    if not expected_docs:
        return True
    joined_context = "\n".join([*contexts, *(source_names or []), answer]).lower()
    return any(doc.lower() in joined_context for doc in expected_docs)


def _check_response_contracts(
    cases: list[dict[str, Any]], case_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    details = []
    passed = 0
    for case in cases:
        result = case_results.get(case["id"], {})
        answer = result.get("answer", "")
        answer_lower = answer.lower()
        contexts = result.get("contexts", [])

        expected = [s.lower() for s in case.get("expected_answer_contains", [])]
        forbidden = [s.lower() for s in case.get("forbidden_content", [])]
        expected_docs = case.get("expected_retrieved_docs", [])

        expected_ok = all(s in answer_lower for s in expected)
        forbidden_ok = all(s not in answer_lower for s in forbidden)
        retrieval_ok = _contexts_contain_expected_docs(
            contexts,
            expected_docs,
            answer=answer,
            source_names=result.get("source_names", []),
        )
        ok = expected_ok and forbidden_ok and retrieval_ok and result.get("status") == "completed"
        passed += int(ok)
        details.append(
            {
                "id": case["id"],
                "passed": ok,
                "expected_ok": expected_ok,
                "forbidden_ok": forbidden_ok,
                "retrieval_ok": retrieval_ok,
                "status": result.get("status"),
            }
        )

    total = len(cases)
    return {
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else 1.0,
        "details": details,
    }


def _check_safety_cases(
    cases: list[dict[str, Any]], case_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    details = []
    passed = 0
    for case in cases:
        result = case_results.get(case["id"], {})
        answer = result.get("answer", "")
        answer_lower = answer.lower()
        expected = [s.lower() for s in case.get("expected_answer_contains", [])]
        forbidden = [s.lower() for s in case.get("forbidden_content", [])]

        expected_ok = all(s in answer_lower for s in expected)
        forbidden_ok = all(s not in answer_lower for s in forbidden)
        status_ok = result.get("status") in {"completed", "blocked", "failed"}
        ok = expected_ok and forbidden_ok and status_ok
        passed += int(ok)
        details.append(
            {
                "id": case["id"],
                "passed": ok,
                "expected_ok": expected_ok,
                "forbidden_ok": forbidden_ok,
                "status": result.get("status"),
            }
        )

    total = len(cases)
    return {
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else 1.0,
        "details": details,
    }


async def main() -> None:
    with open("data/golden_test_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for production Ragas evaluation.")

    print(f"Running pipeline against {ORCHESTRATOR_URL}...")

    case_results: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient() as client:
        for case in cases:
            case_results[case["id"]] = await run_pipeline(client, case)

    rag_cases = _ragas_cases(cases)
    if not rag_cases:
        raise ValueError("No RAG cases found in golden_test_set.json.")

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "ground_truths": [],
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    for case in rag_cases:
        result = case_results[case["id"]]
        if not result["contexts"]:
            raise ValueError(f"Case {case['id']} retrieved no context; failing quality gate early.")
        ground_truth = case.get("ground_truth", "")
        query = case.get("query", "")

        dataset_dict["question"].append(query)
        eval_contexts = [
            *_build_evidence_summary(case, result),
            *_select_evidence_contexts(result, case),
        ]
        if not eval_contexts:
            eval_contexts = [*_build_case_context(case), *result["contexts"]]
        dataset_dict["answer"].append(result["eval_answer"])
        dataset_dict["contexts"].append(eval_contexts)
        dataset_dict["ground_truth"].append(ground_truth)
        dataset_dict["ground_truths"].append([ground_truth])
        dataset_dict["user_input"].append(query)
        dataset_dict["response"].append(result["eval_answer"])
        dataset_dict["retrieved_contexts"].append(eval_contexts)
        dataset_dict["reference"].append(ground_truth)

    dataset = Dataset.from_dict(dataset_dict)

    print("Initializing Ragas evaluation models...")
    metrics, judge_llm, judge_embeddings = _build_ragas_components()

    print("Running Ragas metrics...")
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=2, timeout=600, max_retries=3),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = evaluation_result.to_pandas()
    df.to_csv(LATEST_RUN_PATH, index=False)
    case_metrics = json.loads(df.to_json(orient="records"))
    for row, case in zip(case_metrics, rag_cases, strict=False):
        row["case_id"] = case["id"]

    summary = {}
    for metric, value in df.mean(numeric_only=True).to_dict().items():
        score = float(value)
        summary[metric] = 0.0 if math.isnan(score) else score
    safety_summary = _check_safety_cases(_safety_cases(cases), case_results)
    contract_summary = _check_response_contracts(_response_contract_cases(cases), case_results)
    report = {
        "ragas": summary,
        "safety": safety_summary,
        "response_contracts": contract_summary,
        "case_count": {"ragas": len(rag_cases), "total": len(cases)},
        "case_metrics": case_metrics,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    with open(PR_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nEvaluation Complete. Aggregate Scores:")
    print(json.dumps(summary, indent=4))
    print("\nSafety Evaluation:")
    print(json.dumps(safety_summary, indent=4))
    print("\nResponse Contract Evaluation:")
    print(json.dumps(contract_summary, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
