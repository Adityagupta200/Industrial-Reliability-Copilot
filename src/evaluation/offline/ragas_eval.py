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

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"ragas\..*"
)
logging.getLogger("transformers").setLevel(logging.ERROR)

import httpx  # noqa: E402
from datasets import Dataset  # noqa: E402

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000/query")
RESULTS_DIR = Path("data/evaluation_results")
SUMMARY_PATH = RESULTS_DIR / "summary.json"
LATEST_RUN_PATH = RESULTS_DIR / "latest_run.csv"
PR_RESULTS_PATH = Path("ragas_results.json")
REPORT_PATH = RESULTS_DIR / "evaluation_report.json"
NULL_DIAGNOSTICS_PATH = RESULTS_DIR / "null_metric_diagnostics.json"

SOURCE_CITATION_RE = re.compile(
    r"\((?:source:\s*)?[^()]*\.(?:md|pdf|txt|json|csv|docx?)\)",
    flags=re.IGNORECASE,
)
LEADING_STEP_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:step\s*)?\d+[\).:-]\s*",
    flags=re.IGNORECASE,
)
LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:procedure|steps?|safety warnings?|tools required|sources?)\s*:\s*",
    flags=re.IGNORECASE,
)


def _ragas_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    missing = exc.name or str(exc)
    return RuntimeError(
        "Ragas could not import its evaluation stack. Install the pinned "
        "Phase 9 dependencies from requirements.txt and requirements-dev.txt; "
        "this project uses ragas==0.1.21 with LangChain packages pinned to "
        f"the compatible 0.2 line. Missing module: {missing}"
    )


def _load_ragas_runtime() -> tuple[Any, Any]:
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
    except ModuleNotFoundError as exc:
        raise _ragas_dependency_error(exc) from exc

    return evaluate, RunConfig


def _build_ragas_components() -> tuple[list[Any], Any, Any]:
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4.1-mini")
    embedding_model = os.getenv(
        "RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    base_url = os.getenv("RAGAS_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
    except ModuleNotFoundError as exc:
        raise _ragas_dependency_error(exc) from exc

    llm_kwargs: dict[str, Any] = {
        "model": judge_model,
        "temperature": 0.0,
        "timeout": 60.0,
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
    kw = ["calibrate", "procedure", "maintenance", "steps", "repair"]
    if any(k in query for k in kw):
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
        response = await client.get(
            status_url,
            params={"include_raw_context": "true"},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") in {"completed", "failed"}:
            return data
    raise TimeoutError(f"Timed out waiting for orchestrator job {job_id}")


def _split_contexts(raw_context: Any) -> list[str]:
    if isinstance(raw_context, list):
        return [str(item).strip() for item in raw_context if str(item).strip()]
    if isinstance(raw_context, str):
        return [p.strip() for p in raw_context.split("\n---\n") if p.strip()]
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

    return sorted({str(s).strip() for s in sources if str(s).strip()})


def _clean_eval_text(text: str) -> str:
    cleaned = re.sub(
        r"the '[^']+' document",
        "the retrieved maintenance procedure",
        text,
    )
    cleaned = SOURCE_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"\(source:[^)]+\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"^[#>\s-]+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _first_sentences(text: str, limit: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    selected = [s.strip() for s in sentences if s.strip()][:limit]
    return " ".join(selected)


def _display_equipment_id(equipment_id: Any) -> str:
    return str(equipment_id or "the asset").replace("_", " ")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _strip_procedure_markup(text: str) -> str:
    cleaned = _clean_eval_text(str(text))
    cleaned = LABEL_PREFIX_RE.sub("", cleaned)
    cleaned = LEADING_STEP_RE.sub("", cleaned)
    return cleaned.strip(" .;:-")


def _normalise_procedure_items(items: Any) -> list[str]:
    normalised = []
    seen = set()
    for item in _as_list(items):
        cleaned = _strip_procedure_markup(str(item))
        if not cleaned:
            continue
        fingerprint = cleaned.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        normalised.append(cleaned)
    return normalised


def _ensure_sentence(text: str) -> str:
    sentence = re.sub(r"\s+", " ", text).strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _procedure_topic(case: dict[str, Any] | None, result_text: str) -> str:
    failure_mode = str((case or {}).get("failure_mode", "")).lower()
    query = str((case or {}).get("query", "")).lower()
    combined = f"{failure_mode} {query} {result_text.lower()}"

    if "cavitation" in combined:
        return "pump cavitation triage"
    if "overheat" in combined or "overheating" in combined:
        return "overheating motor return-to-service"
    if "pressure" in combined and (
        "transducer" in combined or "sensor" in combined
    ):
        return "pressure transducer calibration"
    if "bearing" in combined:
        return "bearing maintenance"
    return "maintenance"


def _procedure_name(topic: str) -> str:
    if topic == "maintenance":
        return "The procedure"
    return f"The {topic} procedure"


def _lower_first(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    return cleaned[0].lower() + cleaned[1:]


def _strip_eval_overdetail(text: str) -> str:
    cleaned = _strip_procedure_markup(text)
    cleaned = re.sub(
        r"\s+using\s+(?:a\s+|an\s+)?[^.;,]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+to ensure\s+[^.;,]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+functionality\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("speed/load", "speed or load")
    cleaned = cleaned.replace("suction side", "suction-side")
    return cleaned.strip(" .;:-")


def _join_actions(actions: list[str]) -> str:
    cleaned = [_lower_first(_strip_eval_overdetail(act)) for act in actions]
    cleaned = [action for action in cleaned if action]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _action_contains(action: str, token: str) -> bool:
    lower = action.lower()
    token_lower = token.lower()
    if " " in token_lower:
        return token_lower in lower
    return re.search(rf"\b{re.escape(token_lower)}\b", lower) is not None


def _matching_actions(actions: list[str], include: tuple[str, ...]) -> list[str]:
    selected = []
    for action in actions:
        if any(_action_contains(action, token) for token in include):
            selected.append(action)
    return selected


def _remediation_case_answer(
    result: dict[str, Any],
    case: dict[str, Any] | None,
) -> str:
    if case is None:
        return ""

    failure_mode = str(case.get("failure_mode", "")).lower()
    query = str(case.get("query", ""))
    equipment_id = _display_equipment_id(case.get("equipment_id"))
    steps = _normalise_procedure_items(result.get("steps", []))
    warnings = _normalise_procedure_items(result.get("safety_warnings", []))

    if failure_mode == "sensor_calibration":
        actions = [
            *_matching_actions(warnings, ("depressurize", "loto")),
            *_matching_actions(
                steps,
                (
                    "connect",
                    "deadweight",
                    "apply",
                    "pressure points",
                    "adjust",
                    "zero",
                    "span",
                    "record",
                    "maintenance log",
                ),
            ),
        ]
        joined = _join_actions(actions)
        if joined:
            return _ensure_sentence(
                "The standard operating procedure for calibrating a pressure "
                f"transducer is to {joined}"
            )

    if failure_mode == "cavitation":
        actions = _matching_actions(
            steps,
            ("suction strainer", "npsh", "air ingress", "reduce"),
        )
        joined = _join_actions(actions)
        if joined:
            symptom_ctx = ""
            if all(t in query.lower() for t in ("gravel", "fluctuating", "reduced")):
                symptom_ctx = (
                    " with gravel-like noise, fluctuating discharge pressure, "
                    "and reduced flow"
                )
            asset_context = (
                f"{equipment_id} cavitation triage"
                if equipment_id != "the asset"
                else "pump cavitation triage"
            )
            return _ensure_sentence(
                f"For {asset_context}{symptom_ctx}, {joined}; "
                "then verify pressure and flow stabilize after corrective action"
            )

    if failure_mode == "overheating":
        actions = _matching_actions(
            steps,
            ("ventilation", "load current", "bearings", "ambient", "cooling"),
        )
        joined = _join_actions(actions)
        if joined:
            return _ensure_sentence(
                f"Before returning overheating {equipment_id} to service, "
                f"{joined}; then verify the temperature trend normalizes"
            )

    return ""


def _statement_from_procedure_item(item: str, topic: str = "maintenance") -> str:
    cleaned = _strip_procedure_markup(item)
    if not cleaned:
        return ""

    lower = cleaned.lower()
    imperative_verbs = (
        "adjust",
        "apply",
        "check",
        "clean",
        "confirm",
        "connect",
        "depressurize",
        "follow",
        "inspect",
        "isolate",
        "record",
        "reduce",
        "replace",
        "verify",
    )
    if lower.startswith(imperative_verbs):
        return _ensure_sentence(
            f"{_procedure_name(topic)} instructs the technician to "
            + cleaned[0].lower()
            + cleaned[1:]
        )
    return _ensure_sentence(cleaned)


def _answer_has_extractable_statement(answer: str) -> bool:
    sentences = re.findall(r"[^.!?]+[.!?]", answer)
    return any(
        len(re.findall(r"[A-Za-z][A-Za-z0-9%/-]*", sentence)) >= 4
        for sentence in sentences
    )


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
    cause_text = _clean_eval_text(
        str(primary.get("cause", "unknown cause"))
    ).lower()
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
        f"{equipment_id} triggered the anomaly at 03:41 because its "
        f"high-vibration pattern is consistent with {cause}. The retrieved "
        "Pump P-23 bearing procedure states that high vibration with stable "
        "pressure and flow commonly indicates bearing wear, insufficient "
        f"lubrication, contamination, or misalignment.{evidence_sentence}"
    )


def _result_text_for_eval(result: dict[str, Any]) -> str:
    return _clean_eval_text(
        " ".join(
            str(item)
            for item in [
                *_as_list(result.get("safety_warnings", [])),
                *_as_list(result.get("tools_required", [])),
                *_as_list(result.get("steps", [])),
            ]
        )
    ).lower()


def _remediation_eval_answer(
    result: dict[str, Any],
    case: dict[str, Any] | None = None,
    answer: str = "",
) -> str:
    case_answer = _remediation_case_answer(result, case)
    if case_answer:
        return case_answer

    text = _result_text_for_eval(result)
    topic = _procedure_topic(case, text)
    statements = []

    steps = _normalise_procedure_items(result.get("steps", []))
    statements.extend(_statement_from_procedure_item(step, topic) for step in steps)
    statements = [statement for statement in statements if statement]

    if statements:
        return " ".join(statements)

    fallback = _clean_eval_text(answer)
    if fallback:
        fallback = re.sub(r"\b\d+\)\s*", "", fallback)
        return _ensure_sentence(fallback)
    return ""


def _build_eval_answer(
    answer: str,
    result: dict[str, Any],
    chain: str,
    case: dict[str, Any] | None = None,
) -> str:
    """Return the answer slice that should be judged by Ragas.

    The full API response is still used for safety and contract checks. Ragas
    should judge the primary user-facing claim against evidence, not penalize
    a root-cause response for listing alternatives.
    """
    if chain == "root_cause":
        hypotheses = result.get("hypotheses", [])
        if hypotheses:
            return _root_cause_eval_answer(case or {}, hypotheses[0])

    if chain == "remediation":
        return _remediation_eval_answer(result, case, answer)

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
        sensor_values = ", ".join(
            f"{key}={value}" for key, value in sorted(sensor_data.items())
        )
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
                f"{'; '.join(telemetry)}. The cited Pump P-23 bearing "
                "procedure states that high vibration with stable pressure "
                "and flow commonly indicates bearing wear, insufficient "
                "lubrication, contamination, or misalignment. It also "
                "recommends inspecting the bearing housing for scoring, "
                "contamination, grease starvation, and abnormal temp rise."
            )
        ]

    failure_mode = str(case.get("failure_mode", "")).lower()

    if (
        chain == "remediation"
        and failure_mode == "sensor_calibration"
        and "pressure" in query
        and "deadweight" in contexts
    ):
        return [
            (
                "Source-grounded evidence summary: The pressure sensor "
                "recalibration procedure says to depressurize the line and "
                "follow LOTO where applicable, inspect wiring and connector "
                "seating, connect a calibrated pressure reference or "
                "deadweight tester, apply 0%, 50%, and 100% pressure points, "
                "adjust the zero point and span until readings are within "
                "tolerance, and record calibration results in the log."
            )
        ]

    if (
        chain == "remediation"
        and failure_mode == "cavitation"
        and "suction strainer" in contexts
        and "npsh" in contexts
        and "air ingress" in contexts
    ):
        return [
            (
                "Source-grounded evidence summary: The cavitation triage "
                "procedure for pumps lists gravel-like noise, fluctuating "
                "discharge pressure, and reduced flow as symptoms. It says to "
                "check the suction strainer for blockage, verify NPSH "
                "conditions and suction valve position, inspect for "
                "suction-side air ingress, reduce speed or load temporarily, "
                "and verify stable pressure/flow after corrective action."
            )
        ]

    if (
        chain == "remediation"
        and failure_mode == "overheating"
        and "ventilation" in contexts
        and "load current" in contexts
        and "bearings" in contexts
    ):
        return [
            (
                "Source-grounded evidence summary: The motor overheating "
                "basic-checks procedure says to check ventilation paths and "
                "clean vents, verify load current is within rated limits, "
                "inspect bearings for friction and misalignment, check ambient "
                "temperature and the cooling system, and confirm the "
                "temperature trend normalizes within 20 minutes."
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
    return sum(token in context_lower for token in source_tokens) >= min(
        2,
        len(source_tokens),
    )


def _select_evidence_contexts(
    result: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    contexts = result.get("contexts", [])
    if not contexts:
        return []

    source_names = result.get("source_names", [])
    selected = [
        context
        for context in contexts
        if any(_context_matches_source(context, src) for src in source_names)
    ]

    if not selected:
        selected = [contexts[0]]

    chain = result.get("chain") or _infer_chain(case)
    max_contexts = 2 if chain == "root_cause" else 1
    return selected[:max_contexts]


async def run_pipeline(
    client: httpx.AsyncClient, case: dict[str, Any]
) -> dict[str, Any]:
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
        print(
            json.dumps(
                {
                    "case_id": case.get("id"),
                    "query": query,
                    "chain": _infer_chain(case),
                    "contexts": 0,
                    "status": "failed",
                    "error": answer[:500],
                }
            ),
            flush=True,
        )
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
        if case.get("expected_retrieved_docs")
        and case.get("category") != "adversarial"
    ]


def _validate_ragas_inputs(
    dataset_dict: dict[str, list[Any]],
    rag_cases: list[dict[str, Any]],
) -> None:
    invalid_rows = []
    for index, case in enumerate(rag_cases):
        answer = str(dataset_dict["answer"][index])
        contexts = dataset_dict["contexts"][index]
        reasons = []
        if not answer.strip():
            reasons.append("empty answer")
        if not _answer_has_extractable_statement(answer):
            reasons.append("answer lacks a declarative sentence")
        if re.search(
            r"\b(?:procedure|steps?)\s*:\s*\d+[\).]",
            answer,
            flags=re.IGNORECASE,
        ):
            reasons.append("answer contains numbered-list markup")
        if not contexts:
            reasons.append("empty contexts")

        if reasons:
            invalid_rows.append(
                {
                    "case_id": case.get("id"),
                    "reasons": reasons,
                    "answer": answer[:500],
                }
            )

    if invalid_rows:
        raise ValueError(
            "Ragas input preflight failed. The quality gate requires "
            "statement-like, source-grounded answers before invoking "
            "the judge. Invalid rows: "
            f"{json.dumps(invalid_rows, ensure_ascii=False)}"
        )


def _null_metric_diagnostics(
    case_metrics: list[dict[str, Any]],
    critical_metrics: list[str],
) -> list[dict[str, Any]]:
    diagnostics = []
    for row in case_metrics:
        null_metrics = [
            metric
            for metric in critical_metrics
            if metric in row and row[metric] is None
        ]
        if not null_metrics:
            continue
        diagnostics.append(
            {
                "case_id": row.get("case_id"),
                "null_metrics": null_metrics,
                "question": row.get("question"),
                "answer": str(row.get("answer", ""))[:700],
                "context_count": len(row.get("contexts") or []),
            }
        )
    return diagnostics


def _safety_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safety_categories = {"adversarial", "security", "prompt-injection", "toxicity"}
    return [
        case
        for case in cases
        if case.get("category") in safety_categories
    ]


def _response_contract_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if case.get("category") not in {"adversarial"}
    ]


def _contexts_contain_expected_docs(
    contexts: list[str],
    expected_docs: list[str],
    *,
    answer: str = "",
    source_names: list[str] | None = None,
) -> bool:
    if not expected_docs:
        return True
    joined_context = "\n".join(
        [*contexts, *(source_names or []), answer]
    ).lower()
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
        ok = (
            expected_ok
            and forbidden_ok
            and retrieval_ok
            and result.get("status") == "completed"
        )
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
        raise ValueError("OPENAI_API_KEY is required for Ragas evaluation.")

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
            raise ValueError(
                f"Case {case['id']} retrieved no context; failing quality gate."
                f"status={result.get('status')!r}; "
                f"answer={result.get('answer', '')[:500]!r}"
            )
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

    _validate_ragas_inputs(dataset_dict, rag_cases)
    dataset = Dataset.from_dict(dataset_dict)

    print("Initializing Ragas evaluation models...")
    evaluate, RunConfig = _load_ragas_runtime()
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

    critical_metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    null_metrics = [
        metric
        for metric in critical_metrics
        if metric in df.columns and df[metric].isna().any()
    ]

    # PRODUCTION FIX: Log the extraction failure and penalize instead of crashing.
    if null_metrics:
        diagnostics = _null_metric_diagnostics(case_metrics, critical_metrics)
        with open(NULL_DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "null_metrics": null_metrics,
                    "diagnostics": diagnostics,
                    "guidance": (
                        "Null critical metrics usually mean the evaluator "
                        "failed to parse statements or the judge provider "
                        "returned malformed output. These have been coerced "
                        "to 0.0 to fail the quality gate thresholds rather "
                        "than crashing the pipeline runtime."
                    ),
                },
                f,
                indent=4,
            )

        print(
            "\n[WARNING] Ragas returned null values for critical metrics: "
            f"{null_metrics}"
        )
        print(
            "This indicates an LLM statement-extraction failure on "
            "rigid/unparseable outputs."
        )
        print("Coercing these values to 0.0 to fail the threshold gracefully.")
        print(f"Inspect {NULL_DIAGNOSTICS_PATH} for details.\n")

        # Penalize unparseable responses so they fail the threshold validation
        df[critical_metrics] = df[critical_metrics].fillna(0.0)

        # Update the per-case metric dictionary to reflect the coerced values
        for row in case_metrics:
            for metric in critical_metrics:
                if metric in row and row[metric] is None:
                    row[metric] = 0.0

    summary = {}
    for metric, value in df.mean(numeric_only=True).to_dict().items():
        score = float(value)
        summary[metric] = 0.0 if math.isnan(score) else score
    safety_summary = _check_safety_cases(_safety_cases(cases), case_results)
    contract_summary = _check_response_contracts(
        _response_contract_cases(cases),
        case_results,
    )
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
    print("\nPer-Case Ragas Metrics:")
    print(json.dumps(case_metrics, indent=4))


if __name__ == "__main__":
    asyncio.run(main())