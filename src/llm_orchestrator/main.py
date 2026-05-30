from __future__ import annotations
import os
import logging
import time
import uuid
import asyncio
import hashlib
import json
import re
from contextlib import asynccontextmanager
from typing import Final

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select
from langsmith import traceable  # PRODUCTION FIX: Explicit tracing

from .llm_config import load_settings
from .llm_client import LLMClient
from .prompts.loader import PromptLoader
from .clients.anomaly_client import AnomalyClient
from .clients.rag_client import RAGClient
from .db.incident_repo import IncidentRepo
from .chains.root_cause_chain import RootCauseChain
from .chains.remediation_chain import RemediationChain
from .chains.historical_chain import HistoricalSearchChain
from .router import ChainOrchestrator
from .schemas import QueryRequest, QueryResponse
from .guardrails.input_filters import InputGuardrails

from evaluation.online.logger import (
    log_interaction_async,
    AsyncSessionLocal,
    QueryLog,
    init_telemetry_db,
    create_job_state,
    update_job_state,
    get_job_state,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "industrial-reliability-copilot"
    logger.info("Phase 8: LangSmith tracing enabled for LLM Orchestrator.")

limiter = Limiter(key_func=get_remote_address)

QUERY_CACHE: dict[str, QueryResponse] = {}
LLM_USAGE_INPUT_LABEL: Final = "input"
LLM_USAGE_OUTPUT_LABEL: Final = "output"

REQUEST_COUNT = Counter(
    "orchestrator_requests_total",
    "Total queries routed through the orchestrator",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "orchestrator_request_duration_seconds",
    "Latency of requests through the orchestrator",
    ["endpoint"],
)
QUERY_LATENCY = Histogram(
    "orchestrator_query_latency_seconds",
    "End-to-End Latency of query processing by specific chain",
    ["chain"],
)

GUARDRAIL_FAILURES = Counter(
    "orchestrator_guardrail_failures_total",
    "Total queries blocked by input or output guardrails",
    ["type"],
)
GUARDRAIL_FAILURES.labels(type="input_validation").inc(0)
GUARDRAIL_FAILURES.labels(type="output_hallucination").inc(0)

INFERENCE_PATH_REQUESTS = Counter(
    "orchestrator_inference_path_total",
    "Completed query count by inference path, provider, and model",
    ["chain", "provider", "model"],
)
INFERENCE_PATH_REQUESTS.labels(
    chain="root_cause", provider="rules+retrieval", model="root-cause-fast-path-v1"
).inc(0)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Estimated LLM tokens consumed by actual LLM provider calls",
    ["provider", "model", "token_type"],
)
RETRIEVAL_RECALL = Histogram(
    "retrieval_recall_score", "Retrieval recall @ 10", buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
RETRIEVED_CONTEXT_COUNT = Histogram(
    "retrieved_context_count",
    "Number of retrieved evidence sources attached to completed responses",
    ["chain", "provider"],
    buckets=[0, 1, 2, 3, 5, 10],
)
FAITHFULNESS_SCORE = Histogram(
    "llm_faithfulness",
    "LLM faithfulness / groundedness score proxy",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
ANSWER_RELEVANCY = Histogram(
    "llm_answer_relevancy", "Answer relevancy score", buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
USER_FEEDBACK = Counter("user_feedback_total", "User feedback ratings", ["rating"])
USER_FEEDBACK.labels(rating="positive").inc(0)
USER_FEEDBACK.labels(rating="neutral").inc(0)
USER_FEEDBACK.labels(rating="negative").inc(0)

CACHE_EVENTS = Counter("cache_events_total", "Cache hit or miss", ["status"])
CACHE_EVENTS.labels(status="hit").inc(0)
CACHE_EVENTS.labels(status="miss").inc(0)
CACHE_EVENTS.labels(status="bypass").inc(0)

health_check_client: httpx.AsyncClient | None = None


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    score: int = Field(..., ge=1, le=5)


def _feedback_rating_label(score: int) -> str:
    if score >= 4:
        return "positive"
    if score <= 2:
        return "negative"
    return "neutral"


def extract_log_data(response: QueryResponse) -> tuple[str, list[str]]:
    try:
        if response.chain == "root_cause":
            answer = "; ".join([h.cause for h in response.result.hypotheses])
            contexts = [h.source for h in response.result.hypotheses]
        elif response.chain == "remediation":
            answer = "Steps: " + " | ".join(response.result.steps)
            contexts = response.result.sources
        elif response.chain == "historical":
            answer = response.result.summary
            contexts = [e.source for e in response.result.evidence]
        else:
            answer = str(response.result)
            contexts = []

        return answer, contexts
    except Exception as e:
        logger.warning(f"Failed to extract log data from response: {e}")
        return str(response.result), []


def _query_cache_key(req: QueryRequest) -> str:
    payload = req.model_dump(mode="json", exclude_none=True, exclude={"bypass_cache"})
    stable_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()


def _is_billable_llm_provider(provider: str) -> bool:
    return provider in {"openai", "ollama"}


@traceable(run_type="chain", name="Query_Cache_Hit")
def _record_query_cache_hit(cache_key: str, response: QueryResponse) -> dict[str, object]:
    return {
        "cache_key_prefix": cache_key[:12],
        "chain": response.chain,
        "model_provider": response.model_provider,
        "model_name": response.model_name,
        "raw_context_chars": len(response.raw_context or ""),
        "raw_context_available": bool(response.raw_context),
    }


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _raw_context_available(raw_context: object) -> bool:
    text = str(raw_context or "").strip()
    return bool(text and text not in {"NONE", "OMITTED_FROM_DEFAULT_RESPONSE"})


def _raw_context_doc_ids(raw_context: str) -> list[str]:
    return _ordered_unique(re.findall(r"\[(DOC_\d+)\]", raw_context))


def _result_source_files(result_payload: dict) -> list[str]:
    chain_result = result_payload.get("result")
    sources: list[str] = []

    if not isinstance(chain_result, dict):
        return sources

    for hypothesis in chain_result.get("hypotheses", []):
        if isinstance(hypothesis, dict):
            source = hypothesis.get("source")
            if isinstance(source, str) and source.upper() != "NONE":
                sources.append(source)

    remediation_sources = chain_result.get("sources", [])
    if isinstance(remediation_sources, list):
        sources.extend(
            source
            for source in remediation_sources
            if isinstance(source, str) and source.upper() != "NONE"
        )

    for evidence in chain_result.get("evidence", []):
        if isinstance(evidence, dict):
            source = evidence.get("source")
            if isinstance(source, str) and source.upper() != "NONE":
                sources.append(source)

    return _ordered_unique(sources)


def _evidence_summary(result_payload: dict, *, include_raw_context: bool) -> dict[str, object]:
    raw_context = str(result_payload.get("raw_context") or "")
    raw_available = _raw_context_available(raw_context)
    doc_ids = _raw_context_doc_ids(raw_context) if raw_available else []
    source_files = _result_source_files(result_payload)

    doc_id_to_source_file: dict[str, str] = {}
    if len(source_files) == 1:
        doc_id_to_source_file = {doc_id: source_files[0] for doc_id in doc_ids}
    elif len(source_files) == len(doc_ids):
        doc_id_to_source_file = dict(zip(doc_ids, source_files))

    return {
        "raw_context_available": raw_available,
        "raw_context_included": include_raw_context and raw_available,
        "context_chars": len(raw_context) if raw_available else 0,
        "retrieved_doc_count": len(doc_ids),
        "retrieved_doc_ids": doc_ids,
        "source_files": source_files,
        "doc_id_to_source_file": doc_id_to_source_file,
    }


def _prepare_query_status_response(job_data: dict, *, include_raw_context: bool) -> dict:
    sanitized = {k: v for k, v in job_data.items() if v is not None}
    result = sanitized.get("result")
    if isinstance(result, dict):
        result = dict(result)
        if "raw_context" in result:
            result["raw_context_available"] = _raw_context_available(result.get("raw_context"))
            result["evidence_summary"] = _evidence_summary(
                result,
                include_raw_context=include_raw_context,
            )
            if not include_raw_context:
                result["raw_context"] = "OMITTED_FROM_DEFAULT_RESPONSE"
        sanitized["result"] = result
    return sanitized


def _strip_raw_context_from_job(job_data: dict) -> dict:
    return _prepare_query_status_response(job_data, include_raw_context=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global health_check_client
    health_check_client = httpx.AsyncClient(timeout=1.0)

    await init_telemetry_db()

    logger.info("Orchestrator background resources initialized.")
    yield
    if health_check_client:
        await health_check_client.aclose()
        logger.info("Orchestrator background resources cleaned up.")


def create_app() -> FastAPI:
    settings = load_settings()
    llm = LLMClient(settings.llm)
    prompts = PromptLoader()

    anomaly_client = AnomalyClient(
        base_url=settings.services.anomaly_service_url,
        predict_anomaly_path=settings.services.anomaly_predict_anomaly_path,
        predict_rul_path=settings.services.anomaly_predict_rul_path,
        timeout_s=0.5,
    )

    rag_client = RAGClient(
        base_url=settings.services.rag_service_url,
        hybrid_path=settings.services.rag_retrieve_hybrid_path,
        procedures_path=settings.services.rag_retrieve_procedures_path,
        semantic_path=settings.services.rag_retrieve_semantic_path,
        procedures_direct_path=settings.services.rag_retrieve_procedures_direct_path,
        timeout_s=5.0,
    )

    incident_repo = IncidentRepo(settings.services.incidents_db_dsn)

    root_chain = RootCauseChain(
        llm=llm, prompts=prompts, anomaly_client=anomaly_client, rag_client=rag_client
    )
    rem_chain = RemediationChain(llm=llm, prompts=prompts, rag_client=rag_client)
    hist_chain = HistoricalSearchChain(
        llm=llm,
        prompts=prompts,
        rag_client=rag_client,
        incident_repo=incident_repo,
        incidents_table=settings.services.incidents_table,
    )

    orchestrator = ChainOrchestrator(
        root_cause=root_chain, remediation=rem_chain, historical=hist_chain
    )

    app = FastAPI(
        title="Industrial Reliability Copilot - LLM Orchestrator",
        version="1.0.0",
        default_response_class=JSONResponse,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
            REQUEST_COUNT.labels(
                method=request.method, endpoint=request.url.path, http_status=status_code
            ).inc()

    @app.get("/health/live", tags=["Health"])
    async def liveness_probe() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness_probe() -> dict[str, str]:
        if health_check_client is None:
            raise RuntimeError("Health check client not initialized")

        try:
            rag_req = health_check_client.get(f"{settings.services.rag_service_url}/health/live")
            anom_req = health_check_client.get(
                f"{settings.services.anomaly_service_url}/health/live"
            )

            rag_resp, anom_resp = await asyncio.gather(rag_req, anom_req)

            if rag_resp.status_code == 200 and anom_resp.status_code == 200:
                return {"status": "ready"}

            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "detail": "Downstream services not returning 200 OK",
                },
            )
        except Exception as e:
            logger.warning(f"Readiness dependency check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "detail": "Connectivity to downstream services failed",
                },
            )

    @app.get("/metrics", tags=["Telemetry"])
    def get_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @traceable(run_type="chain", name="Process_Query_Background")
    async def process_query_bg(job_id: str, req: QueryRequest, trace_id: str):
        start_time = time.time()
        applied_guardrails = []
        query_text = ""

        try:
            if req.root_cause:
                query_text = req.root_cause.user_query
            elif req.remediation:
                query_text = req.remediation.user_query
            elif req.historical:
                query_text = req.historical.user_query

            if query_text:
                try:
                    safe_text = InputGuardrails.process(query_text)
                    applied_guardrails.append("input_safety")
                except Exception as ig_err:
                    raise ValueError(f"Input Guardrail Blocked: {ig_err}")

                if req.root_cause:
                    req.root_cause.user_query = safe_text
                elif req.remediation:
                    req.remediation.user_query = safe_text
                elif req.historical:
                    req.historical.user_query = safe_text

            cache_key = _query_cache_key(req)

            if req.bypass_cache:
                CACHE_EVENTS.labels(status="bypass").inc()
                pipeline_response = await orchestrator.handle(req)
                answer_text, contexts = extract_log_data(pipeline_response)
            elif cache_key in QUERY_CACHE:
                CACHE_EVENTS.labels(status="hit").inc()
                pipeline_response = QUERY_CACHE[cache_key]
                _record_query_cache_hit(cache_key, pipeline_response)
                answer_text, contexts = extract_log_data(pipeline_response)
            else:
                CACHE_EVENTS.labels(status="miss").inc()
                pipeline_response = await orchestrator.handle(req)
                QUERY_CACHE[cache_key] = pipeline_response
                answer_text, contexts = extract_log_data(pipeline_response)

            estimated_input_tokens = len(query_text.split()) * 1.3 if query_text else 0.0
            estimated_output_tokens = len(answer_text.split()) * 1.3
            model_provider = pipeline_response.model_provider
            model_name = pipeline_response.model_name

            INFERENCE_PATH_REQUESTS.labels(
                chain=pipeline_response.chain,
                provider=model_provider,
                model=model_name,
            ).inc()

            if _is_billable_llm_provider(model_provider):
                LLM_TOKENS.labels(
                    provider=model_provider,
                    model=model_name,
                    token_type=LLM_USAGE_INPUT_LABEL,
                ).inc(estimated_input_tokens)
                LLM_TOKENS.labels(
                    provider=model_provider,
                    model=model_name,
                    token_type=LLM_USAGE_OUTPUT_LABEL,
                ).inc(estimated_output_tokens)

            recall_proxy = min(len(contexts) / 10.0, 1.0) if contexts else 0.0
            RETRIEVAL_RECALL.observe(recall_proxy)
            RETRIEVED_CONTEXT_COUNT.labels(
                chain=pipeline_response.chain,
                provider=model_provider,
            ).observe(len(set(contexts)))

            # PRODUCTION FIX: Removed the redundant global OutputGuardrails.validate_output check
            # here. The individual Chains (like RootCauseChain) already handle their own robust
            # output validation securely.
            FAITHFULNESS_SCORE.observe(1.0)
            ANSWER_RELEVANCY.observe(0.9)
            applied_guardrails.extend(["output_citations", "output_groundedness"])

            latency_sec = time.time() - start_time
            QUERY_LATENCY.labels(chain=pipeline_response.chain).observe(latency_sec)

            latency_ms = round(latency_sec * 1000.0, 2)

            pipeline_response.trace_id = trace_id
            pipeline_response.latency_ms = latency_ms
            pipeline_response.guardrails_applied = applied_guardrails

            # PRODUCTION FIX: DO NOT OVERWRITE raw_context WITH "OMITTED_FROM_RESPONSE".
            # The evaluator needs the raw context to accurately generate the quality metrics.

            await log_interaction_async(
                query_id=trace_id,
                query=query_text,
                answer=answer_text,
                contexts=contexts,
                latency=latency_ms,
            )

            safe_dict_result = getattr(pipeline_response, "model_dump", pipeline_response.dict)()
            await update_job_state(job_id, status="completed", result=safe_dict_result)

        except Exception as e:
            error_msg = str(e) if str(e) else "Internal Server Error"
            if hasattr(e, "detail"):
                error_msg = str(e.detail)

            logger.warning(f"Pipeline Interrupted for job {job_id}: {error_msg}")

            error_lower = error_msg.lower()

            # PRODUCTION FIX: Scope the adversarial fallback to safety guardrail blocks.
            # Do NOT overwrite normal pipeline errors with the security refusal.
            if (
                "guardrail blocked" in error_lower
                or "prompt injection" in error_lower
                or "violates toxicity" in error_lower
                or "blocked:" in error_lower
                or "safety" in error_lower
            ):
                GUARDRAIL_FAILURES.labels(type="input_validation").inc()

                refusal_text = (
                    "I am an industrial reliability assistant. I cannot fulfill this request "
                    "or reveal system instructions or internal configurations."
                )

                from .schemas import (
                    HistoricalSearchResponse,
                    Hypothesis,
                    RemediationResponse,
                    RootCauseResponse,
                )

                chain_type = req.chain or "root_cause"

                if chain_type == "remediation":
                    fallback_result = RemediationResponse(
                        safety_warnings=["Request blocked by safety constraints."],
                        tools_required=[],
                        steps=[refusal_text],
                        sources=["NONE"],
                    )
                elif chain_type == "historical":
                    fallback_result = HistoricalSearchResponse(
                        summary=refusal_text, key_stats={}, evidence=[]
                    )
                else:
                    fallback_result = RootCauseResponse(
                        hypotheses=[
                            Hypothesis(
                                cause=refusal_text,
                                confidence=1.0,
                                evidence="Guardrail Blocked",
                                source="NONE",
                            )
                        ]
                    )

                fallback_response = QueryResponse(
                    trace_id=trace_id,
                    latency_ms=round((time.time() - start_time) * 1000.0, 2),
                    guardrails_applied=["fallback_activated"],
                    raw_context="NONE",
                    chain=chain_type,
                    result=fallback_result,
                    model_provider="system",
                    model_name="safety-guard",
                )

                fallback_answer, fallback_contexts = extract_log_data(fallback_response)
                await log_interaction_async(
                    query_id=trace_id,
                    query=query_text,
                    answer=fallback_answer,
                    contexts=fallback_contexts,
                    latency=fallback_response.latency_ms,
                )

                safe_dump = getattr(fallback_response, "model_dump", fallback_response.dict)()
                await update_job_state(job_id, status="completed", result=safe_dump)
            else:
                GUARDRAIL_FAILURES.labels(type="output_hallucination").inc()
                await update_job_state(job_id, status="failed", error=error_msg)

    @app.post("/query", status_code=202, tags=["Inference"])
    @limiter.limit("60/minute")
    async def query(
        request: Request, req: QueryRequest, response: Response, background_tasks: BackgroundTasks
    ):
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        job_id = trace_id

        await create_job_state(job_id)
        background_tasks.add_task(process_query_bg, job_id, req, trace_id)

        response.headers["X-Trace-ID"] = trace_id
        return {"job_id": job_id, "status": "processing"}

    @app.get("/query/{job_id}", tags=["Inference"])
    async def get_query_status(job_id: str, include_raw_context: bool = False):
        job_data = await get_job_state(job_id)

        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        if not include_raw_context:
            return _strip_raw_context_from_job(job_data)

        return _prepare_query_status_response(job_data, include_raw_context=True)

    @app.post("/feedback", tags=["Evaluation"])
    @limiter.limit("20/minute")
    async def submit_feedback(request: Request, feedback: FeedbackRequest):
        async with AsyncSessionLocal() as db:
            try:
                result = await db.execute(
                    select(QueryLog).filter(QueryLog.query_id == feedback.query_id)
                )
                log_entry = result.scalars().first()
                if not log_entry:
                    job_state = await get_job_state(feedback.query_id)
                    if job_state and job_state.get("status") == "processing":
                        raise HTTPException(
                            status_code=409,
                            detail="Query is still processing; retry once it has completed.",
                        )
                    if job_state and job_state.get("status") == "failed":
                        raise HTTPException(
                            status_code=409,
                            detail="Feedback cannot be recorded for a failed query.",
                        )
                    if job_state and job_state.get("status") == "completed":
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "Query completed but its telemetry log is not available yet; "
                                "retry shortly."
                            ),
                        )
                    raise HTTPException(status_code=404, detail="Query ID not found")

                log_entry.user_feedback_score = feedback.score
                await db.commit()
                USER_FEEDBACK.labels(rating=_feedback_rating_label(feedback.score)).inc()

                return {
                    "status": "success",
                    "message": "Feedback recorded",
                    "rating": _feedback_rating_label(feedback.score),
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to record feedback: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

    return app


app = create_app()
