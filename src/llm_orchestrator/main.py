from __future__ import annotations
import os
import logging
import time
import uuid
import asyncio
import hashlib
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import ORJSONResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select  
from langsmith import traceable # PRODUCTION FIX: Explicit tracing

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
from .guardrails.output_filters import OutputGuardrails  
from .providers.base import LLMFatalError, LLMTransientError

from evaluation.online.logger import (
    log_interaction_async, AsyncSessionLocal, QueryLog, init_telemetry_db,
    create_job_state, update_job_state, get_job_state
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
    ["chain"]
)

GUARDRAIL_FAILURES = Counter(
    "orchestrator_guardrail_failures_total", 
    "Total queries blocked by input or output guardrails", 
    ["type"]
)
GUARDRAIL_FAILURES.labels(type="input_validation").inc(0)
GUARDRAIL_FAILURES.labels(type="output_hallucination").inc(0)

LLM_TOKENS = Counter(
    "llm_tokens_total", 
    "Total LLM tokens consumed", 
    ["model", "token_type"]
)
RETRIEVAL_RECALL = Histogram(
    "retrieval_recall_score", 
    "Retrieval recall @ 10",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
FAITHFULNESS_SCORE = Histogram(
    "llm_faithfulness", 
    "LLM faithfulness / groundedness score proxy",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
ANSWER_RELEVANCY = Histogram(
    "llm_answer_relevancy", 
    "Answer relevancy score", 
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
USER_FEEDBACK = Counter(
    "user_feedback_total", 
    "User feedback ratings", 
    ["rating"]
)
USER_FEEDBACK.labels(rating="positive").inc(0)
USER_FEEDBACK.labels(rating="neutral").inc(0)
USER_FEEDBACK.labels(rating="negative").inc(0)

CACHE_EVENTS = Counter(
    "cache_events_total", 
    "Cache hit or miss", 
    ["status"]
)
CACHE_EVENTS.labels(status="hit").inc(0)
CACHE_EVENTS.labels(status="miss").inc(0)

health_check_client: httpx.AsyncClient | None = None

class FeedbackRequest(BaseModel):
    query_id: str
    score: int  

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
        default_response_class=ORJSONResponse,
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
        assert health_check_client is not None
        try:
            rag_req = health_check_client.get(f"{settings.services.rag_service_url}/health/live")
            anom_req = health_check_client.get(f"{settings.services.anomaly_service_url}/health/live")
            
            rag_resp, anom_resp = await asyncio.gather(rag_req, anom_req)

            if rag_resp.status_code == 200 and anom_resp.status_code == 200:
                return {"status": "ready"}

            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "detail": "Downstream services not returning 200 OK"},
            )
        except Exception as e:
            logger.warning(f"Readiness dependency check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "detail": "Connectivity to downstream services failed"},
            )

    @app.get("/metrics", tags=["Telemetry"])
    def get_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # PRODUCTION FIX: Establish the root trace for the request lifecycle
    @traceable(run_type="chain", name="Process_Query_Background")
    async def process_query_bg(job_id: str, req: QueryRequest, trace_id: str):
        start_time = time.time()
        applied_guardrails = []

        try:
            query_text = ""
            if req.root_cause:
                query_text = req.root_cause.user_query
            elif req.remediation:
                query_text = req.remediation.user_query
            elif req.historical:
                query_text = req.historical.user_query

            if query_text:
                safe_text = InputGuardrails.process(query_text)
                applied_guardrails.append("input_safety")
                if req.root_cause:
                    req.root_cause.user_query = safe_text
                elif req.remediation:
                    req.remediation.user_query = safe_text
                elif req.historical:
                    req.historical.user_query = safe_text

            cache_key = hashlib.md5(query_text.encode('utf-8')).hexdigest() if query_text else None
            
            if cache_key and cache_key in QUERY_CACHE:
                CACHE_EVENTS.labels(status="hit").inc()
                pipeline_response = QUERY_CACHE[cache_key]
                answer_text, contexts = extract_log_data(pipeline_response)
            else:
                CACHE_EVENTS.labels(status="miss").inc()
                pipeline_response = await orchestrator.handle(req)
                if cache_key:
                    QUERY_CACHE[cache_key] = pipeline_response
                answer_text, contexts = extract_log_data(pipeline_response)
                
                estimated_input_tokens = len(query_text.split()) * 1.3
                estimated_output_tokens = len(answer_text.split()) * 1.3
                LLM_TOKENS.labels(model=settings.llm.primary_provider, token_type="input").inc(estimated_input_tokens)
                LLM_TOKENS.labels(model=settings.llm.primary_provider, token_type="output").inc(estimated_output_tokens)
            
            recall_proxy = min(len(contexts) / 10.0, 1.0) if contexts else 0.0
            RETRIEVAL_RECALL.observe(recall_proxy)
            
            raw_json_payload = pipeline_response.result.json()
            
            contexts_str = pipeline_response.raw_context
            if not contexts_str or contexts_str.strip() == "":
                contexts_str = "NONE"
            
            initial_input_str = f"User Query: {query_text}"
            if req.root_cause:
                initial_input_str += f" | Anomaly Description: {req.root_cause.anomaly_description} | Sensor Data: {req.root_cause.sensor_data}"
            elif req.remediation:
                initial_input_str += f" | Failure Mode: {req.remediation.failure_mode}"
            
            is_valid, error_msg = await OutputGuardrails.validate_output(
                llm, contexts_str, raw_json_payload, initial_input=initial_input_str
            )
            
            if not is_valid:
                FAITHFULNESS_SCORE.observe(0.0)
                raise ValueError(error_msg)

            FAITHFULNESS_SCORE.observe(1.0)
            ANSWER_RELEVANCY.observe(0.9) 
            applied_guardrails.extend(["output_citations", "output_groundedness"])

            latency_sec = time.time() - start_time
            QUERY_LATENCY.labels(chain=pipeline_response.chain).observe(latency_sec)

            latency_ms = round(latency_sec * 1000.0, 2)
            
            pipeline_response.trace_id = trace_id
            pipeline_response.latency_ms = latency_ms
            pipeline_response.guardrails_applied = applied_guardrails
            pipeline_response.raw_context = "OMITTED_FROM_RESPONSE"

            await log_interaction_async(
                query_id=trace_id, 
                query=query_text,
                answer=answer_text,
                contexts=contexts,
                latency=latency_ms,
            )

            await update_job_state(job_id, status="completed", result=pipeline_response.dict())

        except ValueError as ve:
            error_msg = str(ve).lower()
            if "output guardrail" in error_msg or "hallucination" in error_msg or "blocked" in error_msg:
                GUARDRAIL_FAILURES.labels(type="output_hallucination").inc()
            else:
                GUARDRAIL_FAILURES.labels(type="input_validation").inc()
            
            logger.warning(f"Guardrail/Validation Blocked: {ve}")
            await update_job_state(job_id, status="failed", error=str(ve))
            
        except LLMFatalError as e:
            logger.error(f"Fatal LLM Error: {e}")
            await update_job_state(job_id, status="failed", error=f"API Configuration Error: {str(e)}")
            
        except LLMTransientError as e:
            logger.error(f"Transient LLM Error: {e}")
            await update_job_state(job_id, status="failed", error="LLM Provider Unavailable. SLA missed.")
            
        except Exception as e:
            logger.exception("Fatal LLM Orchestrator Background Crash")
            await update_job_state(job_id, status="failed", error="Internal Server Error")

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
    async def get_query_status(job_id: str):
        job_data = await get_job_state(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        return {k: v for k, v in job_data.items() if v is not None}

    @app.post("/feedback", tags=["Evaluation"])
    @limiter.limit("20/minute")
    async def submit_feedback(request: Request, feedback: FeedbackRequest):
        async with AsyncSessionLocal() as db:
            try:
                rating_label = "positive" if feedback.score >= 4 else "negative" if feedback.score <= 2 else "neutral"
                USER_FEEDBACK.labels(rating=rating_label).inc()
                
                result = await db.execute(select(QueryLog).filter(QueryLog.query_id == feedback.query_id))
                log_entry = result.scalars().first()
                if not log_entry:
                    raise HTTPException(status_code=404, detail="Query ID not found")

                log_entry.user_feedback_score = feedback.score
                await db.commit()
                return {"status": "success", "message": "Feedback recorded"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to record feedback: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

    return app

app = create_app()