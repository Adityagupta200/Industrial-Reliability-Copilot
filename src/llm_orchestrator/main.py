from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# Assuming logger.py is placed in src/evaluation/online/ as per Phase 6 instructions
from evaluation.online.logger import log_interaction_sync, SessionLocal, QueryLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


class FeedbackRequest(BaseModel):
    query_id: str
    score: int  # 1 for thumbs up, -1 for thumbs down


def extract_log_data(response: QueryResponse) -> tuple[str, list[str]]:
    """
    Safely extracts a summarized answer and retrieved contexts for logging
    purposes based on the specific chain executed.
    """
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


def create_app() -> FastAPI:
    settings = load_settings()

    llm = LLMClient(settings.llm)
    prompts = PromptLoader()

    anomaly_client = AnomalyClient(
        base_url=settings.services.anomaly_service_url,
        predict_anomaly_path=settings.services.anomaly_predict_anomaly_path,
        predict_rul_path=settings.services.anomaly_predict_rul_path,
        timeout_s=10.0,
    )

    rag_client = RAGClient(
        base_url=settings.services.rag_service_url,
        hybrid_path=settings.services.rag_retrieve_hybrid_path,
        procedures_path=settings.services.rag_retrieve_procedures_path,
        semantic_path=settings.services.rag_retrieve_semantic_path,
        timeout_s=60.0,
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

    app = FastAPI(title="LLM Orchestrator", version="0.1.0", default_response_class=ORJSONResponse)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    @limiter.limit("10/minute")
    async def query(
        request: Request, req: QueryRequest, response: Response, background_tasks: BackgroundTasks
    ) -> QueryResponse:
        start_time = time.time()
        query_id = str(uuid.uuid4())

        # Attach the query ID to the response headers for the frontend to consume
        response.headers["X-Query-ID"] = query_id

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

                if req.root_cause:
                    req.root_cause.user_query = safe_text
                elif req.remediation:
                    req.remediation.user_query = safe_text
                elif req.historical:
                    req.historical.user_query = safe_text

            # Execute pipeline
            pipeline_response = await orchestrator.handle(req)

            # Extract metrics and schedule non-blocking logging
            latency_ms = (time.time() - start_time) * 1000
            answer_text, contexts = extract_log_data(pipeline_response)

            background_tasks.add_task(
                log_interaction_sync,
                query_id=query_id,
                query=query_text,
                answer=answer_text,
                contexts=contexts,
                latency=latency_ms,
            )

            return pipeline_response

        except ValueError as ve:
            logger.warning(f"Guardrail/Validation Blocked: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))

        except HTTPException as he:
            logger.error(f"HTTP Error raised in pipeline: {he.status_code} - {he.detail}")
            raise he

        except Exception as e:
            logger.exception("Fatal LLM Orchestrator Crash:")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {repr(e)}") from e

    @app.post("/feedback")
    @limiter.limit("20/minute")
    async def submit_feedback(request: Request, feedback: FeedbackRequest):
        """Endpoint to receive user thumbs up/down feedback for online evaluation."""
        db = SessionLocal()
        try:
            log_entry = db.query(QueryLog).filter(QueryLog.query_id == feedback.query_id).first()
            if not log_entry:
                raise HTTPException(status_code=404, detail="Query ID not found")

            log_entry.user_feedback_score = feedback.score
            db.commit()
            return {"status": "success", "message": "Feedback recorded"}
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            db.close()

    return app


app = create_app()
