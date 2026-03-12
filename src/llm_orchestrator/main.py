from __future__ import annotations

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


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
    async def query(request: Request, req: QueryRequest) -> QueryResponse:
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

            return await orchestrator.handle(req)

        except ValueError as ve:
            logger.warning(f"Guardrail/Validation Blocked: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))

        except HTTPException as he:
            logger.error(f"HTTP Error raised in pipeline: {he.status_code} - {he.detail}")
            raise he

        except Exception as e:
            logger.exception("Fatal LLM Orchestrator Crash:")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {repr(e)}") from e

    return app


app = create_app()
