from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

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


def create_app() -> FastAPI:
    settings = load_settings()

    llm = LLMClient(settings.llm)
    prompts = PromptLoader()

    anomaly_client = AnomalyClient(
        base_url=settings.services.anomaly_service_url,
        predict_anomaly_path=settings.services.anomaly_predict_anomaly_path,
        predict_rul_path=settings.services.anomaly_predict_rul_path,
        timeout_s=2.0,
    )
    rag_client = RAGClient(
        base_url=settings.services.rag_service_url,
        hybrid_path=settings.services.rag_retrieve_hybrid_path,
        procedures_path=settings.services.rag_retrieve_procedures_path,
        semantic_path=settings.services.rag_retrieve_semantic_path,
        timeout_s=1.5,
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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        try:
            return await orchestrator.handle(req)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


app = create_app()
