from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from llm_orchestrator.tracing import traceable

from .schemas import QueryRequest, QueryResponse
from .chains.root_cause_chain import RootCauseChain
from .chains.remediation_chain import RemediationChain
from .chains.historical_chain import HistoricalSearchChain

ChainName = Literal["root_cause", "remediation", "historical"]


def heuristic_route(text: str) -> ChainName:
    t = text.lower()
    if any(
        k in t
        for k in ["similar", "last ", "past ", "history", "incidents", "mttr", "downtime", "trend"]
    ):
        return "historical"
    if any(
        k in t
        for k in ["fix", "remed", "procedure", "steps", "replace", "calibrate", "maintenance"]
    ):
        return "remediation"
    return "root_cause"


@dataclass(frozen=True)
class ChainOrchestrator:
    root_cause: RootCauseChain
    remediation: RemediationChain
    historical: HistoricalSearchChain

    @traceable(run_type="chain", name="Chain_Orchestrator")  #   Explicit Tracing
    async def handle(self, req: QueryRequest) -> QueryResponse:
        chain: Optional[ChainName] = req.chain

        if chain is None:
            if req.root_cause is not None:
                chain = "root_cause"
            elif req.remediation is not None:
                chain = "remediation"
            elif req.historical is not None:
                chain = "historical"
            else:
                raise ValueError("No chain specified and no payload provided.")

        if chain == "root_cause":
            if req.root_cause is None:
                raise ValueError("root_cause payload is required.")
            result, prov, model, raw_ctx = await self.root_cause.run(req.root_cause)
            return QueryResponse(
                chain=chain,
                result=result,
                model_provider=prov,
                model_name=model,
                raw_context=raw_ctx,
            )

        if chain == "remediation":
            if req.remediation is None:
                raise ValueError("remediation payload is required.")
            result, prov, model, raw_ctx = await self.remediation.run(req.remediation)
            return QueryResponse(
                chain=chain,
                result=result,
                model_provider=prov,
                model_name=model,
                raw_context=raw_ctx,
            )

        if chain == "historical":
            if req.historical is None:
                raise ValueError("historical payload is required.")
            result, prov, model, raw_ctx = await self.historical.run(req.historical)
            return QueryResponse(
                chain=chain,
                result=result,
                model_provider=prov,
                model_name=model,
                raw_context=raw_ctx,
            )

        raise ValueError(f"Unknown chain '{chain}'.")
