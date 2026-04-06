from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    id: str
    title: Optional[str] = None
    source: Optional[str] = None
    text: str
    score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RootCauseRequest(BaseModel):
    user_query: str
    anomaly_description: str = ""
    sensor_data: dict[str, Any] = Field(default_factory=dict)
    equipment_id: Optional[str] = None
    prompt_version: str = "1.0"


class Hypothesis(BaseModel):
    cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    # PRODUCTION FIX: Clear documentation for Pydantic schema on valid source states
    source: str = Field(..., description='Must be a valid mapped ID like "DOC_1" or "NONE"')


class RootCauseResponse(BaseModel):
    hypotheses: list[Hypothesis]


class RemediationRequest(BaseModel):
    user_query: str | None = None
    failure_mode: str = ""
    equipment_id: Optional[str] = None
    prompt_version: str = "1.0"


class RemediationResponse(BaseModel):
    safety_warnings: list[str]
    tools_required: list[str]
    steps: list[str]
    sources: list[str]


class HistoricalSearchRequest(BaseModel):
    user_query: str
    equipment_id: Optional[str] = None
    days_back: int = 180
    limit: int = 50
    prompt_version: str = "1.0"


class EvidenceItem(BaseModel):
    claim: str
    source: str  


class HistoricalSearchResponse(BaseModel):
    summary: str
    key_stats: dict[str, Any]
    evidence: list[EvidenceItem]


class QueryRequest(BaseModel):
    chain: Optional[Literal["root_cause", "remediation", "historical"]] = None
    root_cause: Optional[RootCauseRequest] = None
    remediation: Optional[RemediationRequest] = None
    historical: Optional[HistoricalSearchRequest] = None


class QueryResponse(BaseModel):
    # PRODUCTION FIX: Added distributed observability fields
    trace_id: Optional[str] = None
    latency_ms: Optional[float] = None
    guardrails_applied: list[str] = Field(default_factory=list)
    
    # PRODUCTION FIX: Bubbled up raw context text for global guardrail evaluation
    # This prevents the LLM judge from hallucinating a 0.0 score due to missing context.
    raw_context: str = "" 
    
    chain: Literal["root_cause", "remediation", "historical"]
    result: RootCauseResponse | RemediationResponse | HistoricalSearchResponse
    model_provider: str
    model_name: str