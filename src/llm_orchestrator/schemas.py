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
    cause: str = "Unknown cause"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = "No specific evidence extracted."
    # PRODUCTION FIX: Removed the brittle regex pattern. 
    # LLMs are stochastic; forcing strict regex at the parsing layer causes fatal 500 errors.
    # The RootCauseChain handles sanitization and grounding verification downstream.
    source: str = Field(
        default="NONE",
        description='Should ideally be a mapped ID like "DOC_1", but allows flexible fallback strings',
    )


class RootCauseResponse(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)


class RemediationRequest(BaseModel):
    user_query: str | None = None
    failure_mode: str = ""
    equipment_id: Optional[str] = None
    prompt_version: str = "1.0"


class RemediationResponse(BaseModel):
    safety_warnings: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class HistoricalSearchRequest(BaseModel):
    user_query: str
    equipment_id: Optional[str] = None
    days_back: int = 180
    limit: int = 50
    prompt_version: str = "1.0"


class EvidenceItem(BaseModel):
    claim: str = "Unknown claim"
    source: str = "Unknown source"


class HistoricalSearchResponse(BaseModel):
    summary: str = "No historical summary generated."
    key_stats: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class QueryRequest(BaseModel):
    chain: Optional[Literal["root_cause", "remediation", "historical"]] = None
    root_cause: Optional[RootCauseRequest] = None
    remediation: Optional[RemediationRequest] = None
    historical: Optional[HistoricalSearchRequest] = None


class QueryResponse(BaseModel):
    trace_id: Optional[str] = None
    latency_ms: Optional[float] = None
    guardrails_applied: list[str] = Field(default_factory=list)
    raw_context: str = ""
    chain: Literal["root_cause", "remediation", "historical"]
    result: RootCauseResponse | RemediationResponse | HistoricalSearchResponse
    model_provider: str
    model_name: str