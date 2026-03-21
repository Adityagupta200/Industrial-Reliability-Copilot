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
    # Defaulting to an empty string to allow purely conversational requests
    anomaly_description: str = ""
    # Defaulting to an empty dict so it isn't strictly required in the payload
    sensor_data: dict[str, Any] = Field(default_factory=dict)
    equipment_id: Optional[str] = None
    prompt_version: str = "1.0"


class Hypothesis(BaseModel):
    cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    source: str  # "DOC:<id>"


class RootCauseResponse(BaseModel):
    hypotheses: list[Hypothesis]


class RemediationRequest(BaseModel):
    user_query: str | None = None
    # Defaulting to an empty string to allow conversational queries
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
    source: str  # "SQL" or "DOC:<id>"


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
    chain: Literal["root_cause", "remediation", "historical"]
    result: RootCauseResponse | RemediationResponse | HistoricalSearchResponse
    model_provider: str
    model_name: str