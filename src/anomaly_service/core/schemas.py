from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional


SchemaId = Literal["uah", "hai", "swat", "modbus", "edgeiiot", "nasa_rul"]


class SensorRequest(BaseModel):
    timestamp: datetime
    sensor_values: Dict[str, Any] = Field(
        ...,
        description="Key-value map of sensor/tag names to numeric values.",
        min_length=1,
    )
    schema_id: Optional[SchemaId] = Field(
        default=None,
        description="Optional explicit schema. If omitted, service attempts to infer.",
    )


class AnomalyResponse(BaseModel):
    timestamp: datetime
    schema_id: SchemaId
    anomaly_score: float
    confidence: float
    model_version: str


class RULResponse(BaseModel):
    timestamp: datetime
    predicted_rul: float
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    anomaly_model_loaded: bool
    rul_model_loaded: bool
