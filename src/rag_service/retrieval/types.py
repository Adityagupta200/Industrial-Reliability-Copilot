from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class RetrievalFilters:
    equipment_id: Optional[str] = None
    severity: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    # Added explicitly for Multi-Tenancy and Role-Based Access Control
    plant_id: Optional[str] = None
    user_role: Optional[str] = None


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: Literal["semantic", "keyword", "hybrid", "rerank"] = "semantic"

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)
