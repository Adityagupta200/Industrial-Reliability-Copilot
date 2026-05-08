from __future__ import annotations
from datetime import datetime
from typing import Optional
from qdrant_client.http import models as qmodels
from .types import RetrievalFilters

def _dt_to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    return dt.isoformat()

def build_qdrant_filter(
    filters: Optional[RetrievalFilters],
    *,
    equipment_id_key: str = "equipment_id",
    severity_key: str = "severity",
    date_key: str = "date",
    plant_id_key: str = "plant_id", 
    roles_key: str = "allowed_roles",
) -> Optional[qmodels.Filter]:
    if not filters:
        return None

    must: list[qmodels.Condition] = []

    # PRODUCTION FIX: Replaced MatchAny with explicit Should (OR) condition.
    # Scalar string payloads can cause silent MatchAny dropouts depending on Qdrant index schemas.
    if getattr(filters, "equipment_id", None):
        must.append(
            qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key=equipment_id_key,
                        match=qmodels.MatchValue(value=filters.equipment_id)
                    ),
                    qmodels.FieldCondition(
                        key=equipment_id_key,
                        match=qmodels.MatchValue(value="all")
                    )
                ]
            )
        )

    if getattr(filters, "severity", None):
        must.append(
            qmodels.FieldCondition(
                key=severity_key, match=qmodels.MatchValue(value=filters.severity)
            )
        )

    if getattr(filters, "plant_id", None):
        must.append(
            qmodels.FieldCondition(
                key=plant_id_key, match=qmodels.MatchValue(value=filters.plant_id)
            )
        )

    if getattr(filters, "user_role", None):
        must.append(
            qmodels.FieldCondition(key=roles_key, match=qmodels.MatchAny(any=[filters.user_role]))
        )

    if getattr(filters, "date_from", None) or getattr(filters, "date_to", None):
        rng = qmodels.Range(
            gte=_dt_to_rfc3339(filters.date_from) if filters.date_from else None,
            lte=_dt_to_rfc3339(filters.date_to) if filters.date_to else None,
        )
        must.append(qmodels.FieldCondition(key=date_key, range=rng))

    if not must:
        return None

    return qmodels.Filter(must=must)