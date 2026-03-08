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
    equipment_id_key: str,
    severity_key: str,
    date_key: str,
) -> Optional[qmodels.Filter]:
    if not filters:
        return None

    must: list[qmodels.Condition] = []

    if filters.equipment_id:
        must.append(
            qmodels.FieldCondition(
                key=equipment_id_key,
                match=qmodels.MatchValue(value=filters.equipment_id),
            )
        )

    if filters.severity:
        must.append(
            qmodels.FieldCondition(
                key=severity_key,
                match=qmodels.MatchValue(value=filters.severity),
            )
        )

    if filters.date_from or filters.date_to:
        rng = qmodels.Range(
            gte=_dt_to_rfc3339(filters.date_from) if filters.date_from else None,
            lte=_dt_to_rfc3339(filters.date_to) if filters.date_to else None,
        )
        must.append(qmodels.FieldCondition(key=date_key, range=rng))

    if not must:
        return None

    return qmodels.Filter(must=must)
