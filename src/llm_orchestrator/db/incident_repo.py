from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


def _json_safe(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class IncidentRepo:
    dsn: str

    def __post_init__(self) -> None:
        self._engine = create_async_engine(self.dsn, pool_pre_ping=True)

    async def run_query(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        async with self._engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            rows = result.mappings().all()
            return [{str(k): _json_safe(v) for k, v in dict(r).items()} for r in rows]
