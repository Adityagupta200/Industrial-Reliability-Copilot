from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


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
            return [dict(r) for r in rows]
