from __future__ import annotations

import logging

from fastapi import FastAPI

from rag_service.api.retrieve import router as retrieve_router
from rag_service.core.config import settings


def _configure_logging() -> None:
    level_name = str(settings.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_configure_logging()

app = FastAPI(
    title="Industrial Reliability Copilot - RAG Service",
    version="1.0.0",
)

app.include_router(retrieve_router, prefix="/retrieve", tags=["retrieve"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}
