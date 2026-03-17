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


# --- Phase 7: Kubernetes Probes ---


@app.get("/health/live", tags=["Health"])
async def liveness_probe() -> dict[str, str]:
    """
    Kubernetes liveness probe.
    Returns 200 OK as long as the FastAPI event loop is running.
    """
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe() -> dict[str, str]:
    """
    Kubernetes readiness probe.
    Confirms the application is fully booted and ready to route traffic.
    In a fully fleshed-out production app, you might add a lightweight ping
    to your Qdrant DB here to ensure the connection is active before accepting queries.
    """
    return {"status": "ready"}
