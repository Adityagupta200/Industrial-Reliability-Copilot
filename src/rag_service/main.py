from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from rag_service.api.retrieve import router as retrieve_router
from rag_service.core.config import settings


# --- Production Logging Setup ---
def _configure_logging() -> None:
    level_name = str(settings.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_configure_logging()
logger = logging.getLogger("rag_service.main")

# --- Phase 7: Prometheus Metrics ---
# Tracks the golden signals: Traffic, Errors, and Latency
REQUEST_COUNT = Counter(
    "rag_service_requests_total",
    "Total HTTP requests routed to the RAG service",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "rag_service_request_duration_seconds",
    "Latency of requests through the RAG service",
    ["endpoint"],
)

# Global HTTP Client strictly for K8s dependency health checking
health_check_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for handling background resources.
    Crucial for graceful startups and shutdowns in Kubernetes.
    """
    global health_check_client

    # Initialize a fast-failing client (kept for potential future downstream API calls)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    health_check_client = httpx.AsyncClient(timeout=2.0, limits=limits)

    logger.info(
        {"event": "service_startup", "message": "RAG Service background resources initialized."}
    )

    yield  # Application runs here

    # Graceful shutdown cleanup
    if health_check_client:
        await health_check_client.aclose()
        logger.info(
            {"event": "service_shutdown", "message": "RAG Service background resources cleaned up."}
        )


app = FastAPI(
    title="Industrial Reliability Copilot - RAG Service",
    version="1.0.0",
    lifespan=lifespan,
    description="Microservice responsible for semantic and hybrid retrieval from the Vector DB.",
)

# --- Phase 7 Security: CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In strict production, restrict this to internal cluster IPs/Gateway DNS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Phase 7 Observability: Metrics Middleware ---
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to automatically track metrics for all RAG Service traffic."""
    start_time = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, http_status=status_code
        ).inc()


app.include_router(retrieve_router, prefix="/retrieve", tags=["retrieve"])


# --- Phase 7: Kubernetes Probes & Telemetry ---


@app.get("/health/live", tags=["Health"])
async def liveness_probe() -> dict[str, str]:
    """
    Kubernetes liveness probe.
    Returns 200 OK as long as the FastAPI event loop is running.
    """
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe() -> Response:
    """
    Production Fix: Shallow readiness probe.
    Confirms the application is fully booted and the event loop is responsive.
    By removing the deep downstream check to Qdrant, we ensure Kubernetes
    does not falsely mark this pod as unready (and sever the ELB connection)
    during CPU-heavy embedding inference spikes.
    """
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/metrics", tags=["Telemetry"])
async def get_metrics():
    """Prometheus scrape endpoint for Kubernetes metrics-server."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
