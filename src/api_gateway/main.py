import os
import time
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- Production Logging Setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_gateway")

# --- Prometheus Metrics ---
# Essential for Phase 7: Prometheus/Grafana Monitoring
REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total HTTP requests routed through the gateway",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds", "Latency of requests through the gateway", ["endpoint"]
)

# Internal DNS routing via Kubernetes ClusterIP
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://llm-orchestrator:8000")

# Global HTTP client for connection pooling
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Crucial for establishing a persistent connection pool rather than
    opening/closing a new HTTP client on every request.
    """
    global http_client

    # Configure production-grade connection limits and timeouts
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)

    # 5 seconds to connect to the internal service, 120 seconds for the LLM to stream/generate a response
    timeout = httpx.Timeout(120.0, connect=5.0)

    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.info("Gateway HTTP client initialized with connection pooling.")

    yield  # App runs here

    # Graceful shutdown for Kubernetes pod termination
    if http_client:
        await http_client.aclose()
        logger.info("Gateway HTTP client connection pool closed.")


app = FastAPI(
    title="Industrial Reliability Copilot - API Gateway",
    version="1.0.0",
    lifespan=lifespan,
    description="Main entry point exposing external load balancer to internal ML microservices.",
)

# --- Phase 7 Security: CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in strict production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to automatically track metrics for all Gateway traffic."""
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


# --- Phase 7: Kubernetes Probes & Observability ---


@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    """
    Kubernetes Liveness Probe.
    Validates the gateway container itself is running and event loop is responsive.
    """
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    """
    Kubernetes Readiness Probe.
    Validates the gateway is ready AND the downstream orchestrator is reachable.
    If the orchestrator is restarting, the Gateway should not accept external traffic.
    """
    assert http_client is not None
    try:
        # Fast 2-second timeout just to check if orchestrator is listening
        resp = await http_client.get(f"{ORCHESTRATOR_URL}/health/ready", timeout=2.0)
        if resp.status_code == 200:
            return {"status": "ready"}
    except Exception as e:
        logger.warning(f"Readiness check failed - Downstream orchestrator unavailable: {e}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "degraded", "detail": "Downstream LLM Orchestrator unavailable"},
    )


@app.get("/metrics", tags=["Telemetry"])
async def metrics():
    """Exposes gateway metrics for Kubernetes Prometheus scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Core Gateway Routing ---


@app.post("/query", tags=["Routing"])
async def route_query(request: Request):
    """
    Reverse proxies the natural language query to the internal LLM Orchestrator.
    """
    assert http_client is not None

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload provided.")

    try:
        # Proxy request using the globally pooled client
        # Forward the client's actual IP so the Orchestrator's rate limiter works correctly
        headers = {"X-Forwarded-For": request.client.host} if request.client else {}
        response = await http_client.post(f"{ORCHESTRATOR_URL}/query", json=body, headers=headers)

    except httpx.RequestError as e:
        # Handles connection errors (e.g., DNS resolution fails, orchestrator pod crashed)
        logger.error(f"Gateway to Orchestrator network error: {str(e)}")
        raise HTTPException(
            status_code=502, detail="Bad Gateway: Failed to communicate with downstream service."
        )
    except Exception as e:
        logger.exception("Unexpected Gateway Error")
        raise HTTPException(status_code=500, detail="Internal Gateway Error")

    # FIX: Move validation outside the try-except block to prevent exception swallowing.
    # Pass through the exact status code and payload from the orchestrator
    if response.status_code != 200:
        # Return the exact error content and status code the Orchestrator provided
        return Response(
            content=response.content, 
            status_code=response.status_code, 
            media_type="application/json"
        )
        
    return response.json()