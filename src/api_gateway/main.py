import os
import time
import logging
import uuid
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
REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total HTTP requests routed through the gateway",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds", "Latency of requests through the gateway", ["endpoint"]
)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://llm-orchestrator:8000")
http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    timeout = httpx.Timeout(600.0, connect=10.0)

    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.info("Gateway HTTP client initialized with connection pooling.")

    yield 
    
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
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
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
    return {"status": "alive"}

@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    return {"status": "ready"}

@app.get("/metrics", tags=["Telemetry"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- Core Gateway Routing ---

@app.post("/query", tags=["Routing"])
async def route_query(request: Request):
    assert http_client is not None

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload provided.")

    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))

    try:
        headers = {"X-Trace-ID": trace_id}
        if request.client:
            headers["X-Forwarded-For"] = request.client.host
            
        response = await http_client.post(f"{ORCHESTRATOR_URL}/query", json=body, headers=headers)

    except httpx.RequestError as e:
        logger.error(f"Gateway to Orchestrator network error: {str(e)}")
        raise HTTPException(
            status_code=502, detail="Bad Gateway: Failed to communicate with downstream service."
        )
    except Exception as e:
        logger.exception("Unexpected Gateway Error")
        raise HTTPException(status_code=500, detail="Internal Gateway Error")

    return Response(
        content=response.content, 
        status_code=response.status_code, 
        media_type="application/json",
        headers={"X-Trace-ID": trace_id}
    )

@app.get("/query/{job_id}", tags=["Routing"])
async def get_query_status(job_id: str):
    assert http_client is not None
    try:
        response = await http_client.get(f"{ORCHESTRATOR_URL}/query/{job_id}")
        
        # PRODUCTION FIX: Log explicit errors parsed from the downstream orchestrator payload
        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") == "failed" and "API Configuration Error" in payload.get("error", ""):
                logger.error(f"Gateway observed downstream API Configuration Error for Job {job_id}")
                
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type="application/json"
        )
    except httpx.RequestError as e:
        logger.error(f"Gateway network error fetching job: {e}")
        raise HTTPException(status_code=502, detail="Bad Gateway: Downstream unreachable")