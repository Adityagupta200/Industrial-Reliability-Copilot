import os
import time
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- Production Logging Setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_gateway")
logging.getLogger("httpx").setLevel(os.getenv("HTTPX_LOG_LEVEL", "WARNING"))
logging.getLogger("httpcore").setLevel(os.getenv("HTTPCORE_LOG_LEVEL", "WARNING"))

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


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %d", name, raw, default)
        return default
    return max(value, minimum)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %.2fs", name, raw, default)
        return default
    return max(value, 0.1)


READINESS_TIMEOUT_S = _env_float("GATEWAY_READINESS_TIMEOUT_S", 2.0)
GATEWAY_MAX_CONNECTIONS = _env_int("GATEWAY_MAX_CONNECTIONS", 300)
GATEWAY_MAX_KEEPALIVE_CONNECTIONS = min(
    _env_int("GATEWAY_MAX_KEEPALIVE_CONNECTIONS", 100),
    GATEWAY_MAX_CONNECTIONS,
)
GATEWAY_REQUEST_TIMEOUT_S = _env_float("GATEWAY_REQUEST_TIMEOUT_S", 600.0)
GATEWAY_CONNECT_TIMEOUT_S = _env_float("GATEWAY_CONNECT_TIMEOUT_S", 10.0)
GATEWAY_POOL_TIMEOUT_S = _env_float("GATEWAY_POOL_TIMEOUT_S", 30.0)


def _metrics_endpoint(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if route_path:
        return str(route_path)

    path = request.url.path
    if path.startswith("/query/"):
        return "/query/{job_id}"
    return path


def _forward_headers(request: Request, trace_id: str) -> dict[str, str]:
    headers = {"X-Trace-ID": trace_id}
    client_ip = request.client.host if request.client else ""
    existing_forwarded_for = request.headers.get("X-Forwarded-For", "").strip()

    if existing_forwarded_for and client_ip:
        headers["X-Forwarded-For"] = f"{existing_forwarded_for}, {client_ip}"
    elif existing_forwarded_for:
        headers["X-Forwarded-For"] = existing_forwarded_for
    elif client_ip:
        headers["X-Forwarded-For"] = client_ip

    return headers


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    limits = httpx.Limits(
        max_keepalive_connections=GATEWAY_MAX_KEEPALIVE_CONNECTIONS,
        max_connections=GATEWAY_MAX_CONNECTIONS,
    )
    timeout = httpx.Timeout(
        GATEWAY_REQUEST_TIMEOUT_S,
        connect=GATEWAY_CONNECT_TIMEOUT_S,
        pool=GATEWAY_POOL_TIMEOUT_S,
    )

    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.info(
        "Gateway HTTP client initialized with max_connections=%d, "
        "max_keepalive_connections=%d, pool_timeout_s=%.1f.",
        GATEWAY_MAX_CONNECTIONS,
        GATEWAY_MAX_KEEPALIVE_CONNECTIONS,
        GATEWAY_POOL_TIMEOUT_S,
    )

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
        endpoint = _metrics_endpoint(request)
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, http_status=status_code
        ).inc()


# --- Phase 7: Kubernetes Probes & Observability ---


@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    if http_client is None:
        return JSONResponse(
            content={"status": "degraded", "detail": "HTTP client not initialized"},
            status_code=503,
        )

    try:
        response = await http_client.get(
            f"{ORCHESTRATOR_URL}/health/ready",
            timeout=READINESS_TIMEOUT_S,
        )
    except httpx.TimeoutException:
        logger.warning(
            "Gateway readiness check timed out contacting orchestrator at %s",
            ORCHESTRATOR_URL,
        )
        return JSONResponse(
            content={
                "status": "degraded",
                "dependencies": {"llm_orchestrator": {"status": "timeout"}},
            },
            status_code=503,
        )
    except httpx.RequestError as exc:
        logger.warning(
            "Gateway readiness check failed contacting orchestrator at %s: %s",
            ORCHESTRATOR_URL,
            exc,
        )
        return JSONResponse(
            content={
                "status": "degraded",
                "dependencies": {"llm_orchestrator": {"status": "unreachable"}},
            },
            status_code=503,
        )

    dependency_status = "ready" if response.status_code == 200 else "degraded"
    payload = {
        "status": "ready" if response.status_code == 200 else "degraded",
        "dependencies": {
            "llm_orchestrator": {
                "status": dependency_status,
                "http_status": response.status_code,
            }
        },
    }
    if response.status_code == 200:
        return payload

    return JSONResponse(
        content=payload,
        status_code=503,
    )


@app.get("/metrics", tags=["Telemetry"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Core Gateway Routing ---


@app.post("/query", tags=["Routing"])
async def route_query(request: Request):
    # PRODUCTION FIX: Avoid unsafe assert
    if http_client is None:
        raise RuntimeError("HTTP client not initialized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload provided.")

    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))

    try:
        response = await http_client.post(
            f"{ORCHESTRATOR_URL}/query",
            json=body,
            headers=_forward_headers(request, trace_id),
        )

    except httpx.RequestError as e:
        logger.exception(
            "Gateway to Orchestrator network error: %s: %s",
            type(e).__name__,
            str(e) or repr(e),
        )
        raise HTTPException(
            status_code=502, detail="Bad Gateway: Failed to communicate with downstream service."
        )
    except Exception:
        logger.exception("Unexpected Gateway Error")
        raise HTTPException(status_code=500, detail="Internal Gateway Error")

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type="application/json",
        headers={"X-Trace-ID": trace_id},
    )


@app.post("/feedback", tags=["Routing"])
async def route_feedback(request: Request):
    if http_client is None:
        raise RuntimeError("HTTP client not initialized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload provided.")

    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))

    try:
        response = await http_client.post(
            f"{ORCHESTRATOR_URL}/feedback",
            json=body,
            headers=_forward_headers(request, trace_id),
        )
    except httpx.RequestError as e:
        logger.exception(
            "Gateway to Orchestrator feedback network error: %s: %s",
            type(e).__name__,
            str(e) or repr(e),
        )
        raise HTTPException(
            status_code=502, detail="Bad Gateway: Failed to communicate with downstream service."
        )
    except Exception:
        logger.exception("Unexpected Gateway Feedback Error")
        raise HTTPException(status_code=500, detail="Internal Gateway Error")

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
        headers={"X-Trace-ID": trace_id},
    )


@app.get("/query/{job_id}", tags=["Routing"])
async def get_query_status(job_id: str, request: Request):
    # PRODUCTION FIX: Avoid unsafe assert
    if http_client is None:
        raise RuntimeError("HTTP client not initialized")

    try:
        response = await http_client.get(
            f"{ORCHESTRATOR_URL}/query/{job_id}",
            params=dict(request.query_params),
        )

        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") == "failed" and "API Configuration Error" in payload.get(
                "error", ""
            ):
                logger.error(
                    f"Gateway observed downstream API Configuration Error for Job {job_id}"
                )

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type="application/json",
        )
    except httpx.RequestError as e:
        logger.exception(
            "Gateway network error fetching job: %s: %s",
            type(e).__name__,
            str(e) or repr(e),
        )
        raise HTTPException(status_code=502, detail="Bad Gateway: Downstream unreachable")
