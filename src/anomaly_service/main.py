from __future__ import annotations
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from loguru import logger

from anomaly_service.core.config import settings
from .core.logging import setup_logging
from .core.metrics import REQUESTS, Timer
from .core.model_loader import LoadedModels, load_anomaly_model, load_rul_model
from .core.preprocess import (
    infer_schema_id,
    load_anomaly_artifacts,
    load_schema,
    preprocess_anomaly,
    preprocess_rul,
)
from .core.schemas import SensorRequest, AnomalyResponse, RULResponse, HealthResponse
from .core.inference import anomaly_infer, rul_infer

# Globals for caching models and artifacts in-memory
MODELS: LoadedModels | None = None
ANOM_SCHEMA: dict | None = None
RUL_SCHEMA: dict | None = None
ANOM_ARTIFACTS: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup (loading models, schema, setting up telemetry)
    and shutdown (graceful memory cleanup).
    """
    global MODELS, ANOM_SCHEMA, RUL_SCHEMA, ANOM_ARTIFACTS

    setup_logging(settings.log_level)
    logger.info({"event": "service_startup", "message": "Initializing Anomaly Service..."})

    anomaly_model = None
    anomaly_version = None
    rul_model = None
    rul_version = None

    # 1. Load Anomaly Detection Assets
    try:
        anomaly_model, anomaly_version = load_anomaly_model()
        ANOM_SCHEMA = load_schema(settings.anomaly_schema_path)
        ANOM_ARTIFACTS = load_anomaly_artifacts(settings.anomaly_preprocess_path)
        logger.info({"event": "anomaly_model_loaded", "version": anomaly_version})
    except Exception as e:
        logger.exception({"event": "anomaly_model_load_failed", "error": str(e)})

    # 2. Load Remaining Useful Life (RUL) Assets
    try:
        rul_model, rul_version, RUL_SCHEMA = load_rul_model()
        logger.info({"event": "rul_model_loaded", "version": rul_version})
    except Exception as e:
        logger.exception({"event": "rul_model_load_failed", "error": str(e)})

    # 3. Register Models
    MODELS = LoadedModels(
        anomaly_model=anomaly_model,
        anomaly_version=anomaly_version,
        rul_model=rul_model,
        rul_version=rul_version,
    )

    # 4. Setup MLflow Telemetry
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)

    yield  # Application runs and accepts traffic while yielded

    # --- Phase 7 Enhancement: Graceful Shutdown & Cleanup ---
    # Critical for K8s pod termination to prevent memory leaks in underlying nodes
    logger.info({"event": "service_shutdown", "message": "Cleaning up model resources..."})
    MODELS = None
    ANOM_SCHEMA = None
    RUL_SCHEMA = None
    ANOM_ARTIFACTS = None


app = FastAPI(
    title=settings.service_name,
    version="1.0.0",
    lifespan=lifespan,
    description="Industrial Reliability Copilot - Anomaly & RUL Inference Service",
)

# --- Phase 7 Enhancement: Production Security/CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    # In strict environments, this is constrained to the specific IP/DNS of the API Gateway
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_mw(request: Request, call_next):
    """
    Production-grade metrics middleware that captures status codes
    reliably, even if downstream application code raises unhandled exceptions.
    """
    path = request.url.path
    method = request.method

    # Default to 500 in case of an unhandled exception breaking the call_next
    status_code = 500

    with Timer(path=path, method=method):
        try:
            resp = await call_next(request)
            status_code = resp.status_code
            return resp
        finally:
            # Safely increment requests metric regardless of success or failure
            REQUESTS.labels(path, method, str(status_code)).inc()


# --- Phase 7: Kubernetes Probes ---


@app.get("/health/live", tags=["Health"])
def liveness_probe() -> dict[str, str]:
    """
    Kubernetes Liveness Probe.
    Verifies the container is running and the event loop is active.
    """
    return {"status": "alive"}


@app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
def readiness_probe(response: Response):
    """
    Kubernetes Readiness Probe.
    Ensures traffic is only routed when ALL ML models and their associated artifacts
    are successfully loaded into memory.
    """
    # Fix: A true production readiness check ensures EVERYTHING needed for prediction is loaded.
    ok_anom = bool(
        MODELS is not None
        and MODELS.anomaly_model is not None
        and ANOM_SCHEMA is not None
        and ANOM_ARTIFACTS is not None
    )

    ok_rul = bool(MODELS is not None and MODELS.rul_model is not None and RUL_SCHEMA is not None)

    if not (ok_anom and ok_rul):
        # 503 prevents K8s from routing traffic to a pod that isn't fully initialized
        response.status_code = 503

    return HealthResponse(
        status="ok" if (ok_anom and ok_rul) else "degraded",
        anomaly_model_loaded=ok_anom,
        rul_model_loaded=ok_rul,
    )


# --- Legacy Health Endpoint (Preserved for existing tests) ---
@app.get("/health", response_model=HealthResponse, tags=["Health"], include_in_schema=False)
def health():
    """Legacy health endpoint preserved strictly for backwards compatibility with tests."""
    assert MODELS is not None
    ok_anom = MODELS.anomaly_model is not None
    ok_rul = MODELS.rul_model is not None

    return HealthResponse(
        status="ok" if (ok_anom and ok_rul) else "degraded",
        anomaly_model_loaded=ok_anom,
        rul_model_loaded=ok_rul,
    )


@app.get("/metrics", tags=["Telemetry"])
def metrics():
    """Prometheus metrics scraping endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/predict/anomaly", response_model=AnomalyResponse, tags=["Inference"])
def predict_anomaly(req: SensorRequest):
    """Inference endpoint for real-time sensor anomaly detection."""
    if (
        MODELS is None
        or MODELS.anomaly_model is None
        or ANOM_SCHEMA is None
        or ANOM_ARTIFACTS is None
    ):
        raise HTTPException(status_code=503, detail="Anomaly model or artifacts not available")

    schema_id = req.schema_id or infer_schema_id(req.sensor_values)
    if schema_id not in ANOM_SCHEMA.get("schemas", {}):
        raise HTTPException(status_code=400, detail=f"Unsupported schema_id={schema_id}")

    try:
        x_final, domain_idx = preprocess_anomaly(
            schema_id, req.sensor_values, ANOM_SCHEMA, ANOM_ARTIFACTS
        )
        score, conf = anomaly_infer(MODELS.anomaly_model, x_final, domain_idx)
    except ValueError as e:
        logger.warning({"event": "anomaly_inference_validation_error", "error": str(e)})
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error({"event": "anomaly_inference_failure", "error": str(e)})
        raise HTTPException(status_code=500, detail="Internal inference error")

    return AnomalyResponse(
        timestamp=req.timestamp,
        schema_id=schema_id,
        anomaly_score=score,
        confidence=conf,
        model_version=MODELS.anomaly_version or "unknown",
    )


@app.post("/v1/predict/rul", response_model=RULResponse, tags=["Inference"])
def predict_rul(req: SensorRequest):
    """Inference endpoint for Remaining Useful Life (RUL) prediction."""
    if MODELS is None or MODELS.rul_model is None or RUL_SCHEMA is None:
        raise HTTPException(status_code=503, detail="RUL model or schema not available")

    try:
        x = preprocess_rul(req.sensor_values, RUL_SCHEMA)
        y, conf = rul_infer(MODELS.rul_model, x)
    except ValueError as e:
        logger.warning({"event": "rul_inference_validation_error", "error": str(e)})
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error({"event": "rul_inference_failure", "error": str(e)})
        raise HTTPException(status_code=500, detail="Internal inference error")

    return RULResponse(
        timestamp=req.timestamp,
        predicted_rul=y,
        confidence=conf,
        model_version=MODELS.rul_version or "unknown",
    )
