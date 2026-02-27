from __future__ import annotations

from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException, Request, Response
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

MODELS: LoadedModels | None = None
ANOM_SCHEMA = None
RUL_SCHEMA = None
ANOM_ARTIFACTS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, ANOM_SCHEMA, RUL_SCHEMA, ANOM_ARTIFACTS
    setup_logging(settings.log_level)

    anomaly_model = None
    anomaly_version = None
    rul_model = None
    rul_version = None

    try:
        anomaly_model, anomaly_version = load_anomaly_model()
        ANOM_SCHEMA = load_schema(settings.anomaly_schema_path)
        ANOM_ARTIFACTS = load_anomaly_artifacts(settings.anomaly_preprocess_path)
        logger.info({"event": "anomaly_model_loaded", "version": anomaly_version})
    except Exception as e:
        logger.exception({"event": "anomaly_model_load_failed", "error": str(e)})

    try:
        rul_model, rul_version, RUL_SCHEMA = load_rul_model()
        logger.info({"event": "rul_model_loaded", "version": rul_version})
    except Exception as e:
        logger.exception({"event": "rul_model_load_failed", "error": str(e)})

    MODELS = LoadedModels(
        anomaly_model=anomaly_model,
        anomaly_version=anomaly_version,
        rul_model=rul_model,
        rul_version=rul_version,
    )

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)

    yield


app = FastAPI(title=settings.service_name, version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def prometheus_mw(request: Request, call_next):
    path = request.url.path
    method = request.method
    with Timer(path=path, method=method):
        resp = await call_next(request)
    REQUESTS.labels(path, method, str(resp.status_code)).inc()
    return resp


@app.get("/health", response_model=HealthResponse)
def health():
    assert MODELS is not None
    ok_anom = MODELS.anomaly_model is not None
    ok_rul = MODELS.rul_model is not None
    return HealthResponse(
        status="ok" if (ok_anom and ok_rul) else "degraded",
        anomaly_model_loaded=ok_anom,
        rul_model_loaded=ok_rul,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(req: SensorRequest):
    assert MODELS is not None
    if MODELS.anomaly_model is None or ANOM_SCHEMA is None or ANOM_ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="Anomaly model not available")

    schema_id = req.schema_id or infer_schema_id(req.sensor_values)

    if schema_id not in ANOM_SCHEMA["schemas"]:
        raise HTTPException(status_code=400, detail=f"Unsupported schema_id={schema_id}")

    try:
        x_final, domain_idx = preprocess_anomaly(
            schema_id, req.sensor_values, ANOM_SCHEMA, ANOM_ARTIFACTS
        )
        score, conf = anomaly_infer(MODELS.anomaly_model, x_final, domain_idx)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return AnomalyResponse(
        timestamp=req.timestamp,
        schema_id=schema_id,
        anomaly_score=score,
        confidence=conf,
        model_version=MODELS.anomaly_version or "unknown",
    )


@app.post("/v1/predict/rul", response_model=RULResponse)
def predict_rul(req: SensorRequest):
    assert MODELS is not None
    if MODELS.rul_model is None or RUL_SCHEMA is None:
        raise HTTPException(status_code=503, detail="RUL model not available")

    try:
        x = preprocess_rul(req.sensor_values, RUL_SCHEMA)
        y, conf = rul_infer(MODELS.rul_model, x)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return RULResponse(
        timestamp=req.timestamp,
        predicted_rul=y,
        confidence=conf,
        model_version=MODELS.rul_version or "unknown",
    )
