from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
import httpx
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_gateway")

app = FastAPI(title="Industrial Reliability Copilot - API Gateway", version="1.0.0")

# Internal DNS routing via Docker/Kubernetes networking
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://llm-orchestrator:8000")


@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    """Validates the gateway container is running."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    """Validates the gateway is ready AND the downstream orchestrator is reachable."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{ORCHESTRATOR_URL}/health/ready")
            if resp.status_code == 200:
                return {"status": "ready"}
    except Exception as e:
        logger.warning(f"Downstream orchestrator not ready: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "degraded"}
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "degraded"}
    )


@app.post("/query")
async def route_query(request: Request):
    """Reverse proxies the query to the LLM Orchestrator."""
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{ORCHESTRATOR_URL}/query", json=body)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Gateway Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Gateway Error")
