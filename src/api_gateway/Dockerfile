# syntax=docker/dockerfile:1.6

########################
# Builder Stage
########################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

# The Gateway relies mostly on FastAPI, httpx, and prometheus_client
RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt /app/requirements.txt

# PRODUCTION FIX: Added index-url and extra-index-url to allow the gateway 
# to securely resolve the CPU-only PyTorch wheel and prevent ResolutionImpossible errors.
RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.txt

########################
# Runtime Stage
########################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      tini \
    && rm -rf /var/lib/apt/lists/*

# --- Phase 7 Production Patches ---

# Explicit numeric UID/GID
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid 10001 --create-home appuser

# Single-layer chown copy
COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --chown=10001:10001 src/ /app/src/

# Switch to non-root user
USER appuser:appgroup

# The gateway standard port
EXPOSE 8000

# K8s Healthcheck configuration
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the API Gateway
CMD ["uvicorn", "api_gateway.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info", "--timeout-keep-alive", "600"]