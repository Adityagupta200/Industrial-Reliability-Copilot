# syntax=docker/dockerfile:1.6

########################
# Builder Stage
########################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    VENV_PATH=/opt/venv

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# Explicitly target the service-specific requirements file
COPY src/anomaly_service/requirements.txt /app/requirements.txt

# Force CPU-only PyTorch to prevent massive image bloat and OOM crashes
RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu

# Install dependencies strictly from the synchronized requirements file
# The extra-index-url is included again so pip maps torch==2.1.0 to the cpu wheel
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

# Install runtime utilities: curl (K8s health checks), tini (PID 1 signals), libgomp1 (Required by XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      tini \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Explicit numeric UID/GID for strict Kubernetes SecurityContext policies
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid 10001 --create-home appuser

# Copy VENV and Code with single-layer chown to avoid layer bloat
COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --chown=10001:10001 src/ /app/src/

# Switch to non-root user
USER appuser:appgroup

EXPOSE 8001

# Native Container Healthcheck mapping to the FastAPI live probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1

# Tini handles graceful SIGTERM shutdowns from Kubernetes
ENTRYPOINT ["/usr/bin/tini", "--"]

# Lock workers to 1 (HPA will handle scale-out)
CMD ["uvicorn", "anomaly_service.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--log-level", "info", "--timeout-keep-alive", "75", "--timeout-graceful-shutdown", "30"]
