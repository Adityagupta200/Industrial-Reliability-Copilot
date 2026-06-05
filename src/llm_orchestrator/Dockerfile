# syntax=docker/dockerfile:1.6

########################
# Builder Stage
########################
FROM python:3.11-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    VENV_PATH=/opt/venv

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Upgrade base packaging tools
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# Copy dependency files
COPY src/llm_orchestrator/requirements.txt /app/requirements.txt

# Install only the orchestrator runtime stack. Ragas/LangChain evaluation
# packages stay in requirements-dev.txt and are not shipped in this image.
RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    -r /app/requirements.txt

# Download spaCy model explicitly into the virtual environment
RUN python -m spacy download en_core_web_sm

########################
# Runtime Stage
########################
FROM python:3.11-slim AS runtime

# Production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# Install runtime utilities (tini for signal handling, curl for local health checks)
# Tini ensures graceful shutdown of FastAPI workers upon receiving SIGTERM from Kubernetes
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Runtime containers never unpack or build Python wheels. Remove base-image
# packaging tools from the system interpreter so inactive tooling cannot expand
# the production attack surface or trigger fixed-version CVEs.
RUN rm -rf \
      /usr/local/lib/python3.11/site-packages/_distutils_hack \
      /usr/local/lib/python3.11/site-packages/pkg_resources \
      /usr/local/lib/python3.11/site-packages/setuptools* \
      /usr/local/lib/python3.11/site-packages/wheel* \
      /usr/local/bin/wheel

# Create a strict non-root user and group with explicit explicit numeric IDs
# This ensures compatibility with strict Kubernetes SecurityContext policies
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid 10001 --create-home appuser

# Copy virtual environment from builder using single-layer chown
COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

# Copy application source code using single-layer chown
COPY --chown=10001:10001 src/ /app/src/

# Switch to the non-root user for all subsequent commands and container execution
USER appuser:appgroup

EXPOSE 8000

# Local Docker Compose Healthcheck
# Note: Kubernetes will ignore this and use the Liveness/Readiness probes defined in your manifest
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Use Tini as the entrypoint for proper PID 1 signal forwarding
ENTRYPOINT ["/usr/bin/tini", "--"]
ENV HF_HOME=/tmp/huggingface
# Start the Orchestrator service. 
# Explicitly setting --workers 1 because Kubernetes HPA handles concurrency scaling in production.
CMD ["uvicorn", "llm_orchestrator.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info", "--no-access-log", "--timeout-keep-alive", "75", "--timeout-graceful-shutdown", "30"]
