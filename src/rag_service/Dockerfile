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

RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

COPY requirements-rag.txt /app/requirements-rag.txt

# Explicitly cap numpy below 2.0.0 to maintain ABI compatibility with PyTorch 2.1.0
RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "numpy>=1.26.0,<2.0.0" \
    "typing_extensions>=4.10" \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    torchaudio==2.1.0+cpu

# Install remaining RAG dependencies
RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements-rag.txt

COPY src/ /app/src/

# Fail-fast builder tests to ensure core libraries loaded correctly
RUN python -c "import numpy, typing_extensions; print('builder deps ok')"
RUN python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service; print('builder rag_service ok')"
RUN python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service.main; print('builder rag_service.main ok')"

########################
# Runtime Stage
########################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src

WORKDIR /app

# Install runtime tools: curl (for K8s health checks), libgomp1 (for vector search libs), tini (for PID 1 signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      libgomp1 \
      tini \
    && rm -rf /var/lib/apt/lists/*

# --- Phase 7 Production Patches ---

# 1. Prevent running as root using explicit numeric IDs for strict K8s SecurityContexts
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid 10001 --create-home appuser

# FIXED: Added /app/data/hf_cache to pre-create the directory with non-root ownership.
# When Docker mounts the irc_hf_cache volume, it will inherit these 10001:10001 permissions.
RUN install -d -m 0755 -o 10001 -g 10001 \
      /app/data/raw/manuals \
      /app/data/raw/procedures \
      /app/data/raw/incidents \
      /app/data/processed/texts \
      /app/data/processed/manifest \
      /app/data/hf_cache

# 2. Copy code and environment via single-layer chown to drastically reduce image bloat
COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /app/src /app/src

# Switch to non-root user before running final validation to prevent root-owned pycache generation
USER appuser:appgroup

# Validate environment integrity securely
RUN python -m pip check \
 && python -c "import numpy, typing_extensions; print('runtime deps ok')" \
 && python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service; print('runtime rag_service ok')" \
 && python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service.main; print('runtime rag_service.main ok')"

EXPOSE 8002

# 3. Native Container Healthcheck (Fallback for Docker Compose; K8s relies on manifests)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8002/health/live || exit 1

# 4. Use Tini to handle Kubernetes SIGTERM scaling gracefully
ENTRYPOINT ["/usr/bin/tini", "--"]

# FIXED: Environment variable to explicitly direct HuggingFace downloads to the new volume
ENV HF_HOME=/app/data/hf_cache

# 5. Lock workers to 1 (Rely on Kubernetes HPA for scaling) and set log-level to info for production
CMD ["uvicorn", "rag_service.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--log-level", "info", "--timeout-keep-alive", "600"]