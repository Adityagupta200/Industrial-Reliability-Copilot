# syntax=docker/dockerfile:1.6

########################
# Builder
########################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

COPY requirements-rag.txt /app/requirements-rag.txt

RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "numpy>=1.26" \
    "typing_extensions>=4.10" \
    torch==2.10.0+cpu \
    torchvision==0.25.0+cpu \
    torchaudio==2.10.0+cpu

RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements-rag.txt

COPY src/ /app/src/

RUN python -c "import numpy, typing_extensions; print('builder deps ok')"
RUN python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service; print('builder rag_service ok')"
RUN python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service.main; print('builder rag_service.main ok')"

########################
# Runtime
########################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Phase 7 Production Patches ---

# 1. Prevent running as root (Standardized Security Context)
RUN addgroup --system appgroup && adduser --system --group appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src /app/src

# Create necessary directories for local volume mounting and assign secure ownership
RUN mkdir -p \
      /app/data/raw/manuals \
      /app/data/raw/procedures \
      /app/data/raw/incidents \
      /app/data/processed/texts \
      /app/data/processed/manifest \
 && chown -R appuser:appgroup /app

RUN python -m pip check \
 && python -c "import numpy, typing_extensions; print('runtime deps ok')" \
 && python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service; print('runtime rag_service ok')" \
 && python -c "import sys; sys.path.insert(0, '/app/src'); import rag_service.main; print('runtime rag_service.main ok')"

USER appuser
EXPOSE 8002

# 2. Native Container Healthcheck (Utilizes the new /health/live endpoint mapped to RAG port 8002)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8002/health/live || exit 1

CMD ["uvicorn", "rag_service.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8002", "--log-level", "debug"]