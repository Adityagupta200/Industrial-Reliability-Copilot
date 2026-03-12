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

# Copy the main requirements.txt which has LangChain, Presidio, spaCy, and slowapi
COPY requirements.txt /app/requirements.txt

# FIX: Added --extra-index-url to force CPU-only torch to prevent Docker OOM / EOF crashes
RUN python -m pip install --no-cache-dir --retries 5 --timeout 200 \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.txt

# --- PHASE 5: Use lightweight model to prevent OOM on constrained hardware ---
RUN python -m spacy download en_core_web_sm

# Copy the source code
COPY src/ /app/src/

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
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Copy virtual environment and code from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

USER appuser
EXPOSE 8000

# Start the Orchestrator service
CMD ["uvicorn", "llm_orchestrator.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]