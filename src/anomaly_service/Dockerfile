# Builder: build wheelhouse

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# We'll build in an arbitrary location; not important for imports
WORKDIR /build

# Build deps only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is in the build context root (src/anomaly_service)
COPY requirements.txt .

# Build wheels for all deps (incl. torch CPU index)
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip wheel \
      --wheel-dir /wheels \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# Runtime: minimal image

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Make the package importable: /srv is the parent of /srv/anomaly_service
    PYTHONPATH=/srv

# Runtime OS deps:
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Install deps from wheelhouse
WORKDIR /srv

COPY --from=builder /wheels /wheels
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && python -m pip check \
 && rm -rf /wheels

# Copy the service code as a *package directory*
COPY . /srv/anomaly_service

RUN chown -R appuser:appuser /srv
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

# Use Gunicorn, import via package module path, and chdir to /srv for consistency
CMD ["gunicorn","anomaly_service.main:app","--bind","0.0.0.0:8001","--worker-class","uvicorn.workers.UvicornWorker","--workers","2","--timeout","60","--keep-alive","5","--chdir","/srv"]

