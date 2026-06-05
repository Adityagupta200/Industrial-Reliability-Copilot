import logging
import os
import time
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Aligned with Docker's internal PYTHONPATH namespace
from llm_orchestrator.llm_config import load_settings

logger = logging.getLogger(__name__)
settings = load_settings()

Base = declarative_base()


def _safe_database_target(dsn: str) -> str:
    """Return a credential-free database target for startup diagnostics."""
    try:
        url = make_url(dsn)
    except Exception:
        return "<invalid database DSN>"

    host = url.host or "<missing-host>"
    port = url.port or 5432
    database = url.database or "<missing-database>"
    return f"{url.drivername}://***@{host}:{port}/{database}"


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, unique=True, index=True)
    user_query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_contexts = Column(JSON)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_feedback_score = Column(Integer, nullable=True)  # 1 for upvote, -1 for downvote


class EvalScore(Base):
    __tablename__ = "evaluation_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, unique=True, index=True)
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)


# PRODUCTION FIX: Database-backed Job State for Distributed Kubernetes Pods
class AsyncJobState(Base):
    __tablename__ = "async_job_states"
    job_id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False, default="processing")
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RateLimitBucket(Base):
    __tablename__ = "rate_limit_buckets"

    bucket_key = Column(String, primary_key=True, index=True)
    window_start = Column(DateTime, nullable=False)
    request_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


logger.info(
    "Telemetry database target: %s",
    _safe_database_target(settings.services.incidents_db_dsn),
)

engine = create_async_engine(
    settings.services.incidents_db_dsn,
    pool_pre_ping=True,
    pool_size=int(os.getenv("TELEMETRY_DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("TELEMETRY_DB_MAX_OVERFLOW", "5")),
    pool_timeout=float(os.getenv("TELEMETRY_DB_POOL_TIMEOUT_S", "10")),
)
AsyncSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)


async def init_telemetry_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def log_interaction_async(
    query_id: str, query: str, answer: str, contexts: list, latency: float
) -> None:
    async with AsyncSessionLocal() as db:
        try:
            log_entry = QueryLog(
                query_id=query_id,
                user_query=query,
                answer=answer,
                retrieved_contexts=contexts,
                latency_ms=latency,
            )
            db.add(log_entry)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to log query {query_id}: {e}")


# --- PRODUCTION FIX: Job State Utility Functions ---


async def create_job_state(job_id: str) -> None:
    """Creates a new job in the database with 'processing' status."""
    async with AsyncSessionLocal() as db:
        job = AsyncJobState(job_id=job_id, status="processing")
        db.add(job)
        await db.commit()


async def increment_rate_limit_bucket(
    *,
    bucket_key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int, int]:
    """Atomically increment a shared rate-limit bucket.

    The orchestrator can run with multiple Uvicorn workers or Kubernetes pods. A
    process-local limiter cannot prove a single-client limit in that topology, so
    Phase 11 uses this Postgres-backed bucket as the authoritative gate.
    """
    now_epoch = int(time.time())
    window_epoch = now_epoch - (now_epoch % window_seconds)
    retry_after = max(1, window_seconds - (now_epoch - window_epoch))
    window_start = datetime.utcfromtimestamp(window_epoch)
    now = datetime.utcnow()

    statement = text(
        """
        INSERT INTO rate_limit_buckets (bucket_key, window_start, request_count, updated_at)
        VALUES (:bucket_key, :window_start, 1, :now)
        ON CONFLICT (bucket_key) DO UPDATE SET
            request_count = CASE
                WHEN rate_limit_buckets.window_start = EXCLUDED.window_start
                    THEN rate_limit_buckets.request_count + 1
                ELSE 1
            END,
            window_start = EXCLUDED.window_start,
            updated_at = EXCLUDED.updated_at
        RETURNING request_count
        """
    )

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            statement,
            {
                "bucket_key": bucket_key,
                "window_start": window_start,
                "now": now,
            },
        )
        count = int(result.scalar_one())
        await db.commit()

    return count <= limit, count, retry_after


async def update_job_state(
    job_id: str, status: str, result: dict | None = None, error: str | None = None
) -> None:
    """Updates an existing job's status, result payload, or error message."""
    async with AsyncSessionLocal() as db:
        values: dict = {"status": status, "updated_at": datetime.utcnow()}
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error

        await db.execute(
            update(AsyncJobState).where(AsyncJobState.job_id == job_id).values(**values)
        )
        await db.commit()


async def get_job_state(job_id: str) -> dict | None:
    """Retrieves the current job state from the database."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(AsyncJobState).filter(AsyncJobState.job_id == job_id))
        job = res.scalars().first()
        if job:
            return {
                "job_id": job.job_id,
                "status": job.status,
                "result": job.result,
                "error": job.error,
            }
        return None
