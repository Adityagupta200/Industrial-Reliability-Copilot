import logging
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Aligned with Docker's internal PYTHONPATH namespace
from llm_orchestrator.llm_config import load_settings

logger = logging.getLogger(__name__)
settings = load_settings()

Base = declarative_base()


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


engine = create_async_engine(settings.services.incidents_db_dsn, pool_pre_ping=True)
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


async def update_job_state(
    job_id: str, status: str, result: dict | None = None, error: str | None = None
) -> None:
    """Updates an existing job's status, result payload, or error message."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(AsyncJobState).filter(AsyncJobState.job_id == job_id))
        job = res.scalars().first()
        if job:
            job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
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
