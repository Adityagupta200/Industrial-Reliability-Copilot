import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

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


# Initialize engine synchronously for logging utility
# Note: In a true enterprise environment, tables are created via Alembic migrations,
# but for local microservice booting, create_all() is an acceptable standard.
engine = create_engine(settings.services.incidents_db_dsn)
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def log_interaction_sync(query_id: str, query: str, answer: str, contexts: list, latency: float):
    """
    Synchronous logging utility.
    Designed to be executed safely via FastAPI's BackgroundTasks so it
    never blocks the HTTP response returning to the user.
    """
    db = SessionLocal()
    try:
        log_entry = QueryLog(
            query_id=query_id,
            user_query=query,
            answer=answer,
            retrieved_contexts=contexts,
            latency_ms=latency,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log query {query_id}: {e}")
    finally:
        db.close()
