from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag_service.retrieval import BM25KeywordRetriever, HybridRetriever, SemanticRetriever
from rag_service.retrieval.types import Document, RetrievalFilters

logger = logging.getLogger(__name__)
router = APIRouter()


class RetrievalFiltersPayload(BaseModel):
    equipment_id: str | None = None
    severity: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    plant_id: str | None = None 
    user_role: str | None = None 


class SemanticRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    filters: RetrievalFiltersPayload = Field(default_factory=RetrievalFiltersPayload)


class KeywordRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    filters: RetrievalFiltersPayload = Field(default_factory=RetrievalFiltersPayload)


class HybridRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    semantic_k: int | None = Field(default=None, ge=1, le=100)
    keyword_k: int | None = Field(default=None, ge=1, le=100)
    out_k: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int | None = Field(default=None, ge=1, le=500)
    filters: RetrievalFiltersPayload = Field(default_factory=RetrievalFiltersPayload)


class DocumentResponse(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str


class RetrieveResponse(BaseModel):
    documents: list[DocumentResponse]
    count: int
    latency_ms: float


def _to_filters(payload: RetrievalFiltersPayload) -> RetrievalFilters | None:
    if (
        payload.equipment_id is None
        and payload.severity is None
        and payload.date_from is None
        and payload.date_to is None
        and payload.plant_id is None
        and payload.user_role is None
    ):
        return None

    return RetrievalFilters(
        equipment_id=payload.equipment_id,
        severity=payload.severity,
        date_from=payload.date_from,
        date_to=payload.date_to,
        plant_id=payload.plant_id,
        user_role=payload.user_role, 
    )


def _to_document_response(doc: Document) -> DocumentResponse:
    doc_text = doc.text
    metadata = dict(doc.metadata) 

    # PRODUCTION FIX: Guarantee a unified 'file_name' key for Orchestrator citation mapping
    if "file_name" not in metadata:
        metadata["file_name"] = metadata.get("source", f"unknown_source_{doc.id[:8]}")

    # --- PHASE 5: Data Freshness Check ---
    last_updated_str = metadata.get("last_updated")

    try:
        if last_updated_str:
            last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)
        else:
            last_updated = datetime(2000, 1, 1, tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        if (now - last_updated).days > 730:
            doc_text = f"(outdated) {doc_text}"
            metadata["is_outdated"] = True

    except ValueError:
        logger.warning(
            f"Could not parse last_updated date format for doc {doc.id}: {last_updated_str}"
        )

    return DocumentResponse(
        id=doc.id,
        text=doc_text,
        metadata=metadata,
        score=float(doc.score),
        source=str(doc.source),
    )


def _ensure_runtime(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = {
            "semantic_retriever": None,
            "keyword_retriever": None,
            "hybrid_retriever": None,
            "semantic_lock": Lock(),
            "keyword_lock": Lock(),
            "hybrid_lock": Lock(),
        }
        request.app.state.runtime = runtime
    return runtime


def _get_semantic_retriever(request: Request) -> SemanticRetriever:
    runtime = _ensure_runtime(request)
    retriever = runtime.get("semantic_retriever")
    if retriever is not None:
        return retriever

    lock: Lock = runtime["semantic_lock"]
    with lock:
        retriever = runtime.get("semantic_retriever")
        if retriever is None:
            logger.info("Initializing semantic retriever lazily")
            retriever = SemanticRetriever()
            runtime["semantic_retriever"] = retriever

    return retriever


def _get_keyword_retriever(request: Request) -> BM25KeywordRetriever:
    runtime = _ensure_runtime(request)
    retriever = runtime.get("keyword_retriever")
    if retriever is not None:
        return retriever

    lock: Lock = runtime["keyword_lock"]
    with lock:
        retriever = runtime.get("keyword_retriever")
        if retriever is None:
            logger.info("Initializing keyword retriever lazily")
            retriever = BM25KeywordRetriever()
            retriever.build_or_load(force_rebuild=False)
            runtime["keyword_retriever"] = retriever

    return retriever


def _get_hybrid_retriever(request: Request) -> HybridRetriever:
    runtime = _ensure_runtime(request)
    retriever = runtime.get("hybrid_retriever")
    if retriever is not None:
        return retriever

    lock: Lock = runtime["hybrid_lock"]
    with lock:
        retriever = runtime.get("hybrid_retriever")
        if retriever is None:
            logger.info("Initializing hybrid retriever lazily")
            retriever = HybridRetriever(
                semantic=_get_semantic_retriever(request),
                keyword=_get_keyword_retriever(request),
            )
            runtime["hybrid_retriever"] = retriever

    return retriever


@router.post("/semantic", response_model=RetrieveResponse)
async def retrieve_semantic(
    payload: SemanticRetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    filters = _to_filters(payload.filters)
    t0 = time.perf_counter()

    try:
        retriever = await run_in_threadpool(_get_semantic_retriever, request)
        docs = await run_in_threadpool(
            retriever.semantic_search,
            payload.query,
            payload.k,
            filters=filters,
        )
    except Exception as exc:
        logger.exception("Semantic retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    items = [_to_document_response(doc) for doc in docs]

    return RetrieveResponse(
        documents=items,
        count=len(items),
        latency_ms=latency_ms,
    )


@router.post("/keyword", response_model=RetrieveResponse)
async def retrieve_keyword(
    payload: KeywordRetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    filters = _to_filters(payload.filters)
    t0 = time.perf_counter()

    try:
        retriever = await run_in_threadpool(_get_keyword_retriever, request)
        docs = await run_in_threadpool(
            retriever.keyword_search,
            payload.query,
            payload.k,
            filters=filters,
        )
    except Exception as exc:
        logger.exception("Keyword retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Keyword retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    items = [_to_document_response(doc) for doc in docs]

    return RetrieveResponse(
        documents=items,
        count=len(items),
        latency_ms=latency_ms,
    )


@router.post("/hybrid", response_model=RetrieveResponse)
async def retrieve_hybrid(
    payload: HybridRetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    filters = _to_filters(payload.filters)
    t0 = time.perf_counter()

    try:
        retriever = await run_in_threadpool(_get_hybrid_retriever, request)
        docs = await run_in_threadpool(
            retriever.hybrid_search,
            payload.query,
            filters=filters,
            semantic_k=payload.semantic_k,
            keyword_k=payload.keyword_k,
            out_k=payload.out_k,
            rrf_k=payload.rrf_k,
        )
    except Exception as exc:
        logger.exception("Hybrid retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    items = [_to_document_response(doc) for doc in docs]

    return RetrieveResponse(
        documents=items,
        count=len(items),
        latency_ms=latency_ms,
    )