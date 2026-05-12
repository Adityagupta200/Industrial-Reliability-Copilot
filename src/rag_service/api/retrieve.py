from __future__ import annotations

import logging
import os
import time
import dataclasses
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag_service.retrieval import BM25KeywordRetriever, HybridRetriever, SemanticRetriever
from rag_service.retrieval.types import Document, RetrievalFilters
from rag_service.retrieval.qdrant_backend import QdrantBackend, QdrantSettings

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
    title: str | None = None


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

    bad_tags = {"hybrid", "semantic", "keyword", "unknown"}
    
    for key in ["source", "file_name"]:
        if str(metadata.get(key, "")).lower() in bad_tags:
            metadata.pop(key, None)

    if "file_name" not in metadata:
        raw_source = metadata.get("source") or getattr(doc, "source", None)
        if raw_source and str(raw_source).strip() and str(raw_source).lower() not in bad_tags:
            metadata["file_name"] = os.path.basename(str(raw_source))
        else:
            eq_id = metadata.get("equipment_id", "equipment").lower()
            metadata["file_name"] = f"{eq_id}_maintenance_manual.pdf"

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
        title=metadata.get("file_name", "Untitled")
    )


def _ensure_runtime(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = {
            "semantic_retriever": None,
            "keyword_retriever": None,
            "hybrid_retriever": None,
            "procedure_retriever": None, 
            "semantic_lock": Lock(),
            "keyword_lock": Lock(),
            "hybrid_lock": Lock(),
            "procedure_lock": Lock(), 
        }
        request.app.state.runtime = runtime
    return runtime


def _get_semantic_retriever(request: Request) -> SemanticRetriever | None:
    runtime = _ensure_runtime(request)
    
    if "semantic_retriever" in runtime and runtime["semantic_retriever"] is not None:
        return runtime["semantic_retriever"]

    lock: Lock = runtime["semantic_lock"]
    with lock:
        if runtime.get("semantic_retriever") is None:
            logger.info("Initializing semantic retriever lazily")
            try:
                retriever = SemanticRetriever()
                runtime["semantic_retriever"] = retriever
            except Exception as e:
                # PRODUCTION FIX: Graceful Degradation
                logger.error(f"Graceful Degradation: SemanticRetriever initialization failed: {e}. Semantic search disabled.")
                return None

    return runtime.get("semantic_retriever")


def _get_keyword_retriever(request: Request) -> BM25KeywordRetriever | None:
    runtime = _ensure_runtime(request)
    
    if "keyword_retriever" in runtime and runtime["keyword_retriever"] is not None:
        return runtime["keyword_retriever"]

    lock: Lock = runtime["keyword_lock"]
    with lock:
        if runtime.get("keyword_retriever") is None:
            logger.info("Initializing keyword retriever lazily")
            try:
                retriever = BM25KeywordRetriever()
                retriever.build_or_load(force_rebuild=False)
                runtime["keyword_retriever"] = retriever
            except Exception as e:
                logger.error(f"Graceful Degradation: KeywordRetriever initialization failed: {e}")
                return None

    return runtime.get("keyword_retriever")


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
            # If a retriever fails, it returns None. HybridRetriever gracefully handles None.
            retriever = HybridRetriever(
                semantic=_get_semantic_retriever(request),
                keyword=_get_keyword_retriever(request),
            )
            runtime["hybrid_retriever"] = retriever

    return retriever


def _get_procedure_retriever(request: Request) -> SemanticRetriever | None:
    runtime = _ensure_runtime(request)
    
    if "procedure_retriever" in runtime and runtime["procedure_retriever"] is not None:
        return runtime["procedure_retriever"]

    lock: Lock = runtime["procedure_lock"]
    with lock:
        if runtime.get("procedure_retriever") is None:
            logger.info("Initializing procedure retriever lazily")
            try:
                base_settings = QdrantSettings.from_env()
                proc_settings = dataclasses.replace(base_settings, collection="procedures")
                backend = QdrantBackend(settings=proc_settings)
                
                retriever = SemanticRetriever(qdrant=backend)
                runtime["procedure_retriever"] = retriever
            except Exception as e:
                logger.error(f"Graceful Degradation: Procedure Retriever initialization failed: {e}")
                return None

    return runtime.get("procedure_retriever")


@router.post("/procedures", response_model=RetrieveResponse)
async def retrieve_procedures(
    payload: SemanticRetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    filters = _to_filters(payload.filters)
    t0 = time.perf_counter()

    try:
        retriever = await run_in_threadpool(_get_procedure_retriever, request)
        if retriever is None:
            raise RuntimeError("Procedure Retriever is unavailable due to model initialization failure.")
            
        docs = await run_in_threadpool(
            retriever.semantic_search,
            payload.query,
            payload.k,
            filters=filters,
        )
    except Exception as exc:
        logger.exception("Procedure retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Procedure retrieval failed: {type(exc).__name__}: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    items = [_to_document_response(doc) for doc in docs]

    return RetrieveResponse(
        documents=items,
        count=len(items),
        latency_ms=latency_ms,
    )


@router.post("/semantic", response_model=RetrieveResponse)
async def retrieve_semantic(
    payload: SemanticRetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    filters = _to_filters(payload.filters)
    t0 = time.perf_counter()

    try:
        retriever = await run_in_threadpool(_get_semantic_retriever, request)
        if retriever is None:
            raise RuntimeError("SemanticRetriever is unavailable due to model initialization failure.")
            
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
        if retriever is None:
            raise RuntimeError("Keyword Retriever is unavailable.")
            
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