from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from rag_service.core.config import settings
from rag_service.db.init_db import init_db
from rag_service.db.ingest_incidents import ingest_incidents
from rag_service.db.session import engine
from rag_service.ingestion.pipeline import ingest_all
from rag_service.vectorstore.qdrant_store import QdrantStore


def _log(message: str) -> None:
    print(f"[phase11-bootstrap] {message}", flush=True)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


def _incident_count() -> int:
    with engine.connect() as conn:
        return int(conn.execute(text("select count(*) from incidents")).scalar_one())


def _qdrant_counts() -> dict[str, int]:
    store = QdrantStore()
    return {
        settings.qdrant_collection_docs: store.count(settings.qdrant_collection_docs),
        settings.qdrant_collection_procedures: store.count(settings.qdrant_collection_procedures),
    }


def _validate_bootstrap(report: dict[str, Any]) -> list[str]:
    min_doc_points = _int_env("PHASE11_MIN_DOC_POINTS", 50)
    min_procedure_points = _int_env("PHASE11_MIN_PROCEDURE_POINTS", 5)
    min_incident_rows = _int_env("PHASE11_MIN_INCIDENT_ROWS", 100)

    counts = report["qdrant_counts"]
    failures: list[str] = []
    if counts.get(settings.qdrant_collection_docs, 0) < min_doc_points:
        failures.append(
            f"{settings.qdrant_collection_docs} has "
            f"{counts.get(settings.qdrant_collection_docs, 0)} points; "
            f"expected >= {min_doc_points}"
        )
    if counts.get(settings.qdrant_collection_procedures, 0) < min_procedure_points:
        failures.append(
            f"{settings.qdrant_collection_procedures} has "
            f"{counts.get(settings.qdrant_collection_procedures, 0)} points; "
            f"expected >= {min_procedure_points}"
        )
    if int(report["incident_rows_total"]) < min_incident_rows:
        failures.append(
            f"incidents has {report['incident_rows_total']} rows; expected >= {min_incident_rows}"
        )
    return failures


def run_bootstrap() -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    _log("Initializing PostgreSQL schema")
    init_db()

    _log("Ingesting deterministic incident seed data")
    inserted_incidents = ingest_incidents()

    _log("Ingesting manuals and procedures into Qdrant")
    ingestion_stats = ingest_all(force=True)
    incident_rows_total = _incident_count()
    qdrant_counts = _qdrant_counts()

    report: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "postgres_dsn_configured": bool(settings.postgres_dsn),
        "qdrant_url": settings.qdrant_url,
        "embedding_provider": settings.embedding_provider,
        "incident_rows_inserted": inserted_incidents,
        "incident_rows_total": incident_rows_total,
        "ingestion_stats": ingestion_stats,
        "qdrant_counts": qdrant_counts,
    }
    _log(
        "Validating bootstrap counts: "
        f"incident_rows_total={incident_rows_total}, qdrant_counts={qdrant_counts}"
    )
    failures = _validate_bootstrap(report)
    report["passed"] = not failures
    report["failed_checks"] = failures
    return report


def main() -> int:
    report = run_bootstrap()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
