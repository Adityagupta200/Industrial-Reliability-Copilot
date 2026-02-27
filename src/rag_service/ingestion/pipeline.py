from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from rag_service.core.config import settings
from rag_service.embeddings import get_embedding_provider
from rag_service.vectorstore.qdrant_store import QdrantStore, VectorPoint
from rag_service.ingestion.hashing import sha256_file
from rag_service.ingestion.manifest import Manifest
from rag_service.ingestion.pdf_extractor import extract_pdf_text, remove_common_headers_footers
from rag_service.ingestion.markdown_loader import load_markdown
from rag_service.ingestion.cleaning import clean_text
from rag_service.ingestion.chunking import chunk_text


def _write_processed_text(source_id: str, obj: dict[str, Any]) -> None:
    out_dir = Path(settings.processed_texts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_id}.json"
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_all() -> dict[str, Any]:
    manifest = Manifest.load()

    embedder = get_embedding_provider()
    store = QdrantStore()

    # Create collections with correct vector size (derived from chosen embedding provider)
    dim = embedder.dim()
    store.ensure_collection(settings.qdrant_collection_docs, vector_size=dim)
    store.ensure_collection(settings.qdrant_collection_procedures, vector_size=dim)

    stats = {
        "processed_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "chunks_created": 0,
        "points_upserted": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # --- Manuals (PDF) ---
    manuals_dir = Path(settings.raw_manuals_dir)
    pdfs = sorted(manuals_dir.glob("**/*.pdf"))
    for pdf in pdfs:
        key = f"manual::{pdf.as_posix()}"
        sha = sha256_file(pdf)
        if manifest.is_unchanged(key, sha):
            stats["skipped_files"] += 1
            continue

        try:
            pages = extract_pdf_text(pdf)
            pages = remove_common_headers_footers(pages)
            full_text = clean_text("\n\n".join([p.text for p in pages]))

            source_id = f"manual__{pdf.stem}"
            _write_processed_text(
                source_id,
                {
                    "source_file": pdf.as_posix(),
                    "doc_type": "manual",
                    "pages": [{"page_number": p.page_number, "text": p.text} for p in pages],
                },
            )

            chunks = chunk_text(
                full_text,
                source_id=source_id,
                doc_type="manual",
                extra_meta={
                    "source_file": pdf.name,
                    "path": pdf.as_posix(),
                },
            )

            # Embed + upsert in batches
            texts = [c.text for c in chunks]
            for i in range(0, len(texts), settings.embed_batch_size):
                batch_chunks = chunks[i : i + settings.embed_batch_size]
                batch_texts = [c.text for c in batch_chunks]
                vecs = embedder.embed_texts(batch_texts)

                points: list[VectorPoint] = []
                for c, v in zip(batch_chunks, vecs):
                    payload = dict(c.metadata)
                    payload["text"] = c.text
                    points.append(VectorPoint(id=c.chunk_id, vector=v, payload=payload))

                # upsert in smaller batches to reduce request size
                for j in range(0, len(points), settings.upsert_batch_size):
                    store.upsert(
                        settings.qdrant_collection_docs, points[j : j + settings.upsert_batch_size]
                    )
                    stats["points_upserted"] += len(points[j : j + settings.upsert_batch_size])

            stats["processed_files"] += 1
            stats["chunks_created"] += len(chunks)
            manifest.mark(key, sha, status="ok")
        except Exception as e:
            stats["failed_files"] += 1
            manifest.mark(key, sha, status="failed", detail=str(e))

    # --- Procedures (Markdown) ---
    proc_dir = Path(settings.raw_procedures_dir)
    mds = sorted([*proc_dir.glob("**/*.md"), *proc_dir.glob("**/*.markdown")])
    for md in mds:
        key = f"procedure::{md.as_posix()}"
        sha = sha256_file(md)
        if manifest.is_unchanged(key, sha):
            stats["skipped_files"] += 1
            continue

        try:
            text = clean_text(load_markdown(md))
            source_id = f"procedure__{md.stem}"

            _write_processed_text(
                source_id,
                {
                    "source_file": md.as_posix(),
                    "doc_type": "procedure",
                    "text": text,
                },
            )

            # For procedures, keep steps together by using larger chunks and low overlap
            chunks = chunk_text(
                text,
                source_id=source_id,
                doc_type="procedure",
                extra_meta={
                    "source_file": md.name,
                    "path": md.as_posix(),
                },
            )

            texts = [c.text for c in chunks]
            for i in range(0, len(texts), settings.embed_batch_size):
                batch_chunks = chunks[i : i + settings.embed_batch_size]
                vecs = embedder.embed_texts([c.text for c in batch_chunks])

                points: list[VectorPoint] = []
                for c, v in zip(batch_chunks, vecs):
                    payload = dict(c.metadata)
                    payload["text"] = c.text
                    points.append(VectorPoint(id=c.chunk_id, vector=v, payload=payload))

                for j in range(0, len(points), settings.upsert_batch_size):
                    store.upsert(
                        settings.qdrant_collection_procedures,
                        points[j : j + settings.upsert_batch_size],
                    )
                    stats["points_upserted"] += len(points[j : j + settings.upsert_batch_size])

            stats["processed_files"] += 1
            stats["chunks_created"] += len(chunks)
            manifest.mark(key, sha, status="ok")
        except Exception as e:
            stats["failed_files"] += 1
            manifest.mark(key, sha, status="failed", detail=str(e))

    manifest.save()
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["qdrant_counts"] = {
        settings.qdrant_collection_docs: store.count(settings.qdrant_collection_docs),
        settings.qdrant_collection_procedures: store.count(settings.qdrant_collection_procedures),
    }
    return stats


if __name__ == "__main__":
    out = ingest_all()
    print(json.dumps(out, indent=2))
