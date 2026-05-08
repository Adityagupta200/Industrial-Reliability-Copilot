from __future__ import annotations

import json
import re
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

def _log(msg: str) -> None:
    print(msg, flush=True)

def _write_processed_text(source_id: str, obj: dict[str, Any]) -> None:
    out_dir = Path(settings.processed_texts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_id}.json"
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _extract_equipment_id(filename: str) -> str | None:
    match = re.search(r'(pump_P-\d+|motor_M-\d+|compressor_C-\d+|turbofan_TF-\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# PRODUCTION FIX: Added 'force' parameter to bypass manifest checks
def ingest_all(force: bool = False) -> dict[str, Any]:
    manifest = Manifest.load()

    _log("Initializing embedding provider...")
    embedder = get_embedding_provider()
    _log("Initializing Qdrant store...")
    store = QdrantStore()

    dim = embedder.dim()
    _log(f"Embedding dim={dim}. Ensuring collections exist...")
    store.ensure_collection(settings.qdrant_collection_docs, vector_size=dim)
    store.ensure_collection(settings.qdrant_collection_procedures, vector_size=dim)

    stats: dict[str, Any] = {
        "processed_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "chunks_created": 0,
        "points_upserted": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    manuals_dir = Path(settings.raw_manuals_dir)
    pdfs = sorted(manuals_dir.glob("**/*.pdf"))
    _log(f"Manuals dir: {manuals_dir} (exists={manuals_dir.exists()}); PDFs found={len(pdfs)}")

    for n, pdf in enumerate(pdfs, start=1):
        key = f"manual::{pdf.as_posix()}"
        sha = sha256_file(pdf)
        
        # PRODUCTION FIX: Check if force is True before skipping
        if not force and manifest.is_unchanged(key, sha):
            stats["skipped_files"] += 1
            continue

        _log(f"[manual {n}/{len(pdfs)}] Processing: {pdf}")
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

            eq_id = _extract_equipment_id(pdf.name)
            extra_meta = {
                "source_file": pdf.name,
                "path": pdf.as_posix(),
                "equipment_id": eq_id if eq_id else "all"
            }

            chunks = chunk_text(
                full_text,
                source_id=source_id,
                doc_type="manual",
                extra_meta=extra_meta,
            )

            stats["chunks_created"] += len(chunks)
            _log(f"[manual {n}/{len(pdfs)}] chunks={len(chunks)} embedding/upserting...")

            for i in range(0, len(chunks), settings.embed_batch_size):
                batch_chunks = chunks[i : i + settings.embed_batch_size]
                vecs = embedder.embed_texts([c.text for c in batch_chunks])

                points: list[VectorPoint] = []
                for c, v in zip(batch_chunks, vecs):
                    payload = dict(c.metadata)
                    payload["text"] = c.text
                    points.append(VectorPoint(id=c.chunk_id, vector=v, payload=payload))

                for j in range(0, len(points), settings.upsert_batch_size):
                    batch = points[j : j + settings.upsert_batch_size]
                    store.upsert(settings.qdrant_collection_docs, batch)
                    stats["points_upserted"] += len(batch)

            stats["processed_files"] += 1
            manifest.mark(key, sha, status="ok")
            _log(f"[manual {n}/{len(pdfs)}] done")
        except Exception as e:
            stats["failed_files"] += 1
            manifest.mark(key, sha, status="failed", detail=str(e))
            _log(f"[manual {n}/{len(pdfs)}] FAILED: {e!r}")

    proc_dir = Path(settings.raw_procedures_dir)
    mds = sorted([*proc_dir.glob("**/*.md"), *proc_dir.glob("**/*.markdown")])
    _log(f"Procedures dir: {proc_dir} (exists={proc_dir.exists()}); MD files found={len(mds)}")

    for n, md in enumerate(mds, start=1):
        key = f"procedure::{md.as_posix()}"
        sha = sha256_file(md)
        
        # PRODUCTION FIX: Check if force is True before skipping
        if not force and manifest.is_unchanged(key, sha):
            stats["skipped_files"] += 1
            continue

        _log(f"[procedure {n}/{len(mds)}] Processing: {md}")
        try:
            text = clean_text(load_markdown(md))
            if not text.strip():
                stats["failed_files"] += 1
                manifest.mark(
                    key, sha, status="failed", detail="Empty procedure text after cleaning"
                )
                _log(f"[procedure {n}/{len(mds)}] FAILED: empty text after cleaning")
                continue

            source_id = f"procedure__{md.stem}"
            _write_processed_text(
                source_id,
                {
                    "source_file": md.as_posix(),
                    "doc_type": "procedure",
                    "text": text,
                },
            )

            eq_id = _extract_equipment_id(md.name)
            extra_meta = {
                "source_file": md.name,
                "path": md.as_posix(),
                "equipment_id": eq_id if eq_id else "all"
            }

            chunks = chunk_text(
                text,
                source_id=source_id,
                doc_type="procedure",
                extra_meta=extra_meta,
            )

            stats["chunks_created"] += len(chunks)
            _log(f"[procedure {n}/{len(mds)}] chunks={len(chunks)} embedding/upserting...")

            for i in range(0, len(chunks), settings.embed_batch_size):
                batch_chunks = chunks[i : i + settings.embed_batch_size]
                vecs = embedder.embed_texts([c.text for c in batch_chunks])

                points: list[VectorPoint] = []
                for c, v in zip(batch_chunks, vecs):
                    payload = dict(c.metadata)
                    payload["text"] = c.text
                    points.append(VectorPoint(id=c.chunk_id, vector=v, payload=payload))

                for j in range(0, len(points), settings.upsert_batch_size):
                    batch = points[j : j + settings.upsert_batch_size]
                    store.upsert(settings.qdrant_collection_procedures, batch)
                    stats["points_upserted"] += len(batch)

            stats["processed_files"] += 1
            manifest.mark(key, sha, status="ok")
            _log(f"[procedure {n}/{len(mds)}] done")
        except Exception as e:
            stats["failed_files"] += 1
            manifest.mark(key, sha, status="failed", detail=str(e))
            _log(f"[procedure {n}/{len(mds)}] FAILED: {e!r}")

    manifest.save()

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["qdrant_counts"] = {
        settings.qdrant_collection_docs: store.count(settings.qdrant_collection_docs),
        settings.qdrant_collection_procedures: store.count(settings.qdrant_collection_procedures),
    }
    _log(f"Finished. Qdrant counts: {stats['qdrant_counts']}")
    return stats

if __name__ == "__main__":
    out = ingest_all(force=True)
    print(json.dumps(out, indent=2), flush=True)