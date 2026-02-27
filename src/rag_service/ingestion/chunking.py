from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_service.core.config import settings


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _token_len(text: str, enc) -> int:
    return len(enc.encode(text))


def chunk_text(
    text: str,
    *,
    source_id: str,
    doc_type: str,
    extra_meta: dict[str, Any],
) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=lambda s: _token_len(s, enc),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pieces = splitter.split_text(text)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > settings.max_context_chars_per_chunk:
            piece = piece[: settings.max_context_chars_per_chunk]

        chunk_id = f"{source_id}::c{idx:05d}"
        meta = dict(extra_meta)
        meta.update(
            {
                "source_id": source_id,
                "doc_type": doc_type,
                "chunk_index": idx,
            }
        )
        chunks.append(Chunk(chunk_id=chunk_id, text=piece, metadata=meta))
    return chunks
