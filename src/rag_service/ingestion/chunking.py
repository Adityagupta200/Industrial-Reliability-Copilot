from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

import tiktoken

from rag_service.core.config import settings


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _token_len(text: str, enc) -> int:
    return len(enc.encode(text))


def _stable_chunk_uuid(*, source_id: str, doc_type: str, chunk_index: int) -> str:
    """
    Deterministic UUID for a chunk. Ensures Qdrant-valid point IDs and stable upserts.
    """
    name = f"{doc_type}:{source_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _join(parts: list[str]) -> str:
    return "".join(parts).strip()


def _token_tail(text: str, overlap_tokens: int, enc) -> str:
    if overlap_tokens <= 0:
        return ""
    tokens = enc.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    return enc.decode(tokens[-overlap_tokens:])


def _hard_token_split(text: str, chunk_size: int, enc) -> list[str]:
    tokens = enc.encode(text)
    return [
        enc.decode(tokens[start : start + chunk_size])
        for start in range(0, len(tokens), chunk_size)
    ]


def _split_to_bounded_pieces(
    text: str,
    *,
    chunk_size: int,
    separators: list[str],
    enc,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _token_len(text, enc) <= chunk_size:
        return [text]
    if not separators:
        return _hard_token_split(text, chunk_size, enc)

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        return _hard_token_split(text, chunk_size, enc)

    raw_parts = text.split(separator)
    pieces: list[str] = []
    for idx, part in enumerate(raw_parts):
        if not part:
            continue
        unit = part if idx == len(raw_parts) - 1 else f"{part}{separator}"
        if _token_len(unit, enc) <= chunk_size:
            pieces.append(unit)
        else:
            pieces.extend(
                _split_to_bounded_pieces(
                    unit,
                    chunk_size=chunk_size,
                    separators=remaining,
                    enc=enc,
                )
            )

    return pieces


def _merge_with_overlap(
    pieces: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    enc,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for piece in pieces:
        candidate = _join([*current, piece])
        if current and _token_len(candidate, enc) > chunk_size:
            emitted = _join(current)
            if emitted:
                chunks.append(emitted)

            overlap = _token_tail(emitted, chunk_overlap, enc)
            candidate_with_overlap = _join([overlap, piece])
            current = (
                [overlap, piece]
                if _token_len(candidate_with_overlap, enc) <= chunk_size
                else [piece]
            )
        else:
            current.append(piece)

    emitted = _join(current)
    if emitted:
        chunks.append(emitted)

    return chunks


def _recursive_split_text(text: str, *, chunk_size: int, chunk_overlap: int, enc) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    bounded_pieces = _split_to_bounded_pieces(
        text,
        chunk_size=chunk_size,
        separators=separators,
        enc=enc,
    )
    return _merge_with_overlap(
        bounded_pieces,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enc=enc,
    )


def chunk_text(
    text: str,
    *,
    source_id: str,
    doc_type: str,
    extra_meta: dict[str, Any],
) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    pieces = _recursive_split_text(
        text,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        enc=enc,
    )
    chunks: list[Chunk] = []

    for idx, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > settings.max_context_chars_per_chunk:
            piece = piece[: settings.max_context_chars_per_chunk]

        chunk_id = _stable_chunk_uuid(source_id=source_id, doc_type=doc_type, chunk_index=idx)

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
