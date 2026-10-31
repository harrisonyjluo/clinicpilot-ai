"""Sentence-aware chunking with overlap for dense-vector indexing.

Each chunk keeps a pointer to its parent record so retrieval can return the
whole clinic/doctor/treatment row even when it was indexed in several pieces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.rag.index import KnowledgeDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One embedded slice of a parent knowledge record."""

    chunk_id: str
    parent_id: str
    kind: str
    title: str
    text: str
    chunk_index: int
    payload: dict[str, Any]
    category: str | None = None
    specialty: str | None = None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _split_long_sentence(
    sentence: str,
    *,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Split one over-long sentence into bounded overlapping windows."""

    if len(sentence) <= max_chars:
        return [sentence]

    step = max(1, max_chars - overlap)
    windows: list[str] = []
    start = 0
    while start < len(sentence):
        end = min(start + max_chars, len(sentence))
        window = sentence[start:end].strip()
        if window:
            windows.append(window)
        if end == len(sentence):
            break
        start += step
    return windows


def chunk_text(
    text: str,
    *,
    max_chars: int = 800,
    overlap: int = 120,
) -> list[str]:
    """Split text into sentence-aware chunks bounded by ``max_chars``.

    Short records stay intact. Long records are split at sentence boundaries;
    consecutive chunks share up to ``overlap`` characters of context. A single
    sentence longer than ``max_chars`` is split into fixed overlapping windows.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between 0 and max_chars")

    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(cleaned)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                _split_long_sentence(
                    sentence,
                    max_chars=max_chars,
                    overlap=overlap,
                )
            )
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        tail = current[-overlap:] if overlap and current else ""
        if tail:
            allowed_tail = max_chars - len(sentence) - 1
            if allowed_tail > 0:
                tail = tail[-allowed_tail:]
                current = f"{tail} {sentence}".strip()
            else:
                current = sentence
        else:
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def build_chunks(
    documents: Sequence[KnowledgeDocument],
    *,
    max_chars: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Build parent-linked chunks for every knowledge document."""

    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(
            chunk_text(document.text, max_chars=max_chars, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.id}:{index}",
                    parent_id=document.id,
                    kind=document.kind,
                    title=document.title,
                    text=text,
                    chunk_index=index,
                    payload=document.payload,
                    category=document.category,
                    specialty=document.specialty,
                )
            )
    return chunks
