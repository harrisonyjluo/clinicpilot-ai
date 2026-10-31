"""Dense embedding adapter.

Production uses ``fastembed`` with a local ONNX BGE model. Tests inject a
recording backend so the default suite stays offline and deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
E5_QUERY_PREFIX = "query: "


class EmbeddingBackend(Protocol):
    """Minimal fastembed-compatible surface used by the adapter."""

    def passage_embed(self, texts: Sequence[str]) -> Iterable[np.ndarray]: ...

    def query_embed(self, texts: Sequence[str]) -> Iterable[np.ndarray]: ...


class EmbeddingModel(Protocol):
    """Embedding model seam used by the indexer and retriever."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return unit-L2 float32 vectors; zero vectors stay zero."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (array / safe_norms).astype(np.float32)


def _default_query_prefix(model_name: str) -> str:
    lowered = model_name.casefold()
    if "bge" in lowered:
        return BGE_QUERY_PREFIX
    if "e5" in lowered:
        return E5_QUERY_PREFIX
    return ""


def configure_model_cache(cache_dir: str | Path | None) -> None:
    """Keep Hugging Face caches inside the project on restricted hosts."""

    if cache_dir is None:
        return
    base = Path(cache_dir)
    cache_root = base.parent / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class FastEmbedEmbeddingModel:
    """Local fastembed model with query/passage instruction handling."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | Path | None = None,
        *,
        dimension: int = 384,
        backend: EmbeddingBackend | None = None,
        query_prefix: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._query_prefix = (
            _default_query_prefix(model_name)
            if query_prefix is None
            else query_prefix
        )
        if backend is None:
            configure_model_cache(cache_dir)
            from fastembed import TextEmbedding

            backend = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        self._backend = backend

    def _validate(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.dimension}, got shape {vectors.shape}."
            )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [str(text) for text in texts]
        if not cleaned:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = np.asarray(
            list(self._backend.passage_embed(cleaned)),
            dtype=np.float32,
        )
        self._validate(vectors)
        return normalize_vectors(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        query = f"{self._query_prefix}{text}" if self._query_prefix else text
        vectors = np.asarray(
            list(self._backend.query_embed([query])),
            dtype=np.float32,
        )
        self._validate(vectors)
        return normalize_vectors(vectors)[0]
