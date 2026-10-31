"""Persisted FAISS vector store with cosine search and MMR.

Vectors are normalized at build time, so inner product is cosine similarity.
The store keeps one exact flat index per document kind. That makes per-kind
retrieval exact instead of relying on an over-fetch heuristic, and it leaves a
clean seam for HNSW/IVF or a vector database at larger scale.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from app.rag.chunking import Chunk
from app.rag.embeddings import normalize_vectors

SCHEMA_VERSION = 1
KINDS = ("clinic", "doctor", "treatment")


@dataclass(frozen=True)
class SearchHit:
    """A parent-deduplicated retrieval hit with its cosine similarity."""

    chunk: Chunk
    score: float


class VectorStore:
    """Exact cosine index over chunk embeddings, persisted as FAISS + JSON."""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        metric: str = "cosine",
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.metric = metric
        self.source_fingerprint = ""
        self._chunks_by_kind: dict[str, list[Chunk]] = {}
        self._vectors_by_kind: dict[str, np.ndarray] = {}
        self._indexes: dict[str, Any] = {}
        self._chunk_vectors: dict[str, np.ndarray] = {}

    @classmethod
    def build(
        cls,
        *,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        model_name: str,
        dimension: int,
        source_fingerprint: str = "",
    ) -> VectorStore:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != dimension:
            raise ValueError(
                "Embedding matrix does not match the vector dimension: "
                f"expected (*, {dimension}), got {array.shape}."
            )
        if len(chunks) != array.shape[0]:
            raise ValueError(
                "Every chunk must have exactly one embedding: "
                f"{len(chunks)} chunks, {array.shape[0]} vectors."
            )

        store = cls(model_name=model_name, dimension=dimension)
        store.source_fingerprint = source_fingerprint

        for kind in KINDS:
            indices = [
                index
                for index, chunk in enumerate(chunks)
                if chunk.kind == kind
            ]
            if not indices:
                continue
            kind_chunks = [chunks[index] for index in indices]
            kind_vectors = normalize_vectors(array[indices])
            index = faiss.IndexFlatIP(dimension)
            index.add(kind_vectors)
            store._chunks_by_kind[kind] = kind_chunks
            store._vectors_by_kind[kind] = kind_vectors
            store._indexes[kind] = index
            for chunk, vector in zip(kind_chunks, kind_vectors):
                store._chunk_vectors[chunk.chunk_id] = vector
        return store

    @property
    def ntotal(self) -> int:
        return sum(len(chunks) for chunks in self._chunks_by_kind.values())

    @property
    def chunk_counts(self) -> dict[str, int]:
        """Number of indexed chunks per document kind."""

        return {
            kind: len(self._chunks_by_kind.get(kind, []))
            for kind in KINDS
        }

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        limit: int = 3,
        kind: str | None = None,
        category: str | None = None,
        specialty: str | None = None,
        min_score: float = 0.0,
        mmr_lambda: float | None = 0.7,
        candidate_k: int | None = None,
    ) -> list[SearchHit]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise ValueError(
                "Query dimension mismatch: "
                f"expected {self.dimension}, got {query.shape[0]}."
            )
        if limit <= 0:
            return []
        if kind is not None and kind not in KINDS:
            raise ValueError(
                f"Unknown kind {kind!r}; expected one of {KINDS}."
            )
        if category is not None and specialty is not None:
            return []

        query = normalize_vectors(query)[0]
        kinds: list[str]
        if kind is not None:
            kinds = [kind]
        else:
            kinds = list(KINDS)
        if category is not None:
            kinds = [name for name in kinds if name == "treatment"]
        if specialty is not None:
            kinds = [name for name in kinds if name == "doctor"]

        candidates: list[SearchHit] = []
        for name in kinds:
            index = self._indexes.get(name)
            if index is None:
                continue
            chunks = self._chunks_by_kind[name]
            has_metadata_filter = category is not None or specialty is not None
            requested = len(chunks) if has_metadata_filter else (
                candidate_k or max(limit * 5, limit + 8)
            )
            requested = min(max(requested, 1), len(chunks))
            scores, ids = index.search(query.reshape(1, -1), requested)
            for score, raw_index in zip(scores[0], ids[0]):
                if raw_index < 0:
                    continue
                chunk = chunks[int(raw_index)]
                if not self._matches(chunk, category=category, specialty=specialty):
                    continue
                clipped = float(np.clip(score, -1.0, 1.0))
                if clipped < min_score:
                    continue
                candidates.append(SearchHit(chunk=chunk, score=clipped))

        best_by_parent: dict[tuple[str, str], SearchHit] = {}
        for hit in candidates:
            key = (hit.chunk.kind, hit.chunk.parent_id)
            existing = best_by_parent.get(key)
            if existing is None or hit.score > existing.score:
                best_by_parent[key] = hit

        ranked = sorted(
            best_by_parent.values(),
            key=lambda hit: (-hit.score, hit.chunk.kind, hit.chunk.parent_id),
        )
        if mmr_lambda is None:
            return ranked[:limit]
        return self._mmr(query, ranked, limit=limit, lambda_=mmr_lambda)

    @staticmethod
    def _matches(
        chunk: Chunk,
        *,
        category: str | None,
        specialty: str | None,
    ) -> bool:
        if category is not None:
            if (
                chunk.kind != "treatment"
                or chunk.category is None
                or chunk.category.casefold() != category.casefold()
            ):
                return False
        if specialty is not None:
            if (
                chunk.kind != "doctor"
                or chunk.specialty is None
                or chunk.specialty.casefold() != specialty.casefold()
            ):
                return False
        return True

    def _mmr(
        self,
        query: np.ndarray,
        candidates: Sequence[SearchHit],
        *,
        limit: int,
        lambda_: float,
    ) -> list[SearchHit]:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        remaining = list(candidates)
        selected: list[SearchHit] = []
        while remaining and len(selected) < limit:
            best: SearchHit | None = None
            best_value = float("-inf")
            for hit in remaining:
                if selected:
                    vector = self._chunk_vectors[hit.chunk.chunk_id]
                    max_similarity = max(
                        float(
                            np.dot(
                                vector,
                                self._chunk_vectors[chosen.chunk.chunk_id],
                            )
                        )
                        for chosen in selected
                    )
                else:
                    max_similarity = 0.0
                value = (
                    lambda_ * hit.score
                    - (1.0 - lambda_) * max_similarity
                )
                if value > best_value:
                    best_value = value
                    best = hit
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
        return selected

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        chunks_payload = [
            asdict(chunk)
            for kind in KINDS
            for chunk in self._chunks_by_kind.get(kind, [])
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "chunk_count": len(chunks_payload),
            "source_fingerprint": self.source_fingerprint,
            "kinds": {
                kind: len(self._chunks_by_kind.get(kind, []))
                for kind in KINDS
            },
        }
        self._atomic_write_json(target / "manifest.json", manifest)
        self._atomic_write_json(target / "chunks.json", chunks_payload)

        vectors_tmp = target / "vectors.tmp.npz"
        np.savez(
            vectors_tmp,
            **{
                f"vectors_{kind}": vectors
                for kind, vectors in self._vectors_by_kind.items()
            },
        )
        os.replace(vectors_tmp, target / "vectors.npz")

        for kind, index in self._indexes.items():
            index_tmp = target / f".index_{kind}.faiss.tmp"
            faiss.write_index(index, str(index_tmp))
            os.replace(index_tmp, target / f"index_{kind}.faiss")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_model: str | None = None,
        expected_dimension: int | None = None,
    ) -> VectorStore:
        source = Path(directory)
        manifest_path = source / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"RAG index manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported RAG index schema version: "
                f"{manifest.get('schema_version')!r}."
            )

        model_name = str(manifest["model_name"])
        dimension = int(manifest["dimension"])
        if expected_model is not None and expected_model != model_name:
            raise ValueError(
                "RAG index model mismatch: "
                f"index uses {model_name!r}, configured {expected_model!r}."
            )
        if expected_dimension is not None and expected_dimension != dimension:
            raise ValueError(
                "RAG index dimension mismatch: "
                f"index uses {dimension}, configured {expected_dimension}."
            )

        store = cls(
            model_name=model_name,
            dimension=dimension,
            metric=str(manifest.get("metric", "cosine")),
        )
        store.source_fingerprint = str(
            manifest.get("source_fingerprint", "")
        )

        chunks_path = source / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"RAG chunk metadata not found: {chunks_path}")
        chunks = [
            Chunk(**record)
            for record in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        vectors_path = source / "vectors.npz"
        if not vectors_path.exists():
            raise FileNotFoundError(f"RAG vectors not found: {vectors_path}")
        vector_archive = np.load(vectors_path)

        for kind in KINDS:
            kind_chunks = [chunk for chunk in chunks if chunk.kind == kind]
            if not kind_chunks:
                continue
            key = f"vectors_{kind}"
            if key not in vector_archive:
                raise ValueError(
                    f"RAG index is missing vectors for kind {kind!r}."
                )
            vectors = np.asarray(vector_archive[key], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != dimension:
                raise ValueError(
                    "RAG index vector shape mismatch: "
                    f"expected (*, {dimension}), got {vectors.shape}."
                )
            index_path = source / f"index_{kind}.faiss"
            index = faiss.read_index(str(index_path))
            if index.d != dimension:
                raise ValueError(
                    "RAG index dimension mismatch in FAISS index: "
                    f"{index.d} != {dimension}."
                )
            store._chunks_by_kind[kind] = kind_chunks
            store._vectors_by_kind[kind] = vectors
            store._indexes[kind] = index
            for chunk, vector in zip(kind_chunks, vectors):
                store._chunk_vectors[chunk.chunk_id] = vector
        return store

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
