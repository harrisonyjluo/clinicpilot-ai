"""FAISS-backed vector store: cosine ranking, filters, MMR, persistence."""

import json

import numpy as np
import pytest

from app.rag.chunking import Chunk
from app.rag.vector_store import VectorStore


def make_chunk(
    chunk_id: str,
    *,
    kind: str = "treatment",
    parent_id: str | None = None,
    category: str | None = None,
    specialty: str | None = None,
    payload: dict | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{chunk_id}:0",
        parent_id=parent_id or chunk_id,
        kind=kind,
        title=chunk_id,
        text=f"text for {chunk_id}",
        chunk_index=0,
        payload=payload or {"id": chunk_id},
        category=category,
        specialty=specialty,
    )


def build_store(
    chunks: list[Chunk],
    vectors: list[list[float]],
    *,
    dimension: int = 2,
) -> VectorStore:
    return VectorStore.build(
        chunks=chunks,
        embeddings=np.asarray(vectors, dtype=np.float32),
        model_name="test-model",
        dimension=dimension,
        source_fingerprint="fingerprint-1",
    )


def test_similarity_uses_normalized_cosine_not_raw_dot_product():
    chunks = [make_chunk("A"), make_chunk("B")]
    store = build_store(chunks, [[1.0, 0.0], [8.0, 2.0]])

    hits = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        limit=2,
        min_score=0.0,
        mmr_lambda=None,
    )

    assert [hit.chunk.parent_id for hit in hits] == ["A", "B"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.9701425, rel=1e-5)


def test_minimum_score_filters_weak_matches():
    store = build_store([make_chunk("A")], [[0.0, 1.0]])

    hits = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        min_score=0.5,
        mmr_lambda=None,
    )

    assert hits == []


def test_metadata_filters_restrict_kind_category_and_specialty():
    chunks = [
        make_chunk("T-DERM", category="Dermatology"),
        make_chunk("T-ORTHO", category="Orthopedics"),
        make_chunk("D-DERM", kind="doctor", specialty="Dermatology"),
    ]
    store = build_store(
        chunks,
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    by_category = store.search(
        query,
        kind="treatment",
        category="Dermatology",
        min_score=0.0,
        mmr_lambda=None,
    )
    by_specialty = store.search(
        query,
        kind="doctor",
        specialty="Dermatology",
        min_score=0.0,
        mmr_lambda=None,
    )
    impossible = store.search(
        query,
        category="Dermatology",
        specialty="Dermatology",
        min_score=0.0,
        mmr_lambda=None,
    )

    assert [hit.chunk.parent_id for hit in by_category] == ["T-DERM"]
    assert [hit.chunk.parent_id for hit in by_specialty] == ["D-DERM"]
    assert impossible == []


def test_mmr_prefers_a_diverse_second_result():
    chunks = [make_chunk("A"), make_chunk("B"), make_chunk("C")]
    store = build_store(
        chunks,
        [[1.0, 0.0], [0.99, 0.01], [0.8, 0.6]],
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    without_mmr = store.search(query, limit=2, min_score=0.0, mmr_lambda=None)
    with_mmr = store.search(query, limit=2, min_score=0.0, mmr_lambda=0.4)

    assert [hit.chunk.parent_id for hit in without_mmr] == ["A", "B"]
    assert [hit.chunk.parent_id for hit in with_mmr] == ["A", "C"]


def test_search_deduplicates_chunks_to_the_best_parent_chunk():
    first = Chunk(
        chunk_id="P:0",
        parent_id="P",
        kind="treatment",
        title="Parent",
        text="first chunk",
        chunk_index=0,
        payload={"id": "P"},
    )
    second = Chunk(
        chunk_id="P:1",
        parent_id="P",
        kind="treatment",
        title="Parent",
        text="second chunk",
        chunk_index=1,
        payload={"id": "P"},
    )
    store = build_store([first, second], [[1.0, 0.0], [0.9, 0.1]])

    hits = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        limit=3,
        min_score=0.0,
        mmr_lambda=None,
    )

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "P:0"


def test_save_and_load_round_trip(tmp_path):
    chunks = [make_chunk("A"), make_chunk("B", kind="clinic")]
    store = build_store(chunks, [[1.0, 0.0], [0.0, 1.0]])
    store.save(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    loaded = VectorStore.load(
        tmp_path,
        expected_model="test-model",
        expected_dimension=2,
    )
    hits = loaded.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        limit=2,
        min_score=0.5,
        mmr_lambda=None,
    )

    assert manifest["schema_version"] == 1
    assert manifest["model_name"] == "test-model"
    assert manifest["dimension"] == 2
    assert manifest["chunk_count"] == 2
    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "vectors.npz").exists()
    assert (tmp_path / "index_treatment.faiss").exists()
    assert (tmp_path / "index_clinic.faiss").exists()
    assert [hit.chunk.parent_id for hit in hits] == ["A"]


def test_load_rejects_a_different_model_or_dimension(tmp_path):
    store = build_store([make_chunk("A")], [[1.0, 0.0]])
    store.save(tmp_path)

    with pytest.raises(ValueError, match="model"):
        VectorStore.load(
            tmp_path,
            expected_model="another-model",
            expected_dimension=2,
        )
    with pytest.raises(ValueError, match="dimension"):
        VectorStore.load(
            tmp_path,
            expected_model="test-model",
            expected_dimension=3,
        )


def test_search_rejects_a_query_with_the_wrong_dimension():
    store = build_store([make_chunk("A")], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="dimension"):
        store.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_search_rejects_an_unknown_kind():
    store = build_store([make_chunk("A")], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="kind"):
        store.search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            kind="patient",
        )
