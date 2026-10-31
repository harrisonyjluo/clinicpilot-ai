"""Embedding adapter: query/passage prefixes and normalization."""

import numpy as np
import pytest

from app.rag.embeddings import FastEmbedEmbeddingModel, normalize_vectors

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class RecordingBackend:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.queries: list[str] = []

    def passage_embed(self, texts):
        self.passages.extend(texts)
        return [np.asarray([1.0, 2.0, 2.0], dtype=np.float32) for _ in texts]

    def query_embed(self, texts):
        self.queries.extend(texts)
        return [np.asarray([0.0, 3.0, 4.0], dtype=np.float32) for _ in texts]


def test_bge_queries_get_the_instruction_prefix_and_passages_do_not():
    backend = RecordingBackend()
    model = FastEmbedEmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=3,
        backend=backend,
    )

    passages = model.embed_documents(["skin rash"])
    query = model.embed_query("my face is breaking out")

    assert backend.passages == ["skin rash"]
    assert backend.queries == [BGE_QUERY_PREFIX + "my face is breaking out"]
    assert passages.shape == (1, 3)
    assert np.allclose(passages[0], [1 / 3, 2 / 3, 2 / 3])
    assert np.allclose(query, [0.0, 0.6, 0.8])


def test_non_bge_models_do_not_get_the_bge_instruction():
    backend = RecordingBackend()
    model = FastEmbedEmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=3,
        backend=backend,
    )

    model.embed_query("skin rash")

    assert backend.queries == ["skin rash"]


def test_normalize_vectors_handles_zero_rows_and_is_idempotent():
    vectors = np.asarray(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    normalized = normalize_vectors(vectors)
    normalized_again = normalize_vectors(normalized)

    assert np.allclose(normalized[0], [0.6, 0.8, 0.0])
    assert np.allclose(normalized[1], [0.0, 0.0, 0.0])
    assert np.allclose(normalized, normalized_again)


def test_embedding_dimension_mismatch_is_reported():
    backend = RecordingBackend()
    model = FastEmbedEmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=5,
        backend=backend,
    )

    with pytest.raises(ValueError, match="dimension"):
        model.embed_documents(["skin rash"])
