"""CLI contract for building the persisted dense-vector index."""

from app.rag import build_index as build_index_module
from app.rag.vector_store import VectorStore
from tests.rag_support import TopicEmbedder


def test_build_index_cli_persists_a_loadable_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        build_index_module,
        "FastEmbedEmbeddingModel",
        lambda **_kwargs: TopicEmbedder(),
    )

    summary = build_index_module.main(["--index-dir", str(tmp_path)])

    assert summary["documents"] == 11
    assert summary["chunk_count"] == 11
    assert summary["kinds"] == {"clinic": 2, "doctor": 4, "treatment": 5}
    assert (tmp_path / "manifest.json").exists()

    loaded = VectorStore.load(
        tmp_path,
        expected_model=TopicEmbedder.model_name,
        expected_dimension=TopicEmbedder.dimension,
    )
    assert loaded.ntotal == summary["chunk_count"]
