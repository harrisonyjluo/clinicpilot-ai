"""Runtime wiring for the persisted dense-vector index."""

import json

from app.core.config import Settings
from app.rag.index import KnowledgeBase
from app.rag.runtime import create_configured_retriever, create_retriever
from tests.rag_support import TopicEmbedder


def test_create_retriever_builds_persists_and_reloads_the_index(tmp_path):
    knowledge = KnowledgeBase.load_default()

    first = create_retriever(
        knowledge,
        embedder=TopicEmbedder(),
        index_dir=tmp_path,
    )

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "index_treatment.faiss").exists()
    assert first.search("skin rash", kind="treatment")[0].id == "T003"

    second = create_retriever(
        knowledge,
        embedder=TopicEmbedder(),
        index_dir=tmp_path,
    )

    assert second.store.ntotal == first.store.ntotal
    assert second.store.source_fingerprint == knowledge.source_fingerprint


def test_stale_source_fingerprint_rebuilds_the_index(tmp_path):
    knowledge = KnowledgeBase.load_default()
    create_retriever(knowledge, embedder=TopicEmbedder(), index_dir=tmp_path)

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_fingerprint"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    create_retriever(knowledge, embedder=TopicEmbedder(), index_dir=tmp_path)

    rebuilt = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rebuilt["source_fingerprint"] == knowledge.source_fingerprint


def test_configured_retriever_receives_settings_thresholds(tmp_path):
    settings = Settings(
        rag_index_dir=str(tmp_path),
        rag_min_score=0.31,
        rag_recommend_min_score=1.01,
        rag_recommend_margin=0.22,
        rag_mmr_lambda=0.4,
        rag_chunk_max_chars=500,
        rag_chunk_overlap=50,
    )

    retriever = create_configured_retriever(
        settings,
        knowledge=KnowledgeBase.load_default(),
        embedder=TopicEmbedder(),
        enable_reranker=False,
    )

    assert retriever.min_score == 0.31
    assert retriever.recommend_min_score == 1.01
    assert retriever.recommend_margin == 0.22
    assert retriever.mmr_lambda == 0.4
    assert retriever.reranker is None
    assert retriever.recommend("skin rash") is None


class FixedReranker:
    model_name = "fixed"

    def score(self, query, documents):
        return [1.0 for _ in documents]


def test_configured_retriever_uses_an_injected_reranker(tmp_path):
    settings = Settings(rag_index_dir=str(tmp_path))
    reranker = FixedReranker()

    retriever = create_configured_retriever(
        settings,
        knowledge=KnowledgeBase.load_default(),
        embedder=TopicEmbedder(),
        reranker=reranker,
    )

    assert retriever.reranker is reranker
    assert retriever.rerank_min_score == settings.rag_rerank_min_score
    assert retriever.search("skin rash", kind="treatment")[0].id == "T003"
