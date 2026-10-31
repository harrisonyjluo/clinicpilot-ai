"""Build, persist, and load the dense-vector index used by the chat runtime.

The index is rebuilt when it is missing, corrupt, built by another model, or
older than the source CSVs. Reusing a valid index keeps process startup fast
and makes retrieval deterministic across restarts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.rag.chunking import build_chunks
from app.rag.embeddings import EmbeddingModel, FastEmbedEmbeddingModel
from app.rag.index import KnowledgeBase
from app.rag.reranking import FastEmbedReranker, Reranker
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_store(
    knowledge: KnowledgeBase,
    *,
    embedder: EmbeddingModel,
    index_dir: str | Path,
    chunk_max_chars: int = 800,
    chunk_overlap: int = 120,
) -> VectorStore:
    """Chunk, embed, and persist every knowledge document."""

    chunks = build_chunks(
        knowledge.documents,
        max_chars=chunk_max_chars,
        overlap=chunk_overlap,
    )
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store = VectorStore.build(
        chunks=chunks,
        embeddings=vectors,
        model_name=embedder.model_name,
        dimension=embedder.dimension,
        source_fingerprint=knowledge.source_fingerprint,
    )
    store.save(index_dir)
    return store


def load_or_build_store(
    knowledge: KnowledgeBase,
    *,
    embedder: EmbeddingModel,
    index_dir: str | Path,
    chunk_max_chars: int = 800,
    chunk_overlap: int = 120,
    rebuild: bool = False,
) -> VectorStore:
    """Return a compatible persisted index, rebuilding it when necessary."""

    directory = Path(index_dir)
    if not rebuild:
        try:
            store = VectorStore.load(
                directory,
                expected_model=embedder.model_name,
                expected_dimension=embedder.dimension,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            logger.info("Building the RAG index at %s: %s", directory, exc)
        else:
            if store.source_fingerprint == knowledge.source_fingerprint:
                return store
            logger.info(
                "RAG source CSVs changed; rebuilding the index at %s.",
                directory,
            )

    return build_store(
        knowledge,
        embedder=embedder,
        index_dir=directory,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap=chunk_overlap,
    )


def create_retriever(
    knowledge: KnowledgeBase,
    *,
    embedder: EmbeddingModel,
    index_dir: str | Path,
    min_score: float = 0.48,
    recommend_min_score: float = 0.55,
    recommend_margin: float = 0.03,
    mmr_lambda: float | None = 0.7,
    chunk_max_chars: int = 800,
    chunk_overlap: int = 120,
    rebuild: bool = False,
    reranker: Reranker | None = None,
    rerank_min_score: float | None = None,
) -> KnowledgeRetriever:
    """Wire the persisted vector store into a pure-vector retriever."""

    store = load_or_build_store(
        knowledge,
        embedder=embedder,
        index_dir=index_dir,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap=chunk_overlap,
        rebuild=rebuild,
    )
    return KnowledgeRetriever(
        knowledge,
        embedder,
        store,
        min_score=min_score,
        recommend_min_score=recommend_min_score,
        recommend_margin=recommend_margin,
        mmr_lambda=mmr_lambda,
        reranker=reranker,
        rerank_min_score=rerank_min_score,
    )


def create_configured_retriever(
    settings: Settings,
    *,
    knowledge: KnowledgeBase | None = None,
    embedder: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    enable_reranker: bool | None = None,
) -> KnowledgeRetriever:
    """Build the production retriever from environment-backed settings."""

    knowledge = knowledge or KnowledgeBase.load_default()
    embedder = embedder or FastEmbedEmbeddingModel(
        model_name=settings.rag_embedding_model,
        cache_dir=settings.rag_embedding_cache_dir,
        dimension=settings.rag_embedding_dimension,
    )
    use_reranker = (
        settings.rag_rerank_enabled
        if enable_reranker is None
        else enable_reranker
    )
    if reranker is None and use_reranker:
        reranker = FastEmbedReranker(
            model_name=settings.rag_rerank_model,
            cache_dir=settings.rag_embedding_cache_dir,
        )
    return create_retriever(
        knowledge,
        embedder=embedder,
        index_dir=settings.rag_index_dir,
        min_score=settings.rag_min_score,
        recommend_min_score=settings.rag_recommend_min_score,
        recommend_margin=settings.rag_recommend_margin,
        mmr_lambda=settings.rag_mmr_lambda,
        chunk_max_chars=settings.rag_chunk_max_chars,
        chunk_overlap=settings.rag_chunk_overlap,
        rebuild=settings.rag_rebuild_index,
        reranker=reranker,
        rerank_min_score=(
            settings.rag_rerank_min_score
            if reranker is not None
            else None
        ),
    )
