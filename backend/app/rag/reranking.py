"""Cross-encoder relevance gate for vector candidates.

The bi-encoder retrieves broadly and cheaply. A cross-encoder then reads the
query and each candidate passage together and scores actual relevance. The
gate rejects weak candidates before they reach the model, which is what keeps
off-topic questions from producing confident, unrelated clinic answers.

The cross-encoder does not replace vector ranking; it filters candidates while
the vector score remains the reported similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from app.rag.embeddings import configure_model_cache


class Reranker(Protocol):
    """Minimal cross-encoder surface used by the retriever."""

    model_name: str

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]: ...


class FastEmbedReranker:
    """Local ONNX cross-encoder used as a relevance gate."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        configure_model_cache(cache_dir)
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        cleaned = [str(document) for document in documents]
        if not cleaned:
            return []
        return [
            float(score)
            for score in self._model.rerank(query, cleaned)
        ]
