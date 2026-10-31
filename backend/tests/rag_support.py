"""Deterministic test double for the dense-vector RAG pipeline.

The production retriever has no lexical scoring. This test-only embedder maps
ordinary patient phrases to topic dimensions so the default suite can exercise
the vector index, thresholds, and agent integration without downloading a model.
Real-model semantic quality is verified separately by ``test_model_rag.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.rag.chunking import build_chunks
from app.rag.embeddings import normalize_vectors
from app.rag.index import KnowledgeBase
from app.rag.reranking import Reranker
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import VectorStore

TOPICS = (
    "clinic_metro",
    "clinic_sunset",
    "dermatology",
    "orthopedics",
    "pediatrics",
    "immunization",
    "preventative",
    "hours",
    "saturday_open",
    "cancellation",
    "booking_generic",
    "allergy",
    "weather",
)
_TOPIC_INDEX = {topic: index for index, topic in enumerate(TOPICS)}

TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "clinic_metro": ("metro care", "metro"),
    "clinic_sunset": ("sunset wellness", "sunset"),
    "dermatology": (
        "dermatolog",
        "skin",
        "rash",
        "acne",
        "breakout",
        "breaking out",
        "pimple",
        "eczema",
        "mole",
        "cosmetic",
    ),
    "orthopedics": (
        "orthoped",
        "joint",
        "knee",
        "ankle",
        "sprain",
        "sports injur",
        "muscle",
        "bone",
    ),
    "pediatrics": (
        "pediatric",
        "child",
        "baby",
        "babies",
        "infant",
        "toddler",
        "kid",
        "little one",
        "adolescent",
    ),
    "immunization": ("immuniz", "vaccin", "flu", "jab", "shot", "injection"),
    "preventative": (
        "preventative",
        "preventive",
        "annual",
        "yearly",
        "annual physical",
        "yearly physical",
        "checkup",
        "check-up",
        "general health",
        "general practice",
    ),
    "hours": ("hour", "opening", "open", "closed"),
    "saturday_open": ("saturday hours are", "saturday clinic hours"),
    "cancellation": ("cancellation", "cancel", "no-show", "notice"),
    "booking_generic": (
        "book an appointment",
        "schedule an appointment",
        "make an appointment",
    ),
    "allergy": ("allerg", "penicillin", "peanut"),
    "weather": ("weather", "paris"),
}

QUERY_EXPANSIONS: dict[str, dict[str, float]] = {
    "saturday hours": {"saturday_open": 1.0, "hours": 0.5},
    "open on saturday": {"saturday_open": 1.0},
    "weekend hours": {"hours": 0.5, "saturday_open": 0.5},
}


class TopicEmbedder:
    """Test-only topic embedder; not used by production code."""

    model_name = "test-topic"
    dimension = len(TOPICS)

    def _vector(self, text: str, *, query: bool) -> np.ndarray:
        lowered = " ".join(text.casefold().split())
        weights = np.zeros(self.dimension, dtype=np.float32)
        for topic, terms in TOPIC_TERMS.items():
            index = _TOPIC_INDEX[topic]
            if any(term in lowered for term in terms):
                weights[index] = 1.0
        if query:
            for phrase, additions in QUERY_EXPANSIONS.items():
                if phrase in lowered:
                    for topic, weight in additions.items():
                        weights[_TOPIC_INDEX[topic]] += weight
        if not weights.any():
            return np.zeros(self.dimension, dtype=np.float32)
        return normalize_vectors(weights)[0]

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector(text, query=False) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text, query=True)


def build_test_retriever(
    knowledge: KnowledgeBase | None = None,
    *,
    min_score: float = 0.30,
    recommend_min_score: float = 0.50,
    recommend_margin: float = 0.05,
    mmr_lambda: float | None = 0.7,
    reranker: Reranker | None = None,
    rerank_min_score: float | None = None,
) -> KnowledgeRetriever:
    knowledge = knowledge or KnowledgeBase.load_default()
    embedder = TopicEmbedder()
    chunks = build_chunks(
        knowledge.documents,
        max_chars=800,
        overlap=120,
    )
    store = VectorStore.build(
        chunks=chunks,
        embeddings=embedder.embed_documents([chunk.text for chunk in chunks]),
        model_name=embedder.model_name,
        dimension=embedder.dimension,
        source_fingerprint=knowledge.source_fingerprint,
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
