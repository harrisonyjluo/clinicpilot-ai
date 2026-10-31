"""Pure dense-vector retrieval over the knowledge base.

There is deliberately no lexical scoring here: no tokenization, no BM25/TF-IDF,
no token overlap, no keyword weights, and no synonym matching. A query is
embedded once and ranked by cosine similarity against the persisted vectors.

Exact IDs, dates, and times are parsed by the intent router as booking slots;
they are not retrieval signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rag.embeddings import EmbeddingModel
from app.rag.index import KnowledgeBase
from app.rag.reranking import Reranker
from app.rag.vector_store import SearchHit, VectorStore

KINDS = ("clinic", "doctor", "treatment")


@dataclass(frozen=True)
class SearchResult:
    kind: str
    id: str
    title: str
    score: float
    matched_on: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class Recommendation:
    treatment: dict[str, Any]
    doctor: dict[str, Any]
    clinic: dict[str, Any]
    reason: str


class KnowledgeRetriever:
    """Embedding-only retrieval and recommendation."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        embedder: EmbeddingModel,
        store: VectorStore,
        *,
        min_score: float = 0.48,
        recommend_min_score: float = 0.55,
        recommend_margin: float = 0.03,
        mmr_lambda: float | None = 0.7,
        reranker: Reranker | None = None,
        rerank_min_score: float | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.embedder = embedder
        self.store = store
        self.min_score = min_score
        self.recommend_min_score = recommend_min_score
        self.recommend_margin = recommend_margin
        self.mmr_lambda = mmr_lambda
        self.reranker = reranker
        self.rerank_min_score = rerank_min_score

    def _gate(
        self,
        query: str,
        hits: list[SearchHit],
    ) -> list[SearchHit]:
        """Drop vector candidates that the cross-encoder judges irrelevant."""

        if (
            self.reranker is None
            or self.rerank_min_score is None
            or not hits
        ):
            return hits
        scores = self.reranker.score(
            query,
            [hit.chunk.text for hit in hits],
        )
        if len(scores) != len(hits):
            raise ValueError(
                "Reranker returned a different number of scores than documents."
            )
        return [
            hit
            for hit, score in zip(hits, scores)
            if score >= self.rerank_min_score
        ]

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        specialty: str | None = None,
        kind: str | None = None,
        limit: int = 3,
    ) -> list[SearchResult]:
        cleaned = query.strip()
        if not cleaned or limit <= 0:
            return []

        query_vector = self.embedder.embed_query(cleaned)
        hits = self.store.search(
            query_vector,
            limit=limit,
            kind=kind,
            category=category,
            specialty=specialty,
            min_score=self.min_score,
            mmr_lambda=self.mmr_lambda,
        )
        hits = self._gate(cleaned, hits)
        return [
            SearchResult(
                kind=hit.chunk.kind,
                id=hit.chunk.parent_id,
                title=hit.chunk.title,
                score=round(hit.score, 6),
                matched_on=("vector",),
                payload=hit.chunk.payload,
            )
            for hit in hits
        ]

    def search_all(
        self,
        query: str,
        *,
        category: str | None = None,
        specialty: str | None = None,
        limit_per_kind: int = 3,
    ) -> dict[str, list[SearchResult]]:
        """Search every kind, or only the kind selected by a metadata filter."""

        if category is not None:
            kinds = ("treatment",)
        elif specialty is not None:
            kinds = ("doctor",)
        else:
            kinds = KINDS

        return {
            kind: self.search(
                query,
                kind=kind,
                category=category,
                specialty=specialty,
                limit=limit_per_kind,
            )
            for kind in kinds
        }

    def recommend(self, query: str) -> Recommendation | None:
        """Return a treatment/doctor/clinic only when vector evidence is strong."""

        cleaned = query.strip()
        if not cleaned:
            return None

        query_vector = self.embedder.embed_query(cleaned)
        hits = self.store.search(
            query_vector,
            kind="treatment",
            limit=2,
            min_score=self.recommend_min_score,
            mmr_lambda=None,
        )
        hits = self._gate(cleaned, hits)
        if not hits:
            return None

        best = hits[0]
        if (
            len(hits) > 1
            and best.score - hits[1].score < self.recommend_margin
        ):
            return None

        treatment = best.chunk.payload
        doctor = self.knowledge.doctor_for_treatment(
            treatment["treatment_id"]
        )
        if doctor is None:
            return None
        clinic = self.knowledge.get_clinic(doctor["clinic_id"])
        if clinic is None:
            return None

        return Recommendation(
            treatment=treatment,
            doctor=doctor,
            clinic=clinic,
            reason=(
                f"{treatment['treatment_name']} ({treatment['treatment_id']}) "
                f"is the closest semantic match for the described need "
                f"(cosine similarity {best.score:.3f}); "
                f"{doctor['full_name']} ({doctor['doctor_id']}) specializes in "
                f"{doctor['specialty']}."
            ),
        )
