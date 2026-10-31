"""Cross-encoder relevance gate over vector candidates."""

from app.rag.retriever import KnowledgeRetriever
from tests.rag_support import build_test_retriever


class FixedReranker:
    model_name = "fixed-reranker"

    def __init__(self, score: float) -> None:
        self.score_value = score
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self.score_value for _ in documents]


def test_low_rerank_scores_remove_vector_hits():
    reranker = FixedReranker(-11.0)
    retriever = build_test_retriever(
        reranker=reranker,
        rerank_min_score=-8.0,
    )

    assert retriever.search("skin rash") == []
    assert retriever.recommend("skin rash") is None
    assert reranker.calls
    assert reranker.calls[0][0] == "skin rash"
    assert any("Dermatology" in text for text in reranker.calls[0][1])


def test_high_rerank_scores_preserve_vector_hits():
    reranker = FixedReranker(1.0)
    retriever = build_test_retriever(
        reranker=reranker,
        rerank_min_score=-8.0,
    )

    results = retriever.search("skin rash", kind="treatment")

    assert results
    assert results[0].id == "T003"
    assert retriever.recommend("skin rash") is not None
    assert reranker.calls


def test_without_a_reranker_the_vector_scores_remain_authoritative():
    retriever: KnowledgeRetriever = build_test_retriever()

    assert retriever.reranker is None
    assert retriever.search("skin rash", kind="treatment")[0].id == "T003"
