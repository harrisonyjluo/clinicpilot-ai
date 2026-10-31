"""Golden-set evaluation of pure-vector retrieval."""

from app.rag.evaluate import evaluate_retriever
from tests.rag_support import build_test_retriever


def test_golden_set_is_retrieved_by_the_deterministic_embedder():
    report = evaluate_retriever(build_test_retriever(), k=3)

    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["ndcg_at_k"] == 1.0
    assert report["no_match_accuracy"] == 1.0
    assert report["passed"] is True
    assert len(report["cases"]) >= 8
