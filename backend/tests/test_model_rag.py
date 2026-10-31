"""Opt-in semantic quality checks with the real local embedding model.

Run with:
    RUN_MODEL_TESTS=1 pytest -m model
"""

import os

import pytest

from app.core.config import Settings
from app.rag.evaluate import evaluate_retriever
from app.rag.index import KnowledgeBase
from app.rag.runtime import create_configured_retriever
from app.rag.vector_store import VectorStore

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 to load the real embedding model",
    ),
]


@pytest.fixture(scope="module")
def model_retriever(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("model-rag")
    settings = Settings(
        rag_index_dir=str(index_dir),
        rag_rerank_enabled=True,
    )
    retriever = create_configured_retriever(
        settings,
        knowledge=KnowledgeBase.load_default(),
    )
    return retriever, index_dir


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("My face is breaking out", "T003"),
        ("I twisted my ankle playing soccer", "T005"),
        ("My little one needs their shots", "T002"),
        ("I need a flu jab", "T004"),
        ("I want my yearly physical", "T001"),
    ],
)
def test_paraphrases_rank_the_expected_treatment_first(
    model_retriever,
    query,
    expected_id,
):
    retriever, _ = model_retriever

    results = retriever.search(query, kind="treatment", limit=3)

    assert results
    assert results[0].id == expected_id


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("What are your Saturday hours?", "C001"),
        ("What is the Sunset cancellation policy?", "C002"),
    ],
)
def test_clinic_questions_rank_the_expected_clinic_first(
    model_retriever,
    query,
    expected_id,
):
    retriever, _ = model_retriever

    results = retriever.search(query, kind="clinic", limit=3)

    assert results
    assert results[0].id == expected_id


def test_generic_booking_and_off_topic_queries_do_not_recommend(model_retriever):
    retriever, _ = model_retriever

    assert retriever.recommend("I want to book an appointment") is None
    assert retriever.recommend("What is the weather in Paris?") is None
    assert retriever.search("What is the weather in Paris?") == []


def test_golden_set_meets_the_release_threshold(model_retriever):
    retriever, _ = model_retriever

    report = evaluate_retriever(retriever, k=3)

    assert report["recall_at_k"] == 1.0
    assert report["no_match_accuracy"] == 1.0


def test_model_index_survives_a_save_and_load_round_trip(model_retriever):
    retriever, index_dir = model_retriever

    loaded = VectorStore.load(
        index_dir,
        expected_model=retriever.embedder.model_name,
        expected_dimension=retriever.embedder.dimension,
    )
    query_vector = retriever.embedder.embed_query("skin rash")
    hits = loaded.search(
        query_vector,
        kind="treatment",
        limit=1,
        min_score=retriever.min_score,
    )

    assert loaded.ntotal == retriever.store.ntotal
    assert hits and hits[0].chunk.parent_id == "T003"
