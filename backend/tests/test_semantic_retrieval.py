"""Behavior tests for pure dense-vector retrieval.

These tests initially run against the legacy lexical retriever: the semantic
paraphrase cases fail, which is the RED evidence for the vector rewrite. They
also pin the safety behavior that generic booking phrases and off-topic queries
must never produce a treatment recommendation.
"""

from app.rag.index import KnowledgeBase
from app.rag.retriever import KnowledgeRetriever
from tests.rag_support import build_test_retriever


def _retriever() -> KnowledgeRetriever:
    return build_test_retriever()


def test_face_breakout_paraphrase_recommends_dermatology():
    recommendation = _retriever().recommend("my face is breaking out")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T003"
    assert recommendation.doctor["doctor_id"] == "D103"
    assert recommendation.clinic["clinic_id"] == "C002"


def test_twisted_ankle_paraphrase_recommends_orthopedics():
    recommendation = _retriever().recommend(
        "I twisted my ankle playing soccer"
    )

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T005"
    assert recommendation.doctor["doctor_id"] == "D104"
    assert recommendation.clinic["clinic_id"] == "C001"


def test_child_shot_paraphrase_recommends_pediatrics_not_flu_shot():
    recommendation = _retriever().recommend("my little one needs her shots")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T002"
    assert recommendation.doctor["doctor_id"] == "D102"


def test_results_report_vector_evidence_only():
    results = _retriever().search("skin rash", limit=5)

    assert results
    for result in results:
        assert result.matched_on == ("vector",)
        assert 0.0 <= result.score <= 1.0


def test_index_text_is_natural_language_not_keyword_stuffing():
    knowledge = KnowledgeBase.load_default()

    for document in knowledge.documents:
        text = document.text.casefold()
        assert "hours opening open closed schedule address phone location" not in text
        assert "treatment service appointment" not in text
        assert "doctor physician" not in text


def test_generic_booking_phrase_never_infers_a_treatment():
    assert _retriever().recommend("I want to book an appointment.") is None
    assert _retriever().recommend("next Tuesday at 10am") is None


def test_off_topic_query_returns_no_recommendation_or_results():
    retriever = _retriever()

    assert retriever.recommend("What is the weather in Paris?") is None
    assert retriever.search("What is the weather in Paris?", limit=10) == []
