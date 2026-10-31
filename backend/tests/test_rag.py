"""Knowledge-base loading, retrieval, and recommendation tests."""

import pytest

from app.rag.index import KnowledgeBase
from app.rag.retriever import KnowledgeRetriever
from tests.rag_support import build_test_retriever


@pytest.fixture(scope="module")
def knowledge() -> KnowledgeBase:
    return KnowledgeBase.load_default()


@pytest.fixture(scope="module")
def retriever(knowledge: KnowledgeBase) -> KnowledgeRetriever:
    return build_test_retriever(knowledge)


def test_loads_every_clinic_doctor_and_treatment(knowledge):
    assert [clinic["clinic_id"] for clinic in knowledge.clinics] == ["C001", "C002"]
    assert [doctor["doctor_id"] for doctor in knowledge.doctors] == [
        "D101",
        "D102",
        "D103",
        "D104",
    ]
    assert [treatment["treatment_id"] for treatment in knowledge.treatments] == [
        "T001",
        "T002",
        "T003",
        "T004",
        "T005",
    ]


def test_preserves_quoted_csv_fields_with_commas(knowledge):
    clinic = knowledge.get_clinic("C001")

    assert clinic is not None
    assert clinic["address"] == "123 Medical Center Way, Suite 400, New York, NY 10001"
    assert "Sat: 9:00 AM - 1:00 PM" in clinic["operating_hours"]


def test_skin_rash_recommends_dermatology(knowledge, retriever):
    recommendation = retriever.recommend("I have a skin rash on my arm")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T003"
    assert recommendation.doctor["doctor_id"] == "D103"
    assert recommendation.clinic["clinic_id"] == "C002"


def test_knee_pain_recommends_orthopedics(knowledge, retriever):
    recommendation = retriever.recommend("My knee hurts when I walk")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T005"
    assert recommendation.doctor["doctor_id"] == "D104"
    assert recommendation.clinic["clinic_id"] == "C001"


def test_child_vaccination_recommends_pediatrics(knowledge, retriever):
    recommendation = retriever.recommend("Vaccination for my child")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T002"
    assert recommendation.doctor["doctor_id"] == "D102"


def test_flu_shot_recommends_immunization(knowledge, retriever):
    recommendation = retriever.recommend("I need a flu shot")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T004"
    assert recommendation.doctor["doctor_id"] == "D101"


def test_annual_checkup_recommends_preventative_care(knowledge, retriever):
    recommendation = retriever.recommend("I want an annual checkup")

    assert recommendation is not None
    assert recommendation.treatment["treatment_id"] == "T001"
    assert recommendation.doctor["doctor_id"] == "D101"


def test_saturday_hours_returns_grounded_clinic_record(retriever):
    results = retriever.search("What are your Saturday hours?", limit=10)

    clinics = [result for result in results if result.kind == "clinic"]
    assert clinics
    assert clinics[0].id == "C001"
    assert "Sat: 9:00 AM - 1:00 PM" in clinics[0].payload["operating_hours"]


def test_clinic_name_query_ranks_that_clinic_first(retriever):
    results = retriever.search("What is Sunset Wellness cancellation policy?", limit=10)

    clinics = [result for result in results if result.kind == "clinic"]
    assert clinics
    assert clinics[0].id == "C002"
    assert "12 hours notice" in clinics[0].payload["cancellation_policy"]


def test_nonsense_query_returns_no_fabricated_results(retriever):
    results = retriever.search("What is the weather in Paris?", limit=10)

    assert results == []
    assert retriever.recommend("What is the weather in Paris?") is None


def test_every_retrieved_identifier_exists_in_the_seed_data(knowledge, retriever):
    results = retriever.search("skin rash child knee flu checkup", limit=20)

    for result in results:
        if result.kind == "clinic":
            assert knowledge.get_clinic(result.id) is not None
        elif result.kind == "doctor":
            assert knowledge.get_doctor(result.id) is not None
        else:
            assert knowledge.get_treatment(result.id) is not None


def test_recommendation_joins_doctor_and_clinic_to_the_treatment(knowledge, retriever):
    recommendation = retriever.recommend("skin rash")

    assert recommendation is not None
    assert recommendation.treatment["duration_minutes"] == 30
    assert recommendation.doctor["clinic_id"] == recommendation.clinic["clinic_id"]
    assert (
        recommendation.doctor["specialty"]
        == recommendation.treatment["category"]
        == "Dermatology"
    )
