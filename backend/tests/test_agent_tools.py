"""REST-backed agent tools: reads, writes, errors, and LLM schemas."""

import httpx
import pytest
from sqlalchemy import func, select

from app.agent.tools import (
    TOOL_DEFINITIONS,
    AgentTools,
    BackendClient,
    ToolError,
)
from app.db.session import SessionLocal
from app.main import app
from app.models import PatientBooking
from app.rag.index import KnowledgeBase
from tests.rag_support import build_test_retriever

NEW_BOOKING_ARGS = {
    "doctor_id": "D103",
    "treatment_id": "T003",
    "booking_date": "2026-12-01",
    "booking_time": "10:00:00",
    "notes": "Skin rash on left arm",
}


@pytest.fixture()
async def http_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client


@pytest.fixture()
def backend(http_client) -> BackendClient:
    return BackendClient(http_client)


@pytest.fixture()
def tools(backend) -> AgentTools:
    knowledge = KnowledgeBase.load_default()
    return AgentTools(knowledge, build_test_retriever(knowledge), backend)


def _p001_booking_count() -> int:
    with SessionLocal() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(PatientBooking)
                .where(PatientBooking.patient_id == "P001")
            )
            or 0
        )


@pytest.mark.asyncio
async def test_get_patient_profile_reads_the_rest_endpoint(tools):
    result = await tools.execute("get_patient_profile", {}, patient_id="P001")

    assert result["patient"]["patient_id"] == "P001"
    assert result["patient"]["first_name"] == "John"
    assert result["allergies"] == ["Penicillin", "Peanuts"]
    assert result["patient"]["emergency_contact"].startswith("Jane Doe")


@pytest.mark.asyncio
async def test_get_patient_bookings_reads_the_rest_endpoint(tools):
    result = await tools.execute("get_patient_bookings", {}, patient_id="P001")

    assert result["patient_id"] == "P001"
    assert result["count"] == 2
    assert [booking["booking_id"] for booking in result["bookings"]] == [
        "B1001",
        "B1004",
    ]


@pytest.mark.asyncio
async def test_create_booking_writes_exactly_one_row_through_rest(tools):
    before = _p001_booking_count()

    created = await tools.execute(
        "create_booking",
        NEW_BOOKING_ARGS,
        patient_id="P001",
        client_request_id="tool-write-1",
    )

    assert created["patient_id"] == "P001"
    assert created["booking_id"] == "B1005"
    assert _p001_booking_count() == before + 1

    listing = await tools.execute("get_patient_bookings", {}, patient_id="P001")
    assert listing["count"] == 3
    assert created["booking_id"] in {
        booking["booking_id"] for booking in listing["bookings"]
    }


@pytest.mark.asyncio
async def test_create_booking_replay_does_not_write_a_second_row(tools):
    first = await tools.execute(
        "create_booking",
        NEW_BOOKING_ARGS,
        patient_id="P001",
        client_request_id="tool-idempotent-1",
    )
    before = _p001_booking_count()

    second = await tools.execute(
        "create_booking",
        NEW_BOOKING_ARGS,
        patient_id="P001",
        client_request_id="tool-idempotent-1",
    )

    assert second["booking_id"] == first["booking_id"]
    assert _p001_booking_count() == before


@pytest.mark.asyncio
async def test_unknown_patient_raises_a_structured_tool_error(tools):
    with pytest.raises(ToolError) as excinfo:
        await tools.execute(
            "create_booking",
            NEW_BOOKING_ARGS,
            patient_id="P999",
            client_request_id="tool-unknown-1",
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.code == "PATIENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected_before_the_http_call(tools):
    with pytest.raises(ToolError) as excinfo:
        await tools.execute(
            "create_booking",
            {**NEW_BOOKING_ARGS, "doctor_id": "not-a-doctor"},
            patient_id="P001",
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_unknown_tool_name_raises_a_tool_error(tools):
    with pytest.raises(ToolError) as excinfo:
        await tools.execute("drop_database", {}, patient_id="P001")

    assert excinfo.value.code == "UNKNOWN_TOOL"


@pytest.mark.asyncio
async def test_search_knowledge_returns_grounded_recommendation(tools):
    result = await tools.execute(
        "search_knowledge",
        {"query": "I have a skin rash on my arm"},
        patient_id="P001",
    )

    assert result["results"]
    assert result["recommendation"]["treatment"]["treatment_id"] == "T003"
    assert result["recommendation"]["doctor"]["doctor_id"] == "D103"
    assert result["recommendation"]["clinic"]["clinic_id"] == "C002"
    assert result["recommendation"]["treatment"]["price_usd"] == 200.0


@pytest.mark.asyncio
async def test_search_knowledge_enriches_doctor_results_with_clinic(tools):
    result = await tools.execute(
        "search_knowledge",
        {"query": "Dr. Elena Rostova dermatologist"},
        patient_id="P001",
    )

    doctor_result = next(
        item
        for item in result["results"]
        if item["kind"] == "doctor" and item["id"] == "D103"
    )
    assert doctor_result["clinic"]["clinic_id"] == "C002"
    assert doctor_result["clinic"]["clinic_name"] == "Sunset Wellness Center"
    assert doctor_result["clinic"]["address"]


@pytest.mark.asyncio
async def test_search_knowledge_returns_no_results_for_off_topic_queries(tools):
    result = await tools.execute(
        "search_knowledge",
        {"query": "What is the weather in Paris?"},
        patient_id="P001",
    )

    assert result["results"] == []
    assert result["recommendation"] is None


def test_tool_definitions_are_strict_and_complete():
    definitions = {definition["name"]: definition for definition in TOOL_DEFINITIONS}

    assert set(definitions) == {
        "search_knowledge",
        "get_patient_profile",
        "get_patient_bookings",
        "create_booking",
    }
    for definition in definitions.values():
        schema = definition["input_schema"]
        assert definition["description"]
        assert definition["strict"] is True
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

    booking_schema = definitions["create_booking"]["input_schema"]
    assert booking_schema["required"] == [
        "doctor_id",
        "treatment_id",
        "booking_date",
        "booking_time",
        "notes",
    ]
    assert "patient_id" not in booking_schema["properties"]
    assert "client_request_id" not in booking_schema["properties"]
    assert "clinic_id" not in booking_schema["properties"]
