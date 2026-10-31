"""Live end-to-end chat tests against a running backend.

These tests exercise the real HTTP server, the real loopback REST tool path,
the real SQLite database, and the configured Anthropic provider. They are
opt-in: the default suite deselects the `e2e` marker, and each test skips
unless RUN_E2E_TESTS=1 and a healthy server answers at E2E_BASE_URL.

See the README and `TECH_DOC.md` for the full procedure.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from app.core.config import Settings
from app.rag.index import KnowledgeBase

pytestmark = pytest.mark.e2e

E2E_BASE_URL = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000")
PATIENT_ID = Settings().default_patient_id or "P001"
CLINIC_PROMPT_FRAGMENT = "which clinic"
DOCTOR_PROMPT_FRAGMENT = "which doctor or specialty"


@pytest.fixture(scope="module")
def client():
    if os.environ.get("RUN_E2E_TESTS") != "1":
        pytest.skip("Set RUN_E2E_TESTS=1 and select -m e2e to run live E2E tests.")
    try:
        health = httpx.get(f"{E2E_BASE_URL}/api/health", timeout=5)
    except httpx.HTTPError as exc:
        pytest.skip(f"No healthy backend at {E2E_BASE_URL}: {exc}")
    if health.status_code != 200:
        pytest.skip(f"Backend health returned {health.status_code}.")
    with httpx.Client(base_url=E2E_BASE_URL, timeout=120) as http_client:
        yield http_client


def chat(
    client: httpx.Client,
    message: str,
    *,
    conversation_id: str | None = None,
) -> dict:
    payload = {"message": message}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def bookings_for(client: httpx.Client, patient_id: str) -> list[dict]:
    response = client.get("/api/bookings", params={"patient_id": patient_id})
    assert response.status_code == 200, response.text
    return response.json()["bookings"]


def test_reported_multi_turn_transcript_does_not_repeat(client):
    conversation_id = str(uuid.uuid4())
    replies: list[str] = []

    first = chat(
        client,
        "我想要预约明天的医生",
        conversation_id=conversation_id,
    )
    replies.append(first["reply"])
    assert first["intent"] == "new_booking"
    assert first["booking"] is None
    assert "clinic" in first["reply"].casefold()

    second = chat(
        client,
        "我想预约Dr. NoSuch医生",
        conversation_id=conversation_id,
    )
    replies.append(second["reply"])
    assert second["booking"] is None
    assert "dr. nosuch" in second["reply"].casefold()
    assert "couldn't find" in second["reply"].casefold()

    known_names = [
        doctor["full_name"] for doctor in KnowledgeBase.load_default().doctors
    ]
    assert any(name in second["reply"] for name in known_names), second["reply"]

    third = chat(
        client,
        "你可以看到我发的文字吗",
        conversation_id=conversation_id,
    )
    replies.append(third["reply"])
    assert third["booking"] is None
    assert third["reply"].strip().casefold() != replies[0].strip().casefold()
    reminder = "to continue booking"
    assert reminder in third["reply"].casefold()
    answer = third["reply"].casefold().split(reminder, 1)[0].strip()
    assert answer, "answer the question before reminding about the booking"
    assert not answer.startswith(CLINIC_PROMPT_FRAGMENT)
    assert not answer.startswith(DOCTOR_PROMPT_FRAGMENT)

    fourth = chat(
        client,
        "你好，为什么你重复之前的话呢",
        conversation_id=conversation_id,
    )
    replies.append(fourth["reply"])
    assert fourth["booking"] is None
    assert fourth["reply"].strip().casefold() != replies[0].strip().casefold()

    assert len(set(replies)) == len(replies), replies


def test_multi_turn_booking_persists_and_is_visible(client):
    conversation_id = str(uuid.uuid4())
    before = len(bookings_for(client, PATIENT_ID))

    profile_response = client.post(
        "/api/chat",
        json={
            "message": "I want to book a dermatology appointment.",
            "conversation_id": conversation_id,
        },
    )
    assert profile_response.status_code == 200, profile_response.text
    prepared = profile_response.json()
    assert prepared["intent"] == "new_booking"
    assert prepared["booking"] is None
    assert "date" in prepared["reply"].casefold()

    date_reply = chat(
        client,
        "next Monday",
        conversation_id=conversation_id,
    )
    assert date_reply["booking"] is None
    assert "time" in date_reply["reply"].casefold()

    time_reply = chat(
        client,
        "下午3点",
        conversation_id=conversation_id,
    )
    assert time_reply["booking"] is None
    assert "confirm" in time_reply["reply"].casefold()
    assert "3:00 PM" in time_reply["reply"]

    confirmed = chat(
        client,
        "好的",
        conversation_id=conversation_id,
    )
    booking = confirmed["booking"]
    assert booking is not None
    assert booking["patient_id"] == PATIENT_ID
    assert "confirmed" in confirmed["reply"].casefold()

    booked = bookings_for(client, PATIENT_ID)
    assert len(booked) == before + 1
    assert booking["booking_id"] in {
        row["booking_id"] for row in booked
    }


def test_doctor_first_booking_derives_the_clinic_without_asking(client):
    conversation_id = str(uuid.uuid4())

    response = chat(
        client,
        "I want to book with Dr. Sarah Jenkins.",
        conversation_id=conversation_id,
    )

    assert response["intent"] == "new_booking"
    assert response["booking"] is None
    assert "which clinic" not in response["reply"].casefold()
    assert "Metro Care Health Clinic" in response["reply"]
    assert "Dr. Sarah Jenkins" in response["reply"]
    assert "treatment" in response["reply"].casefold()


def test_booking_on_a_closed_day_is_refused(client):
    conversation_id = str(uuid.uuid4())

    response = chat(
        client,
        "I'd like to book a dermatology appointment next Sunday at 10am.",
        conversation_id=conversation_id,
    )

    assert response["intent"] == "new_booking"
    assert response["booking"] is None
    reply = response["reply"]
    assert "Sunday" in reply
    assert "closed" in reply.casefold()
    assert "9:00 AM - 5:00 PM" in reply
    assert "date" in reply.casefold()


def test_booking_without_time_for_the_treatment_is_refused(client):
    conversation_id = str(uuid.uuid4())

    response = chat(
        client,
        "I'd like a dermatology appointment next Tuesday at 4:45pm.",
        conversation_id=conversation_id,
    )

    assert response["intent"] == "new_booking"
    assert response["booking"] is None
    reply = response["reply"]
    assert "30 minutes" in reply
    assert "5:00 PM" in reply
    assert "close" in reply.casefold()
    assert "time" in reply.casefold()


def test_full_demo_flow_from_questions_to_visible_booking(client):
    """Exercise the five interview stages in one continuous conversation."""

    conversation_id = str(uuid.uuid4())
    before_bookings = bookings_for(client, PATIENT_ID)
    before = len(before_bookings)
    next_booking = before_bookings[0]

    # 1. Asking a clinic question.
    question = chat(
        client,
        "What are your Saturday hours?",
        conversation_id=conversation_id,
    )
    assert question["intent"] == "clinic_info"
    assert "saturday" in question["reply"].casefold()
    assert "9:00" in question["reply"] or "9 AM" in question["reply"].upper()

    # 2. Getting a grounded treatment recommendation, clinic first.
    recommendation = chat(
        client,
        "I have a skin rash on my arm. Which treatment and doctor do you recommend?",
        conversation_id=conversation_id,
    )
    assert recommendation["intent"] == "treatment_recommendation"
    recommendation_reply = recommendation["reply"]
    assert "Sunset" in recommendation_reply
    assert "Elena" in recommendation_reply
    assert "Dermatology Consultation" in recommendation_reply
    assert recommendation_reply.index("Sunset") < recommendation_reply.index(
        "Elena"
    )

    # 3. Checking existing appointments.
    existing = chat(
        client,
        "When is my next appointment?",
        conversation_id=conversation_id,
    )
    assert existing["intent"] == "booking_query"
    assert (
        next_booking["doctor_name"].split()[-1].casefold()
        in existing["reply"].casefold()
    )
    assert (
        next_booking["treatment_name"].casefold()
        in existing["reply"].casefold()
    )

    # 4. Booking through the Chatbot Agent.
    confirmation = chat(
        client,
        (
            "I'd like to book a dermatology appointment next Tuesday at 10am."
        ),
        conversation_id=conversation_id,
    )
    assert confirmation["intent"] == "new_booking"
    assert confirmation["booking"] is None
    confirmation_reply = confirmation["reply"]
    assert "confirm" in confirmation_reply.casefold()
    assert "Sunset Wellness Center" in confirmation_reply
    assert "Elena" in confirmation_reply
    assert "Dermatology Consultation" in confirmation_reply
    assert "10:00 AM" in confirmation_reply
    assert confirmation_reply.index(
        "Sunset Wellness Center"
    ) < confirmation_reply.index("Elena") < confirmation_reply.index(
        "Dermatology Consultation"
    )

    booked = chat(
        client,
        "Yes, please.",
        conversation_id=conversation_id,
    )
    booking = booked["booking"]
    assert booking is not None
    assert booking["patient_id"] == PATIENT_ID
    assert booking["clinic_name"] == "Sunset Wellness Center"
    assert booking["doctor_name"] == "Dr. Elena Rostova"
    assert booking["treatment_name"] == "Dermatology Consultation"
    assert "create_booking" in {
        call["name"] for call in booked["tool_calls"]
    }

    # 5. Verify the persisted row through the same endpoint the UI renders.
    visible = bookings_for(client, PATIENT_ID)
    assert len(visible) == before + 1
    persisted = next(
        row for row in visible if row["booking_id"] == booking["booking_id"]
    )
    assert persisted["clinic_name"] == "Sunset Wellness Center"
    assert persisted["doctor_name"] == "Dr. Elena Rostova"
    assert persisted["treatment_name"] == "Dermatology Consultation"
    assert persisted["booking_time"].startswith("10:00")


def test_transcript_summarizer_extracts_required_fields_and_flags_allergy(client):
    transcript = (
        "Doctor: What brings you in today? "
        "Patient: I have a sore throat and fever. "
        "Doctor: You have strep throat. I am prescribing penicillin 500 mg "
        "three times daily. "
        "Patient: I am allergic to penicillin. "
        "Doctor: Thank you for telling me. Do not take penicillin; I will "
        "prescribe azithromycin instead. Rest, drink fluids, and follow up "
        "in three days if you are not improving."
    )

    response = client.post(
        "/api/summarize",
        json={"transcript": transcript},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    for field in (
        "chief_complaint",
        "diagnosis",
        "medication",
        "doctor_instructions",
    ):
        assert field in body
    assert body["chief_complaint"] != "not stated"
    assert body["diagnosis"] != "not stated"
    assert body["medication"]
    assert body["doctor_instructions"]
    assert body["red_flags"]
    assert body["needs_review"] is True


def test_emergency_guidance_creates_no_booking(client):
    conversation_id = str(uuid.uuid4())
    response = chat(
        client,
        "I have chest pain",
        conversation_id=conversation_id,
    )

    assert response["intent"] == "emergency"
    assert "911" in response["reply"]
    assert response["booking"] is None


def test_allergy_conflict_is_refused(client):
    conversation_id = str(uuid.uuid4())
    response = chat(
        client,
        "Can I book an appointment for my peanut allergy?",
        conversation_id=conversation_id,
    )

    assert response["intent"] == "new_booking"
    assert "allerg" in response["reply"].casefold()
    assert response["booking"] is None
