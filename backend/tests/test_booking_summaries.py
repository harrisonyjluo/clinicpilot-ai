"""Booking summaries are persisted as patient treatment-history records."""

from datetime import date, datetime, time, timezone

import pytest
from sqlalchemy import func, select

from app.api.dependencies import get_booking_summary_service
from app.core.config import Settings
from app.core.errors import NotFoundError, ServiceUnavailableError
from app.main import app
from app.models import Base, PatientTreatmentHistory
from app.repositories.booking_repository import BookingRepository
from app.repositories.booking_summary_repository import BookingSummaryRepository
from app.schemas.booking import BookingCreate
from app.schemas.summary import (
    BookingSummaryRead,
    BookingSummaryRequest,
    TranscriptSummary,
)
from app.services.booking_service import BookingService
from app.services.booking_summary_service import BookingSummaryService

LONG_TRANSCRIPT = (
    "Doctor: What brings you in today? "
    "Patient: I have had an itchy rash on my left arm for four days. "
    "Doctor: Please keep the area clean and avoid new lotions until we examine it. "
    "Patient: I am allergic to penicillin and peanuts. "
    "Doctor: Thank you; we will avoid those."
)


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "anthropic_api_key": "test-key",
        "default_patient_id": "P001",
    }
    values.update(overrides)
    return Settings(**values)


def transcript_summary(**overrides) -> TranscriptSummary:
    values = {
        "chief_complaint": "itchy rash on the left arm",
        "diagnosis": "not stated",
        "medication": ["hydrocortisone cream"],
        "doctor_instructions": ["keep the area clean", "avoid new lotions"],
        "follow_up": "return if the rash spreads",
        "red_flags": [],
        "confidence": 0.91,
        "needs_review": False,
    }
    values.update(overrides)
    return TranscriptSummary(**values)


class FakeSummarizer:
    def __init__(self, result: TranscriptSummary | None = None) -> None:
        self.result = result or transcript_summary()
        self.calls: list[dict] = []

    async def summarize(self, transcript: str, *, allergies: list[str]):
        self.calls.append({"transcript": transcript, "allergies": allergies})
        return self.result


def make_booking(session):
    service = BookingService(session, BookingRepository(session))
    booking, created = service.create_booking(
        BookingCreate(
            patient_id="P001",
            doctor_id="D103",
            treatment_id="T003",
            booking_date=date(2026, 11, 20),
            booking_time=time(10, 0),
            notes="Rash on the left arm",
        )
    )
    assert created is True
    return booking


def make_service(
    session,
    *,
    summarizer: FakeSummarizer | None = None,
    settings_override: Settings | None = None,
) -> BookingSummaryService:
    return BookingSummaryService(
        session=session,
        booking_repository=BookingRepository(session),
        summary_repository=BookingSummaryRepository(session),
        settings=settings_override or settings(),
        summarizer=summarizer or FakeSummarizer(),
    )


def history_count(session) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(PatientTreatmentHistory)
        )
        or 0
    )


def test_booking_summaries_is_no_longer_part_of_the_schema():
    assert "booking_summaries" not in Base.metadata.tables


@pytest.mark.asyncio
async def test_generate_persists_the_summary_as_patient_treatment_history(db_session):
    booking = make_booking(db_session)
    summarizer = FakeSummarizer()
    service = make_service(db_session, summarizer=summarizer)
    before = history_count(db_session)

    result = await service.generate(
        booking.booking_id,
        BookingSummaryRequest(transcript=LONG_TRANSCRIPT),
    )

    assert result.record_id == "H9003"
    assert result.booking_id == booking.booking_id
    assert result.visit_date == date(2026, 11, 20)
    assert result.chief_complaint == "itchy rash on the left arm"
    assert result.diagnosis == "not stated"
    assert result.medication == ["hydrocortisone cream"]
    assert result.doctor_instructions == [
        "keep the area clean",
        "avoid new lotions",
    ]
    assert summarizer.calls[0]["allergies"] == ["Penicillin", "Peanuts"]
    assert summarizer.calls[0]["transcript"] == LONG_TRANSCRIPT
    assert history_count(db_session) == before + 1

    stored = BookingSummaryRepository(db_session).get(booking.booking_id)
    assert isinstance(stored, PatientTreatmentHistory)
    assert stored.record_id == "H9003"
    assert stored.booking_id == booking.booking_id
    assert stored.patient_id == "P001"
    assert stored.doctor_id == "D103"
    assert stored.visit_date == date(2026, 11, 20)
    assert stored.chief_complaint == "itchy rash on the left arm"
    assert stored.diagnosis == "not stated"
    assert stored.prescribed_medication == "hydrocortisone cream"
    assert stored.doctor_notes == "keep the area clean\navoid new lotions"
    assert stored.follow_up == "return if the rash spreads"
    assert stored.red_flags == []
    assert stored.confidence == 0.91
    assert stored.needs_review is False
    assert stored.created_at is not None
    assert "transcript" not in PatientTreatmentHistory.__table__.columns


@pytest.mark.asyncio
async def test_generate_updates_the_same_history_record_instead_of_duplicating(db_session):
    booking = make_booking(db_session)
    summarizer = FakeSummarizer()
    service = make_service(db_session, summarizer=summarizer)
    before = history_count(db_session)

    first = await service.generate(
        booking.booking_id,
        BookingSummaryRequest(transcript=LONG_TRANSCRIPT),
    )
    summarizer.result = transcript_summary(
        chief_complaint="rash improved",
        diagnosis="contact dermatitis",
    )
    second = await service.generate(
        booking.booking_id,
        BookingSummaryRequest(transcript=LONG_TRANSCRIPT),
    )

    assert second.record_id == first.record_id
    assert second.diagnosis == "contact dermatitis"
    assert history_count(db_session) == before + 1
    stored = BookingSummaryRepository(db_session).get(booking.booking_id)
    assert stored is not None
    assert stored.record_id == first.record_id
    assert stored.chief_complaint == "rash improved"


def test_a_seeded_booking_without_a_summary_reports_not_found(db_session):
    service = make_service(db_session)
    before = history_count(db_session)

    with pytest.raises(NotFoundError) as excinfo:
        service.get("B1001")

    assert excinfo.value.code == "BOOKING_SUMMARY_NOT_FOUND"
    assert history_count(db_session) == before


@pytest.mark.asyncio
async def test_unknown_booking_is_rejected(db_session):
    service = make_service(db_session)
    before = history_count(db_session)

    with pytest.raises(NotFoundError) as excinfo:
        await service.generate(
            "B9999",
            BookingSummaryRequest(transcript=LONG_TRANSCRIPT),
        )

    assert excinfo.value.code == "BOOKING_NOT_FOUND"
    assert history_count(db_session) == before


@pytest.mark.asyncio
async def test_short_transcript_is_flagged_for_review(db_session):
    booking = make_booking(db_session)
    service = make_service(
        db_session,
        summarizer=FakeSummarizer(
            transcript_summary(confidence=0.99, needs_review=False)
        ),
    )

    result = await service.generate(
        booking.booking_id,
        BookingSummaryRequest(transcript="short"),
    )

    assert result.needs_review is True


@pytest.mark.asyncio
async def test_missing_api_key_is_rejected_before_the_model_is_called(db_session):
    booking = make_booking(db_session)
    summarizer = FakeSummarizer()
    service = make_service(
        db_session,
        summarizer=summarizer,
        settings_override=settings(anthropic_api_key=None),
    )
    before = history_count(db_session)

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await service.generate(
            booking.booking_id,
            BookingSummaryRequest(transcript=LONG_TRANSCRIPT),
        )

    assert excinfo.value.code == "AI_NOT_CONFIGURED"
    assert summarizer.calls == []
    assert history_count(db_session) == before


def booking_summary_read(booking_id: str = "B1005") -> BookingSummaryRead:
    return BookingSummaryRead(
        record_id="H9003",
        booking_id=booking_id,
        visit_date=date(2026, 11, 20),
        chief_complaint="itchy rash on the left arm",
        diagnosis="not stated",
        medication=["hydrocortisone cream"],
        doctor_instructions=["keep the area clean"],
        follow_up=None,
        red_flags=[],
        confidence=0.91,
        needs_review=False,
        created_at=datetime.now(timezone.utc),
    )


class StubBookingSummaryService:
    def __init__(self, result: BookingSummaryRead | None = None) -> None:
        self.result = result or booking_summary_read()
        self.generate_calls: list[tuple[str, BookingSummaryRequest]] = []

    async def generate(self, booking_id: str, request: BookingSummaryRequest):
        self.generate_calls.append((booking_id, request))
        return self.result

    def get(self, booking_id: str):
        return self.result


@pytest.fixture()
def use_summary_service():
    def apply(stub):
        app.dependency_overrides[get_booking_summary_service] = lambda: stub

    yield apply
    app.dependency_overrides.pop(get_booking_summary_service, None)


def test_get_returns_the_linked_treatment_history_record(client, use_summary_service):
    use_summary_service(StubBookingSummaryService())

    response = client.get("/api/bookings/B1005/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["record_id"] == "H9003"
    assert body["booking_id"] == "B1005"
    assert body["visit_date"] == "2026-11-20"
    assert body["chief_complaint"] == "itchy rash on the left arm"
    assert body["doctor_instructions"] == ["keep the area clean"]


def test_seeded_bookings_without_a_summary_return_structured_404(client):
    response = client.get("/api/bookings/B1001/summary")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BOOKING_SUMMARY_NOT_FOUND"


def test_post_generates_a_summary_for_the_booking(client, use_summary_service):
    stub = StubBookingSummaryService()
    use_summary_service(stub)

    response = client.post(
        "/api/bookings/B1005/summary",
        json={"transcript": LONG_TRANSCRIPT},
    )

    assert response.status_code == 200
    assert response.json()["record_id"] == "H9003"
    assert response.json()["booking_id"] == "B1005"
    assert len(stub.generate_calls) == 1
    booking_id, request = stub.generate_calls[0]
    assert booking_id == "B1005"
    assert request.transcript == LONG_TRANSCRIPT


def test_post_rejects_an_empty_transcript(client, use_summary_service):
    use_summary_service(StubBookingSummaryService())

    response = client.post(
        "/api/bookings/B1005/summary",
        json={"transcript": ""},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_post_rejects_unknown_fields(client, use_summary_service):
    use_summary_service(StubBookingSummaryService())

    response = client.post(
        "/api/bookings/B1005/summary",
        json={"transcript": LONG_TRANSCRIPT, "patient_id": "P001"},
    )

    assert response.status_code == 422
