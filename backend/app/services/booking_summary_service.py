"""Persist and fetch treatment-history summaries for chat-created bookings."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.agent.summarizer import AnthropicTranscriptSummarizer
from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError, ServiceUnavailableError
from app.models import PatientTreatmentHistory
from app.repositories.booking_repository import BookingRepository
from app.repositories.booking_summary_repository import (
    BookingSummaryRepository,
    split_summary_items,
)
from app.schemas.summary import (
    BookingSummaryRead,
    BookingSummaryRequest,
    TranscriptSummary,
)
from app.services.summary_service import apply_review_rules


class BookingSummaryService:
    """Generates a summary for a booking and stores only the structured result."""

    def __init__(
        self,
        *,
        session: Session,
        booking_repository: BookingRepository,
        summary_repository: BookingSummaryRepository,
        settings: Settings | None = None,
        summarizer: AnthropicTranscriptSummarizer | None = None,
    ) -> None:
        self._session = session
        self._booking_repository = booking_repository
        self._summary_repository = summary_repository
        self._settings = settings or get_settings()
        self._summarizer = summarizer

    async def generate(
        self,
        booking_id: str,
        request: BookingSummaryRequest,
    ) -> BookingSummaryRead:
        if not self._settings.anthropic_api_key:
            raise ServiceUnavailableError(
                "ANTHROPIC_API_KEY is not set. Configure it to enable summaries.",
                code="AI_NOT_CONFIGURED",
            )

        booking = self._booking_repository.get(booking_id)
        if booking is None:
            raise NotFoundError(
                f"Booking {booking_id} does not exist.",
                code="BOOKING_NOT_FOUND",
            )

        patient = self._booking_repository.get_patient(booking.patient_id)
        if patient is None:
            raise NotFoundError(
                f"Patient {booking.patient_id} does not exist.",
                code="PATIENT_NOT_FOUND",
            )
        allergies = [
            allergy.strip()
            for allergy in (patient.known_allergies or "").split(",")
            if allergy.strip()
        ]

        summarizer = self._summarizer or AnthropicTranscriptSummarizer(
            self._settings
        )
        summary = await summarizer.summarize(
            request.transcript,
            allergies=allergies,
        )
        summary = apply_review_rules(summary, request.transcript)

        record = self._summary_repository.save(booking, summary)
        self._session.commit()
        return self._to_read(record)

    def get(self, booking_id: str) -> BookingSummaryRead:
        if self._booking_repository.get(booking_id) is None:
            raise NotFoundError(
                f"Booking {booking_id} does not exist.",
                code="BOOKING_NOT_FOUND",
            )

        record = self._summary_repository.get(booking_id)
        if record is None:
            raise NotFoundError(
                f"No summary exists for booking {booking_id}.",
                code="BOOKING_SUMMARY_NOT_FOUND",
            )
        return self._to_read(record)

    @staticmethod
    def _to_read(record: PatientTreatmentHistory) -> BookingSummaryRead:
        return BookingSummaryRead(
            record_id=record.record_id,
            booking_id=record.booking_id or "",
            visit_date=record.visit_date,
            chief_complaint=record.chief_complaint,
            diagnosis=record.diagnosis,
            medication=split_summary_items(record.prescribed_medication),
            doctor_instructions=split_summary_items(record.doctor_notes),
            follow_up=record.follow_up,
            red_flags=list(record.red_flags or []),
            confidence=record.confidence,
            needs_review=record.needs_review,
            created_at=record.created_at,
        )
