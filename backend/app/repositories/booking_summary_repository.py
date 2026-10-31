"""Persistence for booking-linked structured summaries.

The system of record is ``patient_treatment_history``: the four required
summary fields map onto its clinical columns, and the unique ``booking_id``
link lets the bookings UI retrieve the record. This layer owns SQL only.
"""

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from app.core.constants import HISTORY_ID_PREFIX
from app.models import PatientBooking, PatientTreatmentHistory
from app.schemas.summary import TranscriptSummary


def join_summary_items(items: Iterable[str]) -> str:
    """Store a structured list as newline-separated clinical text."""

    values = [item.strip() for item in items if item and item.strip()]
    return "\n".join(values) if values else "not stated"


def split_summary_items(value: str | None) -> list[str]:
    """Read newline-separated clinical text back into a structured list."""

    if not value or value.strip().casefold() == "not stated":
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


class BookingSummaryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, booking_id: str) -> PatientTreatmentHistory | None:
        statement = select(PatientTreatmentHistory).where(
            PatientTreatmentHistory.booking_id == booking_id
        )
        return self._session.scalar(statement)

    def save(
        self,
        booking: PatientBooking,
        summary: TranscriptSummary,
    ) -> PatientTreatmentHistory:
        record = self.get(booking.booking_id)
        if record is None:
            record = PatientTreatmentHistory(
                record_id=self.next_record_id(),
                booking_id=booking.booking_id,
                patient_id=booking.patient_id,
                doctor_id=booking.doctor_id,
                visit_date=booking.booking_date,
                created_at=datetime.now(timezone.utc),
            )
            self._session.add(record)

        record.patient_id = booking.patient_id
        record.doctor_id = booking.doctor_id
        record.visit_date = booking.booking_date
        record.chief_complaint = summary.chief_complaint
        record.diagnosis = summary.diagnosis
        record.prescribed_medication = join_summary_items(summary.medication)
        record.doctor_notes = join_summary_items(summary.doctor_instructions)
        record.follow_up = summary.follow_up
        record.red_flags = list(summary.red_flags)
        record.confidence = summary.confidence
        record.needs_review = summary.needs_review
        self._session.flush()
        return record

    def next_record_id(self) -> str:
        """Return the next sequential history ID from the current maximum."""

        max_number = self._session.scalar(
            select(
                func.max(
                    func.cast(
                        func.substr(PatientTreatmentHistory.record_id, 2),
                        Integer,
                    )
                )
            )
        )
        return f"{HISTORY_ID_PREFIX}{int(max_number or 9000) + 1:04d}"
