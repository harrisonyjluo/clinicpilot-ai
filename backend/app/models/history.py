"""Patient treatment history table.

A row is either seeded clinical history or the structured result of summarizing
a chat-created booking. AI-generated rows carry a unique ``booking_id`` link so
the bookings UI can open the corresponding history record. The raw transcript is
never stored here.
"""

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.doctor import Doctor
    from app.models.patient import Patient


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PatientTreatmentHistory(Base):
    __tablename__ = "patient_treatment_history"
    __table_args__ = (
        Index("ix_history_patient_id", "patient_id"),
        Index("ix_history_booking_id", "booking_id", unique=True),
    )

    record_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.doctor_id"), nullable=False)
    visit_date: Mapped[date] = mapped_column(Date, nullable=False)
    chief_complaint: Mapped[str] = mapped_column(Text, nullable=False)
    diagnosis: Mapped[str] = mapped_column(Text, nullable=False)
    prescribed_medication: Mapped[str] = mapped_column(Text, nullable=False)
    doctor_notes: Mapped[str] = mapped_column(Text, nullable=False)

    booking_id: Mapped[str | None] = mapped_column(
        String(10),
        ForeignKey("patient_bookings.booking_id", ondelete="SET NULL"),
        nullable=True,
    )
    follow_up: Mapped[str | None] = mapped_column(Text, nullable=True)
    red_flags: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    needs_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )

    patient: Mapped["Patient"] = relationship(back_populates="history")
    doctor: Mapped["Doctor"] = relationship(back_populates="history")
