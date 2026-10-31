"""Patient booking table."""

from datetime import date, datetime, time, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, ForeignKey, Index, String, Text, Time
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.doctor import Doctor
    from app.models.patient import Patient
    from app.models.treatment import Treatment


class PatientBooking(Base):
    __tablename__ = "patient_bookings"
    __table_args__ = (Index("ix_patient_bookings_patient_date", "patient_id", "booking_date"),)

    booking_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.doctor_id"), nullable=False)
    treatment_id: Mapped[str] = mapped_column(
        ForeignKey("treatments.treatment_id"), nullable=False
    )
    booking_date: Mapped[date] = mapped_column(Date, nullable=False)
    booking_time: Mapped[time] = mapped_column(Time, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="Confirmed")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    client_request_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, unique=True
    )

    patient: Mapped["Patient"] = relationship(back_populates="bookings")
    doctor: Mapped["Doctor"] = relationship(back_populates="bookings")
    treatment: Mapped["Treatment"] = relationship(back_populates="bookings")
