"""Doctor table."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.booking import PatientBooking
    from app.models.clinic import Clinic
    from app.models.history import PatientTreatmentHistory


class Doctor(Base):
    __tablename__ = "doctors"
    __table_args__ = (Index("ix_doctors_clinic_id", "clinic_id"),)

    doctor_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    clinic_id: Mapped[str] = mapped_column(ForeignKey("clinics.clinic_id"), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    specialty: Mapped[str] = mapped_column(String(100), nullable=False)
    languages_spoken: Mapped[str] = mapped_column(String(200), nullable=False)
    consultation_fee_usd: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    bio: Mapped[str] = mapped_column(Text, nullable=False)

    clinic: Mapped["Clinic"] = relationship(back_populates="doctors")
    bookings: Mapped[list["PatientBooking"]] = relationship(back_populates="doctor")
    history: Mapped[list["PatientTreatmentHistory"]] = relationship(back_populates="doctor")
