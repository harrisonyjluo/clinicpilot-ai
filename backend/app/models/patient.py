"""Patient table."""

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.booking import PatientBooking
    from app.models.history import PatientTreatmentHistory


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dob: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(String(30), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    known_allergies: Mapped[str] = mapped_column(Text, nullable=False)
    emergency_contact: Mapped[str] = mapped_column(Text, nullable=False)

    bookings: Mapped[list["PatientBooking"]] = relationship(back_populates="patient")
    history: Mapped[list["PatientTreatmentHistory"]] = relationship(back_populates="patient")
