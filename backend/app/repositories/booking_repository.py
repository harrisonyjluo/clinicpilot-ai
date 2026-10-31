"""Booking and patient queries. This layer owns SQL only."""

from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session, selectinload

from app.core.constants import BOOKING_ID_PREFIX
from app.models import (
    Doctor,
    Patient,
    PatientBooking,
    PatientTreatmentHistory,
    Treatment,
)


class BookingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _booking_options():
        return (
            selectinload(PatientBooking.doctor).selectinload(Doctor.clinic),
            selectinload(PatientBooking.treatment),
        )

    def list_for_patient(self, patient_id: str) -> list[PatientBooking]:
        statement = (
            select(PatientBooking)
            .where(PatientBooking.patient_id == patient_id)
            .options(*self._booking_options())
            .order_by(PatientBooking.booking_date, PatientBooking.booking_time)
        )
        return list(self._session.scalars(statement).all())

    def get(self, booking_id: str) -> PatientBooking | None:
        statement = (
            select(PatientBooking)
            .where(PatientBooking.booking_id == booking_id)
            .options(*self._booking_options())
        )
        return self._session.scalar(statement)

    def get_by_client_request_id(self, client_request_id: str) -> PatientBooking | None:
        statement = (
            select(PatientBooking)
            .where(PatientBooking.client_request_id == client_request_id)
            .options(*self._booking_options())
        )
        return self._session.scalar(statement)

    def get_patient(self, patient_id: str) -> Patient | None:
        return self._session.get(Patient, patient_id)

    def get_doctor(self, doctor_id: str) -> Doctor | None:
        return self._session.get(Doctor, doctor_id)

    def get_treatment(self, treatment_id: str) -> Treatment | None:
        return self._session.get(Treatment, treatment_id)

    def get_patient_context(self, patient_id: str) -> Patient | None:
        statement = (
            select(Patient)
            .where(Patient.patient_id == patient_id)
            .options(
                selectinload(Patient.bookings)
                .selectinload(PatientBooking.doctor)
                .selectinload(Doctor.clinic),
                selectinload(Patient.bookings).selectinload(PatientBooking.treatment),
                selectinload(Patient.history).selectinload(
                    PatientTreatmentHistory.doctor
                ),
            )
        )
        return self._session.scalar(statement)

    def next_booking_id(self) -> str:
        """Return the next sequential booking ID from the current maximum."""

        max_number = self._session.scalar(
            select(
                func.max(
                    func.cast(func.substr(PatientBooking.booking_id, 2), Integer)
                )
            )
        )
        return f"{BOOKING_ID_PREFIX}{int(max_number or 1000) + 1:04d}"

    def add(self, booking: PatientBooking) -> None:
        self._session.add(booking)
