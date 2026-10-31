"""Booking use cases: validation, persistence, idempotency, and mapping."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.models import PatientBooking
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import BookingCreate, BookingListResponse, BookingRead
from app.schemas.patient import (
    PatientContextResponse,
    PatientProfileRead,
    TreatmentHistoryRead,
)


class BookingService:
    def __init__(self, session: Session, repository: BookingRepository) -> None:
        self._session = session
        self._repository = repository

    def list_bookings(self, patient_id: str) -> BookingListResponse:
        bookings = self._repository.list_for_patient(patient_id)
        items = [self._to_read(booking) for booking in bookings]
        return BookingListResponse(
            patient_id=patient_id,
            count=len(items),
            bookings=items,
        )

    def create_booking(self, payload: BookingCreate) -> tuple[BookingRead, bool]:
        if self._repository.get_patient(payload.patient_id) is None:
            raise NotFoundError(
                f"Patient {payload.patient_id} does not exist.",
                code="PATIENT_NOT_FOUND",
                details={"field": "patient_id"},
            )
        if self._repository.get_doctor(payload.doctor_id) is None:
            raise NotFoundError(
                f"Doctor {payload.doctor_id} does not exist.",
                code="DOCTOR_NOT_FOUND",
                details={"field": "doctor_id"},
            )
        if self._repository.get_treatment(payload.treatment_id) is None:
            raise NotFoundError(
                f"Treatment {payload.treatment_id} does not exist.",
                code="TREATMENT_NOT_FOUND",
                details={"field": "treatment_id"},
            )

        if payload.client_request_id is not None:
            existing = self._repository.get_by_client_request_id(
                payload.client_request_id
            )
            if existing is not None:
                return self._to_read(existing), False

        booking = PatientBooking(
            booking_id=self._repository.next_booking_id(),
            patient_id=payload.patient_id,
            doctor_id=payload.doctor_id,
            treatment_id=payload.treatment_id,
            booking_date=payload.booking_date,
            booking_time=payload.booking_time,
            status="Confirmed",
            notes=payload.notes,
            client_request_id=payload.client_request_id,
        )
        self._repository.add(booking)

        try:
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            if payload.client_request_id is not None:
                existing = self._repository.get_by_client_request_id(
                    payload.client_request_id
                )
                if existing is not None:
                    return self._to_read(existing), False
            raise ConflictError(
                "The booking could not be saved.",
                code="BOOKING_CONFLICT",
            ) from exc

        return self._to_read(booking), True

    def get_patient_context(self, patient_id: str) -> PatientContextResponse:
        patient = self._repository.get_patient_context(patient_id)
        if patient is None:
            raise NotFoundError(
                f"Patient {patient_id} does not exist.",
                code="PATIENT_NOT_FOUND",
            )

        bookings = sorted(
            patient.bookings,
            key=lambda booking: (booking.booking_date, booking.booking_time),
        )
        history = sorted(
            patient.history,
            key=lambda record: record.visit_date,
            reverse=True,
        )
        return PatientContextResponse(
            patient=PatientProfileRead(
                patient_id=patient.patient_id,
                first_name=patient.first_name,
                last_name=patient.last_name,
                dob=patient.dob,
                gender=patient.gender,
                phone=patient.phone,
                email=patient.email,
                known_allergies=patient.known_allergies,
                emergency_contact=patient.emergency_contact,
            ),
            allergies=[
                allergy.strip()
                for allergy in patient.known_allergies.split(",")
                if allergy.strip()
            ],
            bookings=[self._to_read(booking) for booking in bookings],
            history=[
                TreatmentHistoryRead(
                    record_id=record.record_id,
                    doctor_id=record.doctor_id,
                    doctor_name=record.doctor.full_name,
                    visit_date=record.visit_date,
                    chief_complaint=record.chief_complaint,
                    diagnosis=record.diagnosis,
                    prescribed_medication=record.prescribed_medication,
                    doctor_notes=record.doctor_notes,
                )
                for record in history
            ],
        )

    @staticmethod
    def _to_read(booking: PatientBooking) -> BookingRead:
        doctor = booking.doctor
        treatment = booking.treatment
        clinic = doctor.clinic
        return BookingRead(
            booking_id=booking.booking_id,
            patient_id=booking.patient_id,
            doctor_id=booking.doctor_id,
            doctor_name=doctor.full_name,
            specialty=doctor.specialty,
            clinic_id=clinic.clinic_id,
            clinic_name=clinic.clinic_name,
            treatment_id=booking.treatment_id,
            treatment_name=treatment.treatment_name,
            duration_minutes=treatment.duration_minutes,
            price_usd=treatment.price_usd,
            booking_date=booking.booking_date,
            booking_time=booking.booking_time,
            status=booking.status,
            notes=booking.notes,
            created_at=booking.created_at,
        )
