"""Patient context response schemas used by the chat agent."""

from datetime import date

from pydantic import BaseModel

from app.schemas.booking import BookingRead


class PatientProfileRead(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    dob: date
    gender: str
    phone: str
    email: str
    known_allergies: str
    emergency_contact: str


class TreatmentHistoryRead(BaseModel):
    record_id: str
    doctor_id: str
    doctor_name: str
    visit_date: date
    chief_complaint: str
    diagnosis: str
    prescribed_medication: str
    doctor_notes: str


class PatientContextResponse(BaseModel):
    patient: PatientProfileRead
    allergies: list[str]
    bookings: list[BookingRead]
    history: list[TreatmentHistoryRead]
