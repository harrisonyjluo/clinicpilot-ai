"""Booking request and response schemas."""

from datetime import date, datetime, time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BookingCreate(BaseModel):
    """Payload used by the agent's create_booking tool."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    patient_id: str = Field(pattern=r"^P\d{3}$")
    doctor_id: str = Field(pattern=r"^D\d{3}$")
    treatment_id: str = Field(pattern=r"^T\d{3}$")
    booking_date: date
    booking_time: time
    notes: str | None = Field(default=None, max_length=500)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=36)

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class BookingRead(BaseModel):
    """Booking representation returned to the UI and the agent."""

    booking_id: str
    patient_id: str
    doctor_id: str
    doctor_name: str
    specialty: str
    clinic_id: str
    clinic_name: str
    treatment_id: str
    treatment_name: str
    duration_minutes: int
    price_usd: Decimal
    booking_date: date
    booking_time: time
    status: str
    notes: str | None = None
    created_at: datetime


class BookingListResponse(BaseModel):
    patient_id: str
    count: int
    bookings: list[BookingRead]
