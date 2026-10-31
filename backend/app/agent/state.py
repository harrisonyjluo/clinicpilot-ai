"""In-memory conversation state and booking-slot merging.

State is intentionally process-local for the demo. Restarting the backend
resets conversations; the database remains the source of truth for bookings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from threading import Lock
from typing import Any

REQUIRED_SLOTS = ("doctor_id", "treatment_id", "booking_date", "booking_time")


class BookingStage(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"


@dataclass
class BookingSlots:
    doctor_id: str | None = None
    treatment_id: str | None = None
    booking_date: date | None = None
    booking_time: time | None = None
    notes: str | None = None

    def merge(self, other: BookingSlots) -> None:
        """Fill empty slots from `other` without erasing collected values."""

        for name in ("doctor_id", "treatment_id", "booking_date", "booking_time", "notes"):
            incoming = getattr(other, name)
            if incoming is not None:
                setattr(self, name, incoming)

    def merged_with(self, other: BookingSlots) -> BookingSlots:
        merged = BookingSlots(
            doctor_id=self.doctor_id,
            treatment_id=self.treatment_id,
            booking_date=self.booking_date,
            booking_time=self.booking_time,
            notes=self.notes,
        )
        merged.merge(other)
        return merged

    def copy(self) -> BookingSlots:
        return BookingSlots(
            doctor_id=self.doctor_id,
            treatment_id=self.treatment_id,
            booking_date=self.booking_date,
            booking_time=self.booking_time,
            notes=self.notes,
        )

    def missing_slots(self) -> list[str]:
        return [name for name in REQUIRED_SLOTS if getattr(self, name) is None]

    @property
    def is_complete(self) -> bool:
        return not self.missing_slots()


@dataclass
class ConversationState:
    conversation_id: str
    patient_id: str
    allergies: list[str] = field(default_factory=list)
    stage: BookingStage = BookingStage.IDLE
    slots: BookingSlots = field(default_factory=BookingSlots)
    clinic_preference: str | None = None
    pending: BookingSlots | None = None
    last_intent: str | None = None
    last_question: str | None = None
    client_request_id: str | None = None
    profile: dict[str, Any] | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def clear_booking(self) -> None:
        self.stage = BookingStage.IDLE
        self.slots = BookingSlots()
        self.clinic_preference = None
        self.pending = None
        self.client_request_id = None
        self.last_question = None


class ConversationStore:
    """Thread-safe process-local store keyed by conversation ID."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}
        self._lock = Lock()

    def get_or_create(self, conversation_id: str, patient_id: str) -> ConversationState:
        with self._lock:
            existing = self._states.get(conversation_id)
            if existing is not None and existing.patient_id == patient_id:
                return existing
            state = ConversationState(
                conversation_id=conversation_id,
                patient_id=patient_id,
            )
            self._states[conversation_id] = state
            return state

    def reset(self, conversation_id: str) -> None:
        with self._lock:
            state = self._states.get(conversation_id)
            if state is not None:
                state.clear_booking()

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
