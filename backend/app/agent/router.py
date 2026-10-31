"""LLM-driven intent routing with catalog validation.

Every user message is interpreted by an LLM classifier that returns a
``RouteAnalysis`` structured object. This module validates that object against
the clinic catalog, merges the validated booking slots into conversation state,
and produces a ``RouterDecision``.

There is deliberately no keyword, regex, substring, or token matching over user
text here. Natural-language understanding belongs to the classifier; this module
only validates structured values (doctor/treatment IDs, ISO dates and times).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, field_validator

from app.agent.state import BookingSlots, ConversationState
from app.rag.index import KnowledgeBase


class Intent(str, Enum):
    EMERGENCY = "emergency"
    BOOKING_QUERY = "booking_query"
    NEW_BOOKING = "new_booking"
    BOOKING_CONFIRMATION = "booking_confirmation"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    CLINIC_INFO = "clinic_info"
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"


class RouteAnalysis(BaseModel):
    """Structured routing output produced by the LLM classifier.

    Only ``intent`` is required. Gateways that do not enforce JSON-schema
    defaults may omit ``confidence``; the default keeps the turn routable
    instead of turning a provider quirk into a 500.
    """

    intent: Intent
    confidence: float = 0.5
    clinic_id: str | None = None
    clinic_name: str | None = None
    doctor_id: str | None = None
    doctor_name: str | None = None
    treatment_id: str | None = None
    treatment_name: str | None = None
    booking_date: str | None = None
    booking_time: str | None = None
    notes: str | None = None
    safety_flag: Literal[
        "emergency",
        "allergy_conflict",
        "out_of_scope",
    ] | None = None

    @field_validator("intent", mode="before")
    @classmethod
    def _normalize_intent(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, number))

    @field_validator("safety_flag", mode="before")
    @classmethod
    def _normalize_safety_flag(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null", "n/a"}:
                return None
            return normalized
        return value


class RouteClassifier(Protocol):
    """The LLM seam used by the router; tests inject a scripted classifier."""

    async def classify(
        self,
        message: str,
        state: ConversationState,
        *,
        today: date,
    ) -> RouteAnalysis: ...


@dataclass
class RouterDecision:
    intent: Intent
    confidence: float
    entities: BookingSlots = field(default_factory=BookingSlots)
    missing_slots: list[str] = field(default_factory=list)
    clinic_id: str | None = None
    safety_flag: str | None = None
    unknown_clinic_name: str | None = None
    unknown_doctor_name: str | None = None
    unknown_treatment_name: str | None = None


def _normalize_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return normalized or None


def _normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _parse_iso_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_iso_time(value: str | None) -> time | None:
    if value is None:
        return None
    try:
        return time.fromisoformat(value.strip())
    except ValueError:
        return None


class IntentRouter:
    """Dispatches the classifier's structured analysis to a RouterDecision."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        classifier: RouteClassifier,
    ) -> None:
        self.knowledge = knowledge
        self.classifier = classifier

    async def route(
        self,
        message: str,
        state: ConversationState,
        *,
        today: date | None = None,
    ) -> RouterDecision:
        today = today or date.today()
        analysis = await self.classifier.classify(
            message,
            state,
            today=today,
        )

        entities = BookingSlots(
            booking_date=_parse_iso_date(analysis.booking_date),
            booking_time=_parse_iso_time(analysis.booking_time),
            notes=(analysis.notes or "").strip()[:500] or None,
        )

        clinic_id = _normalize_identifier(analysis.clinic_id)
        valid_clinic_id: str | None = None
        unknown_clinic: str | None = None
        if clinic_id is not None and self.knowledge.get_clinic(clinic_id):
            valid_clinic_id = clinic_id
        elif clinic_id is not None or analysis.clinic_name:
            unknown_clinic = _normalize_name(analysis.clinic_name) or clinic_id

        doctor_id = _normalize_identifier(analysis.doctor_id)
        unknown_doctor: str | None = None
        if doctor_id is not None and self.knowledge.get_doctor(doctor_id):
            entities.doctor_id = doctor_id
        elif doctor_id is not None or analysis.doctor_name:
            unknown_doctor = _normalize_name(analysis.doctor_name) or doctor_id

        treatment_id = _normalize_identifier(analysis.treatment_id)
        unknown_treatment: str | None = None
        if treatment_id is not None and self.knowledge.get_treatment(treatment_id):
            entities.treatment_id = treatment_id
        elif treatment_id is not None or analysis.treatment_name:
            unknown_treatment = (
                _normalize_name(analysis.treatment_name) or treatment_id
            )

        if analysis.intent is Intent.NEW_BOOKING:
            entities = self._complete_pair(
                entities,
                unknown_doctor,
                unknown_treatment,
                explicit_clinic_id=valid_clinic_id,
            )

        intent = analysis.intent
        safety_flag = analysis.safety_flag
        if intent is Intent.EMERGENCY:
            safety_flag = "emergency"
        elif safety_flag == "emergency":
            intent = Intent.EMERGENCY
        elif safety_flag == "out_of_scope":
            intent = Intent.OUT_OF_SCOPE
        elif safety_flag == "allergy_conflict" and intent not in {
            Intent.NEW_BOOKING,
            Intent.TREATMENT_RECOMMENDATION,
        }:
            intent = Intent.TREATMENT_RECOMMENDATION

        missing_slots: list[str] = []
        if intent is Intent.NEW_BOOKING:
            missing_slots = state.slots.merged_with(entities).missing_slots()

        return RouterDecision(
            intent=intent,
            confidence=analysis.confidence,
            entities=entities,
            missing_slots=missing_slots,
            clinic_id=valid_clinic_id,
            safety_flag=safety_flag,
            unknown_clinic_name=unknown_clinic,
            unknown_doctor_name=unknown_doctor,
            unknown_treatment_name=unknown_treatment,
        )

    def _complete_pair(
        self,
        entities: BookingSlots,
        unknown_doctor: str | None,
        unknown_treatment: str | None,
        *,
        explicit_clinic_id: str | None,
    ) -> BookingSlots:
        """Fill a missing doctor/treatment from the catalog when unambiguous."""

        if (
            unknown_doctor is None
            and entities.doctor_id is None
            and entities.treatment_id is not None
        ):
            doctor = self.knowledge.doctor_for_treatment(
                entities.treatment_id,
                clinic_id=explicit_clinic_id,
            )
            if doctor is not None:
                entities.doctor_id = doctor["doctor_id"]

        if (
            unknown_treatment is None
            and entities.treatment_id is None
            and entities.doctor_id is not None
        ):
            treatments = self.knowledge.treatments_for_doctor(
                entities.doctor_id
            )
            if len(treatments) == 1:
                entities.treatment_id = treatments[0]["treatment_id"]

        return entities
