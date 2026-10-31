"""Transcript summary request and structured medical summary schemas."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transcript: str = Field(min_length=1, max_length=20_000)
    patient_id: str | None = Field(default=None, pattern=r"^P\d{3}$")


class TranscriptSummary(BaseModel):
    """The four required assignment fields plus safety and review metadata."""

    chief_complaint: str = "not stated"
    diagnosis: str = "not stated"
    medication: list[str] = Field(default_factory=list)
    doctor_instructions: list[str] = Field(default_factory=list)
    follow_up: str | None = None
    red_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = True

    @field_validator(
        "medication",
        "doctor_instructions",
        "red_flags",
        mode="before",
    )
    @classmethod
    def _accept_a_single_string_as_a_list(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.casefold() == "not stated":
                return []
            return [stripped]
        return value

    @field_validator("medication", "doctor_instructions", "red_flags")
    @classmethod
    def _drop_blank_entries(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class BookingSummaryRequest(BaseModel):
    """Internal payload used by the agent to persist a booking summary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transcript: str = Field(min_length=1, max_length=20_000)


class BookingSummaryRead(TranscriptSummary):
    """A persisted summary, including its treatment-history identity."""

    record_id: str
    booking_id: str
    visit_date: date
    created_at: datetime
