"""Chat request and response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.booking import BookingRead


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    patient_id: str | None = Field(default=None, pattern=r"^P\d{3}$")
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)


class ToolCallRead(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    intent: str
    conversation_id: str
    booking: BookingRead | None = None
    tool_calls: list[ToolCallRead] = Field(default_factory=list)
