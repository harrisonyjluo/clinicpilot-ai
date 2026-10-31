"""Chat service boundary: resolve the patient before orchestration."""

import pytest

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


class RecordingOrchestrator:
    def __init__(self) -> None:
        self.seen: list[ChatRequest] = []

    async def handle(self, request: ChatRequest) -> ChatResponse:
        self.seen.append(request)
        return ChatResponse(
            reply="ok",
            intent="smalltalk",
            conversation_id=request.conversation_id or "test-conversation",
        )


def settings_with(**overrides) -> Settings:
    values = {"anthropic_api_key": "test-key", "default_patient_id": None}
    values.update(overrides)
    return Settings(_env_file=None, **values)


async def test_configured_default_patient_is_applied_when_the_request_omits_it():
    orchestrator = RecordingOrchestrator()
    service = ChatService(
        settings=settings_with(default_patient_id="P123"),
        orchestrator=orchestrator,
    )

    await service.handle(ChatRequest(message="Hello"))

    assert orchestrator.seen[0].patient_id == "P123"


async def test_explicit_patient_id_wins_over_the_configured_default():
    orchestrator = RecordingOrchestrator()
    service = ChatService(
        settings=settings_with(default_patient_id="P123"),
        orchestrator=orchestrator,
    )

    await service.handle(ChatRequest(message="Hello", patient_id="P002"))

    assert orchestrator.seen[0].patient_id == "P002"


async def test_missing_patient_without_a_configured_default_is_rejected():
    service = ChatService(
        settings=settings_with(default_patient_id=None),
        orchestrator=RecordingOrchestrator(),
    )

    with pytest.raises(BadRequestError) as excinfo:
        await service.handle(ChatRequest(message="Hello"))

    assert excinfo.value.code == "PATIENT_ID_REQUIRED"
