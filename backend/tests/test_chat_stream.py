"""SSE chat streaming contract: deltas, terminal metadata, and errors."""

import json

from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


class StreamingOrchestrator:
    """Drive the stream without calling a model or the database."""

    def __init__(self) -> None:
        self.seen: list[ChatRequest] = []

    async def handle(self, request: ChatRequest, on_text=None) -> ChatResponse:
        self.seen.append(request)
        if on_text is not None:
            await on_text("Hello ")
            await on_text("world")
        return ChatResponse(
            reply="Hello world",
            intent="smalltalk",
            conversation_id=request.conversation_id or "conv-stream",
        )


class FailingOrchestrator:
    async def handle(self, request: ChatRequest, on_text=None) -> ChatResponse:
        raise ServiceUnavailableError(
            "The AI provider could not be reached.",
            code="AI_UPSTREAM_UNAVAILABLE",
        )


def chat_settings() -> Settings:
    return Settings(
        _env_file=None,
        anthropic_api_key="test-key",
        default_patient_id="P001",
    )


def sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.strip().split("\n\n"):
        data = "\n".join(
            line[len("data:"):].lstrip()
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if data:
            events.append(json.loads(data))
    return events


def post_stream(service: ChatService, payload: dict):
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        with TestClient(app) as client:
            return client.post("/api/chat/stream", json=payload)
    finally:
        app.dependency_overrides.pop(get_chat_service, None)


def test_chat_stream_emits_incremental_deltas_then_a_terminal_response():
    service = ChatService(
        settings=chat_settings(),
        orchestrator=StreamingOrchestrator(),
    )

    response = post_stream(
        service,
        {"message": "Hello", "patient_id": "P001"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert [event["delta"] for event in events if event["type"] == "delta"] == [
        "Hello ",
        "world",
    ]
    done = [event for event in events if event["type"] == "done"]
    assert len(done) == 1
    assert done[0]["response"]["reply"] == "Hello world"
    assert done[0]["response"]["conversation_id"] == "conv-stream"


def test_chat_stream_turns_an_orchestration_failure_into_a_structured_event():
    service = ChatService(
        settings=chat_settings(),
        orchestrator=FailingOrchestrator(),
    )

    response = post_stream(
        service,
        {"message": "Hello", "patient_id": "P001"},
    )

    assert response.status_code == 200
    events = sse_events(response.text)
    assert events == [
        {
            "type": "error",
            "error": {
                "code": "AI_UPSTREAM_UNAVAILABLE",
                "message": "The AI provider could not be reached.",
            },
        }
    ]


def test_chat_stream_rejects_a_missing_api_key_before_opening_the_stream(client):
    response = client.post(
        "/api/chat/stream",
        json={"message": "Hello", "patient_id": "P001"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_NOT_CONFIGURED"


def test_chat_stream_rejects_an_empty_message(client):
    response = client.post(
        "/api/chat/stream",
        json={"message": "   ", "patient_id": "P001"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
