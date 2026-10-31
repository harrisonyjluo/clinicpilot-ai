"""Anthropic adapter configuration and provider-error translation."""

import json
from datetime import date
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.agent.llm import AnthropicIntentClassifier, AnthropicLLM
from app.agent.router import Intent
from app.agent.state import ConversationState
from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.rag.index import KnowledgeBase

PROVIDER_BASE_URL = "https://provider.example/anthropic"


class RecordingAnthropic:
    """Capture constructor kwargs instead of opening a real client."""

    instances: list["RecordingAnthropic"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        RecordingAnthropic.instances.append(self)


class FailingMessages:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def create(self, **_kwargs):
        raise self._error

    async def parse(self, **_kwargs):
        raise self._error


class FailingAnthropic:
    def __init__(self, error: Exception) -> None:
        self.messages = FailingMessages(error)


class TextMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
        )


class TextAnthropic:
    def __init__(self, text: str) -> None:
        self.messages = TextMessages(text)


@pytest.fixture(scope="module")
def knowledge() -> KnowledgeBase:
    return KnowledgeBase.load_default()


def llm_settings(**overrides) -> Settings:
    values = {
        "anthropic_api_key": "test-key",
        "anthropic_base_url": PROVIDER_BASE_URL,
        "anthropic_model": "test-model",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def provider_status_error(error_type, status_code: int):
    request = httpx2.Request("POST", f"{PROVIDER_BASE_URL}/v1/messages")
    response = httpx2.Response(status_code, request=request)
    return error_type(
        f"provider returned {status_code}",
        response=response,
        body={"error": {"type": "provider_error", "message": "rejected"}},
    )


@pytest.fixture(autouse=True)
def reset_recording_client():
    RecordingAnthropic.instances.clear()
    yield
    RecordingAnthropic.instances.clear()


def test_llm_client_uses_the_configured_provider_base_url(monkeypatch):
    monkeypatch.setattr("app.agent.llm.AsyncAnthropic", RecordingAnthropic)

    AnthropicLLM(llm_settings())

    assert RecordingAnthropic.instances[0].kwargs["base_url"] == PROVIDER_BASE_URL
    assert RecordingAnthropic.instances[0].kwargs["api_key"] == "test-key"


def test_intent_classifier_uses_the_configured_provider_base_url(
    monkeypatch,
    knowledge,
):
    monkeypatch.setattr("app.agent.llm.AsyncAnthropic", RecordingAnthropic)

    AnthropicIntentClassifier(llm_settings(), knowledge)

    assert RecordingAnthropic.instances[0].kwargs["base_url"] == PROVIDER_BASE_URL


async def test_classifier_parses_json_text_into_route_analysis(knowledge):
    text = json.dumps(
        {
            "intent": "new_booking",
            "confidence": 0.91,
            "doctor_id": "D103",
            "treatment_id": "T003",
            "booking_date": "2026-11-01",
            "booking_time": "15:00",
        }
    )
    client = TextAnthropic(text)
    classifier = AnthropicIntentClassifier(
        llm_settings(),
        knowledge,
        client=client,
    )
    state = ConversationState(
        conversation_id="conv-1",
        patient_id="P001",
        allergies=["Penicillin", "Peanuts"],
    )

    result = await classifier.classify(
        "我想预约明天的皮肤科医生",
        state,
        today=date(2026, 10, 31),
    )

    assert result.intent is Intent.NEW_BOOKING
    assert result.confidence == 0.91
    assert result.doctor_id == "D103"
    assert result.treatment_id == "T003"
    kwargs = client.messages.kwargs
    assert kwargs["messages"] == [
        {"role": "user", "content": "我想预约明天的皮肤科医生"}
    ]
    system = kwargs["system"]
    assert "2026-10-31" in system
    assert "D103" in system
    assert "T003" in system
    assert "Penicillin" in system
    assert "booking_stage" in system
    assert "confidence" in system


async def test_classifier_prompt_includes_clinics_and_clinic_context(knowledge):
    text = json.dumps(
        {
            "intent": "new_booking",
            "confidence": 0.9,
            "clinic_id": "C001",
        }
    )
    client = TextAnthropic(text)
    classifier = AnthropicIntentClassifier(
        llm_settings(),
        knowledge,
        client=client,
    )
    state = ConversationState(
        conversation_id="conv-1",
        patient_id="P001",
        clinic_preference="C002",
    )

    result = await classifier.classify(
        "Metro Care please",
        state,
        today=date(2026, 10, 31),
    )

    assert result.clinic_id == "C001"
    system = client.messages.kwargs["system"]
    assert "Valid clinics" in system
    assert "Metro Care Health Clinic" in system
    assert "Sunset Wellness Center" in system
    assert '"clinic_id": "C002"' in system
    assert "clinic_name" in system


async def test_classifier_accepts_a_missing_confidence(knowledge):
    """The configured gateway may omit schema defaults; do not fail the turn."""

    classifier = AnthropicIntentClassifier(
        llm_settings(),
        knowledge,
        client=TextAnthropic('{"intent": "emergency"}'),
    )

    result = await classifier.classify(
        "I have chest pain",
        ConversationState(conversation_id="c", patient_id="P001"),
        today=date(2026, 10, 31),
    )

    assert result.intent is Intent.EMERGENCY
    assert result.confidence == 0.5


async def test_classifier_extracts_json_from_a_fenced_block(knowledge):
    classifier = AnthropicIntentClassifier(
        llm_settings(),
        knowledge,
        client=TextAnthropic(
            'Here is the result:\n```json\n'
            '{"intent": "smalltalk", "confidence": 0.9}\n'
            "```"
        ),
    )

    result = await classifier.classify(
        "hello",
        ConversationState(conversation_id="c", patient_id="P001"),
        today=date(2026, 10, 31),
    )

    assert result.intent is Intent.SMALLTALK
    assert result.confidence == 0.9


async def test_classifier_rejects_unusable_text(knowledge):
    classifier = AnthropicIntentClassifier(
        llm_settings(),
        knowledge,
        client=TextAnthropic("I cannot classify that message."),
    )

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await classifier.classify(
            "hello",
            ConversationState(conversation_id="c", patient_id="P001"),
            today=date(2026, 10, 31),
        )

    assert excinfo.value.code == "AI_RESPONSE_INVALID"


async def test_classifier_translates_provider_errors(knowledge):
    error = provider_status_error(anthropic.PermissionDeniedError, 403)
    classifier = AnthropicIntentClassifier(
        llm_settings(),
        knowledge,
        client=FailingAnthropic(error),
    )

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await classifier.classify(
            "hello",
            ConversationState(conversation_id="c", patient_id="P001"),
            today=date(2026, 10, 31),
        )

    assert excinfo.value.code == "AI_AUTHENTICATION_FAILED"


async def test_permission_denied_is_reported_as_a_configuration_error():
    error = provider_status_error(anthropic.PermissionDeniedError, 403)
    llm = AnthropicLLM(llm_settings(), client=FailingAnthropic(error))

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await llm.complete(
            system="system",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "AI_AUTHENTICATION_FAILED"
    assert "403" in excinfo.value.message


async def test_provider_connection_failure_is_reported_as_unavailable():
    request = httpx2.Request("POST", f"{PROVIDER_BASE_URL}/v1/messages")
    error = anthropic.APIConnectionError(request=request)
    llm = AnthropicLLM(llm_settings(), client=FailingAnthropic(error))

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await llm.complete(
            system="system",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

    assert excinfo.value.code == "AI_UPSTREAM_UNAVAILABLE"


async def test_other_provider_errors_are_reported_as_upstream_errors():
    error = provider_status_error(anthropic.InternalServerError, 500)
    llm = AnthropicLLM(llm_settings(), client=FailingAnthropic(error))

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await llm.complete(
            system="system",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

    assert excinfo.value.code == "AI_UPSTREAM_ERROR"


class FakeMessageStream:
    def __init__(self, events, message) -> None:
        self.events = events
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def __aiter__(self):
        async def iterate():
            for event in self.events:
                yield event

        return iterate()

    async def get_final_message(self):
        return self.message


class StreamingMessages:
    def __init__(self, stream: FakeMessageStream) -> None:
        self._stream = stream
        self.kwargs: dict | None = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return self._stream


class StreamingAnthropic:
    def __init__(self, stream: FakeMessageStream) -> None:
        self.messages = StreamingMessages(stream)


class FailingMessageStream:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *_args):
        return False


class FailingStreamingMessages:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def stream(self, **_kwargs):
        return FailingMessageStream(self._error)


class FailingStreamingAnthropic:
    def __init__(self, error: Exception) -> None:
        self.messages = FailingStreamingMessages(error)


async def test_stream_complete_forwards_text_deltas_and_returns_tool_calls():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(
                type="tool_use",
                id="tool-1",
                name="get_patient_bookings",
                input={},
            ),
        ],
        stop_reason="tool_use",
    )
    client = StreamingAnthropic(
        FakeMessageStream(
            [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Let me "),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="check."),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(
                        type="thinking_delta",
                        thinking="private reasoning",
                    ),
                ),
            ],
            message,
        )
    )
    llm = AnthropicLLM(llm_settings(), client=client)
    deltas: list[str] = []

    async def collect(delta: str) -> None:
        deltas.append(delta)

    turn = await llm.stream_complete(
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        on_text=collect,
    )

    assert deltas == ["Let me ", "check."]
    assert turn.text == "Let me check."
    assert turn.tool_calls[0].id == "tool-1"
    assert turn.tool_calls[0].name == "get_patient_bookings"
    assert client.messages.kwargs["model"] == "test-model"


async def test_stream_complete_translates_provider_errors():
    error = provider_status_error(anthropic.PermissionDeniedError, 403)
    llm = AnthropicLLM(
        llm_settings(),
        client=FailingStreamingAnthropic(error),
    )

    async def collect(_delta: str) -> None:
        return None

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await llm.stream_complete(
            system="system",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            on_text=collect,
        )

    assert excinfo.value.code == "AI_AUTHENTICATION_FAILED"
