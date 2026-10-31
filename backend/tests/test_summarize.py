"""Transcript summarizer: schema, service rules, prompt, and HTTP contract."""

import json
from types import SimpleNamespace

import pytest

from app.agent.summarizer import AnthropicTranscriptSummarizer
from app.api.dependencies import get_summary_service
from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.main import app
from app.schemas.summary import TranscriptSummary, TranscriptSummaryRequest
from app.services.summary_service import TranscriptSummaryService

LONG_TRANSCRIPT = (
    "Doctor: What brings you in today? "
    "Patient: I have had a sore throat and fever for three days. "
    "Doctor: I will prescribe amoxicillin for the infection. "
    "Patient: I am allergic to penicillin. "
    "Doctor: Thank you for telling me; I will choose a different antibiotic. "
    "Please rest, drink fluids, and return if your fever worsens."
)


def summary(**overrides) -> TranscriptSummary:
    values = {
        "chief_complaint": "sore throat and fever",
        "diagnosis": "bacterial throat infection",
        "medication": ["amoxicillin 500 mg"],
        "doctor_instructions": ["rest", "drink fluids", "return if worse"],
        "follow_up": "return in three days if symptoms persist",
        "red_flags": [],
        "confidence": 0.92,
        "needs_review": False,
    }
    values.update(overrides)
    return TranscriptSummary(**values)


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "anthropic_api_key": "test-key",
        "default_patient_id": "P001",
    }
    values.update(overrides)
    return Settings(**values)


class FakeRepository:
    def __init__(self, *, known_allergies="Penicillin, Peanuts", exists=True):
        self.known_allergies = known_allergies
        self.exists = exists
        self.calls: list[str] = []

    def get_patient_context(self, patient_id: str):
        self.calls.append(patient_id)
        if not self.exists:
            return None
        return SimpleNamespace(known_allergies=self.known_allergies)


class FakeSummarizer:
    def __init__(self, result: TranscriptSummary) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def summarize(self, transcript: str, *, allergies: list[str]):
        self.calls.append({"transcript": transcript, "allergies": allergies})
        return self.result


class StubSummaryService:
    async def summarize(self, _request):
        return summary()


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


@pytest.mark.asyncio
async def test_summarize_service_returns_all_four_required_fields():
    repository = FakeRepository()
    summarizer = FakeSummarizer(summary())
    service = TranscriptSummaryService(
        settings=settings(),
        repository=repository,
        summarizer=summarizer,
    )

    result = await service.summarize(
        TranscriptSummaryRequest(transcript=LONG_TRANSCRIPT)
    )

    assert result.chief_complaint == "sore throat and fever"
    assert result.diagnosis == "bacterial throat infection"
    assert result.medication == ["amoxicillin 500 mg"]
    assert result.doctor_instructions == [
        "rest",
        "drink fluids",
        "return if worse",
    ]
    assert result.needs_review is False
    assert summarizer.calls[0]["allergies"] == ["Penicillin", "Peanuts"]
    assert repository.calls == ["P001"]


@pytest.mark.asyncio
async def test_summarize_service_marks_red_flags_for_clinician_review():
    service = TranscriptSummaryService(
        settings=settings(),
        repository=FakeRepository(),
        summarizer=FakeSummarizer(
            summary(red_flags=["penicillin allergy conflict"], needs_review=False)
        ),
    )

    result = await service.summarize(
        TranscriptSummaryRequest(transcript=LONG_TRANSCRIPT)
    )

    assert result.red_flags == ["penicillin allergy conflict"]
    assert result.needs_review is True


@pytest.mark.asyncio
async def test_summarize_service_marks_low_confidence_or_short_input_for_review():
    service = TranscriptSummaryService(
        settings=settings(),
        repository=FakeRepository(),
        summarizer=FakeSummarizer(summary(confidence=0.4, needs_review=False)),
    )

    low_confidence = await service.summarize(
        TranscriptSummaryRequest(transcript=LONG_TRANSCRIPT)
    )
    short = await service.summarize(
        TranscriptSummaryRequest(transcript="Doctor: hi. Patient: hello.")
    )

    assert low_confidence.needs_review is True
    assert short.needs_review is True


@pytest.mark.asyncio
async def test_summarize_service_requires_a_configured_patient():
    service = TranscriptSummaryService(
        settings=settings(default_patient_id=None),
        repository=FakeRepository(),
        summarizer=FakeSummarizer(summary()),
    )

    from app.core.errors import BadRequestError

    with pytest.raises(BadRequestError) as excinfo:
        await service.summarize(
            TranscriptSummaryRequest(transcript=LONG_TRANSCRIPT)
        )

    assert excinfo.value.code == "PATIENT_ID_REQUIRED"


@pytest.mark.asyncio
async def test_summarize_service_returns_503_without_an_api_key():
    service = TranscriptSummaryService(
        settings=settings(anthropic_api_key=None),
        repository=FakeRepository(),
        summarizer=FakeSummarizer(summary()),
    )

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await service.summarize(
            TranscriptSummaryRequest(transcript=LONG_TRANSCRIPT)
        )

    assert excinfo.value.code == "AI_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_summarizer_prompt_includes_allergies_and_json_contract():
    payload = {
        "chief_complaint": "sore throat",
        "diagnosis": "infection",
        "medication": ["amoxicillin 500 mg"],
        "doctor_instructions": ["rest"],
        "follow_up": "three days",
        "red_flags": [],
        "confidence": 0.88,
        "needs_review": False,
    }
    client = TextAnthropic(json.dumps(payload))
    summarizer = AnthropicTranscriptSummarizer(settings(), client=client)

    result = await summarizer.summarize(
        LONG_TRANSCRIPT,
        allergies=["Penicillin", "Peanuts"],
    )

    assert result.chief_complaint == "sore throat"
    assert result.medication == ["amoxicillin 500 mg"]
    system = client.messages.kwargs["system"]
    assert "Penicillin" in system
    assert "Peanuts" in system
    assert "chief_complaint" in system
    assert "doctor_instructions" in system
    assert "not stated" in system.lower()
    assert "clinician review" in system.lower()
    assert client.messages.kwargs["messages"] == [
        {"role": "user", "content": LONG_TRANSCRIPT}
    ]


@pytest.mark.asyncio
async def test_summarizer_rejects_unusable_model_output():
    summarizer = AnthropicTranscriptSummarizer(
        settings(),
        client=TextAnthropic("I could not summarize that."),
    )

    with pytest.raises(ServiceUnavailableError) as excinfo:
        await summarizer.summarize(
            LONG_TRANSCRIPT,
            allergies=["Penicillin"],
        )

    assert excinfo.value.code == "AI_RESPONSE_INVALID"


def test_summarize_route_returns_structured_json(client):
    app.dependency_overrides[get_summary_service] = lambda: StubSummaryService()
    try:
        response = client.post(
            "/api/summarize",
            json={"transcript": LONG_TRANSCRIPT},
        )
    finally:
        app.dependency_overrides.pop(get_summary_service, None)

    assert response.status_code == 200
    body = response.json()
    for field in (
        "chief_complaint",
        "diagnosis",
        "medication",
        "doctor_instructions",
    ):
        assert field in body
    assert body["medication"] == ["amoxicillin 500 mg"]


def test_summarize_route_rejects_empty_and_oversized_transcripts(client):
    empty = client.post("/api/summarize", json={"transcript": "   "})
    oversized = client.post(
        "/api/summarize",
        json={"transcript": "x" * 20_001},
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422


def test_summarize_route_returns_503_without_llm_configuration(client):
    response = client.post(
        "/api/summarize",
        json={"transcript": LONG_TRANSCRIPT},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_NOT_CONFIGURED"
