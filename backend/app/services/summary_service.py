"""Transcript summary use case: patient context, LLM call, and review rules."""

from __future__ import annotations

from app.agent.summarizer import AnthropicTranscriptSummarizer
from app.core.config import Settings, get_settings
from app.core.errors import BadRequestError, NotFoundError, ServiceUnavailableError
from app.repositories.booking_repository import BookingRepository
from app.schemas.summary import TranscriptSummary, TranscriptSummaryRequest

MIN_REVIEWABLE_TRANSCRIPT_CHARS = 40
LOW_CONFIDENCE_THRESHOLD = 0.7


def apply_review_rules(
    summary: TranscriptSummary,
    transcript: str,
) -> TranscriptSummary:
    """Force clinician review for flags, low confidence, or thin input."""

    needs_review = (
        summary.needs_review
        or bool(summary.red_flags)
        or summary.confidence < LOW_CONFIDENCE_THRESHOLD
        or len(transcript.strip()) < MIN_REVIEWABLE_TRANSCRIPT_CHARS
    )
    return summary.model_copy(update={"needs_review": needs_review})


class TranscriptSummaryService:
    """Summarizes a transcript for a patient; it never persists either input."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        repository: BookingRepository | None = None,
        summarizer: AnthropicTranscriptSummarizer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repository = repository
        self._summarizer = summarizer

    async def summarize(
        self,
        request: TranscriptSummaryRequest,
    ) -> TranscriptSummary:
        if not self._settings.anthropic_api_key:
            raise ServiceUnavailableError(
                "ANTHROPIC_API_KEY is not set. Configure it to enable summaries.",
                code="AI_NOT_CONFIGURED",
            )

        patient_id = request.patient_id or self._settings.default_patient_id
        if patient_id is None:
            raise BadRequestError(
                "patient_id is required when DEFAULT_PATIENT_ID is not configured.",
                code="PATIENT_ID_REQUIRED",
            )
        if self._repository is None:
            raise ServiceUnavailableError(
                "The patient context repository is unavailable.",
                code="PATIENT_CONTEXT_UNAVAILABLE",
            )

        patient = self._repository.get_patient_context(patient_id)
        if patient is None:
            raise NotFoundError(
                f"Patient {patient_id} does not exist.",
                code="PATIENT_NOT_FOUND",
            )
        allergies = [
            allergy.strip()
            for allergy in (patient.known_allergies or "").split(",")
            if allergy.strip()
        ]

        summarizer = self._summarizer or AnthropicTranscriptSummarizer(
            self._settings
        )
        summary = await summarizer.summarize(
            request.transcript,
            allergies=allergies,
        )
        return apply_review_rules(summary, request.transcript)
