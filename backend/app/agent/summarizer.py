"""Structured transcript summarization through the Anthropic Messages API."""

from __future__ import annotations

import json
import logging

import anthropic
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from app.agent.llm import (
    _create_client,
    _extract_json_object,
    _translate_provider_error,
)
from app.core.config import Settings
from app.core.errors import ServiceUnavailableError
from app.schemas.summary import TranscriptSummary

logger = logging.getLogger("clinicpilot.summarizer")

_SUMMARY_RULES = """You are a clinical documentation assistant. Summarize the doctor-patient transcript into structured JSON.

Rules:
- Extract only information that is explicitly stated or directly implied by the transcript. Do not diagnose beyond what the clinician said.
- Preserve medication names and dosages exactly as spoken.
- Flag any prescribed medication or instruction that conflicts with the known patient allergies.
- Use "not stated" when chief_complaint, diagnosis, or follow_up is absent.
- Use an empty list when medication, doctor_instructions, or red_flags is absent.
- red_flags must include allergy conflicts, urgent symptoms, and safety concerns.
- confidence is a number from 0 to 1. Set needs_review to true when confidence is low, a red flag exists, or the transcript is too short to support the summary.
- Return only one JSON object with no prose. The summary requires clinician review and is not a substitute for clinical judgment.

JSON keys:
{"chief_complaint": "", "diagnosis": "", "medication": [], "doctor_instructions": [], "follow_up": null, "red_flags": [], "confidence": 0.0, "needs_review": true}"""


def _parse_summary(text: str) -> TranscriptSummary:
    candidates = [text.strip()]
    extracted = _extract_json_object(text)
    if extracted is not None and extracted not in candidates:
        candidates.append(extracted)

    error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            error = exc
            continue
        if not isinstance(payload, dict):
            error = ValueError("transcript summary must be a JSON object")
            continue
        try:
            return TranscriptSummary.model_validate(payload)
        except ValidationError as exc:
            error = exc

    raise ServiceUnavailableError(
        "The AI provider returned an invalid transcript summary.",
        code="AI_RESPONSE_INVALID",
    ) from error


class AnthropicTranscriptSummarizer:
    """Extract structured medical JSON without persisting the transcript."""

    def __init__(
        self,
        settings: Settings,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or _create_client(settings)

    @staticmethod
    def _system_prompt(allergies: list[str]) -> str:
        return "\n\n".join(
            (
                _SUMMARY_RULES,
                "Known patient allergies:\n"
                + json.dumps(allergies, ensure_ascii=False),
            )
        )

    async def summarize(
        self,
        transcript: str,
        *,
        allergies: list[str],
    ) -> TranscriptSummary:
        try:
            response = await self._client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=4096,
                system=self._system_prompt(allergies),
                messages=[{"role": "user", "content": transcript}],
            )
        except anthropic.APIError as exc:
            raise _translate_provider_error(exc) from exc

        text = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        try:
            return _parse_summary(text)
        except ServiceUnavailableError:
            logger.warning(
                "summary_response_invalid stop_reason=%s chars=%s",
                getattr(response, "stop_reason", None),
                len(text),
            )
            raise
