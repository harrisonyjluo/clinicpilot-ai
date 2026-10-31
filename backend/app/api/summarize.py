"""Optional transcript summarizer endpoint; independent of chat and bookings."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_summary_service
from app.schemas.summary import TranscriptSummary, TranscriptSummaryRequest
from app.services.summary_service import TranscriptSummaryService

router = APIRouter(prefix="/api/summarize", tags=["summary"])


@router.post(
    "",
    response_model=TranscriptSummary,
    summary="Summarize a doctor-patient transcript into structured medical JSON",
    responses={
        400: {"description": "No patient ID is configured"},
        404: {"description": "The patient does not exist"},
        422: {"description": "Empty or oversized transcript"},
        503: {"description": "The LLM is not configured or unavailable"},
    },
)
async def summarize(
    payload: TranscriptSummaryRequest,
    service: TranscriptSummaryService = Depends(get_summary_service),
) -> TranscriptSummary:
    return await service.summarize(payload)
