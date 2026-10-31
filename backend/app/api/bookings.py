"""Booking endpoints used by the UI and by the AI agent's tool calls."""

from fastapi import APIRouter, Depends, Query, Response

from app.api.dependencies import (
    get_booking_service,
    get_booking_summary_service,
)
from app.schemas.booking import BookingCreate, BookingListResponse, BookingRead
from app.schemas.summary import BookingSummaryRead, BookingSummaryRequest
from app.services.booking_service import BookingService
from app.services.booking_summary_service import BookingSummaryService

router = APIRouter(prefix="/api/bookings", tags=["bookings"])


@router.get(
    "",
    response_model=BookingListResponse,
    summary="List bookings for a patient",
)
def list_bookings(
    patient_id: str = Query(..., pattern=r"^P\d{3}$"),
    service: BookingService = Depends(get_booking_service),
) -> BookingListResponse:
    return service.list_bookings(patient_id)


@router.post(
    "",
    response_model=BookingRead,
    status_code=201,
    summary="Create a booking for a patient",
    responses={
        200: {"description": "Idempotent replay of an existing booking"},
        404: {"description": "Patient, doctor, or treatment does not exist"},
        422: {"description": "Payload validation failed"},
    },
)
def create_booking(
    payload: BookingCreate,
    response: Response,
    service: BookingService = Depends(get_booking_service),
) -> BookingRead:
    booking, created = service.create_booking(payload)
    if not created:
        response.status_code = 200
    return booking


@router.get(
    "/{booking_id}/summary",
    response_model=BookingSummaryRead,
    summary="Read the treatment-history summary linked to a booking",
    responses={
        404: {"description": "Booking or summary does not exist"},
    },
)
def get_booking_summary(
    booking_id: str,
    service: BookingSummaryService = Depends(get_booking_summary_service),
) -> BookingSummaryRead:
    return service.get(booking_id)


@router.post(
    "/{booking_id}/summary",
    response_model=BookingSummaryRead,
    summary="Generate and persist a treatment-history summary for a booking",
    responses={
        404: {"description": "Booking does not exist"},
        422: {"description": "Transcript validation failed"},
        503: {"description": "The LLM is not configured or unavailable"},
    },
)
async def create_booking_summary(
    booking_id: str,
    payload: BookingSummaryRequest,
    service: BookingSummaryService = Depends(get_booking_summary_service),
) -> BookingSummaryRead:
    return await service.generate(booking_id, payload)
