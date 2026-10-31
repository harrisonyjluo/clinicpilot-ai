"""Patient context endpoint used by the chat agent and the UI."""

from fastapi import APIRouter, Depends, Path

from app.api.dependencies import get_booking_service
from app.schemas.patient import PatientContextResponse
from app.services.booking_service import BookingService

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.get(
    "/{patient_id}/context",
    response_model=PatientContextResponse,
    summary="Fetch a patient's profile, allergies, bookings, and history",
)
def get_patient_context(
    patient_id: str = Path(pattern=r"^P\d{3}$"),
    service: BookingService = Depends(get_booking_service),
) -> PatientContextResponse:
    return service.get_patient_context(patient_id)
