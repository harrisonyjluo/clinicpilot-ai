"""FastAPI dependency factories that wire services to repositories."""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.booking_repository import BookingRepository
from app.repositories.booking_summary_repository import BookingSummaryRepository
from app.services.booking_service import BookingService
from app.services.booking_summary_service import BookingSummaryService
from app.services.chat_service import ChatService
from app.services.summary_service import TranscriptSummaryService


def get_booking_service(db: Session = Depends(get_db)) -> BookingService:
    return BookingService(db, BookingRepository(db))


def get_booking_summary_service(
    db: Session = Depends(get_db),
) -> BookingSummaryService:
    return BookingSummaryService(
        session=db,
        booking_repository=BookingRepository(db),
        summary_repository=BookingSummaryRepository(db),
    )


@lru_cache
def get_chat_service() -> ChatService:
    """Process-wide service so conversations and their slots survive requests."""

    return ChatService()


def get_summary_service(
    db: Session = Depends(get_db),
) -> TranscriptSummaryService:
    return TranscriptSummaryService(repository=BookingRepository(db))
