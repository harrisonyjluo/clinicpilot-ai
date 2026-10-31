"""Treatment table."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.booking import PatientBooking


class Treatment(Base):
    __tablename__ = "treatments"

    treatment_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    treatment_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    duration_minutes: Mapped[int] = mapped_column(nullable=False)
    price_usd: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    pre_treatment_instructions: Mapped[str] = mapped_column(Text, nullable=False)

    bookings: Mapped[list["PatientBooking"]] = relationship(back_populates="treatment")
