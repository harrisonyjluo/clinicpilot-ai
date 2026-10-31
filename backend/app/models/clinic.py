"""Clinic table."""

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.doctor import Doctor


class Clinic(Base):
    __tablename__ = "clinics"

    clinic_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    clinic_name: Mapped[str] = mapped_column(String(200), nullable=False)
    address: Mapped[str] = mapped_column(Text, nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    operating_hours: Mapped[str] = mapped_column(Text, nullable=False)
    cancellation_policy: Mapped[str] = mapped_column(Text, nullable=False)
    emergency_instructions: Mapped[str] = mapped_column(Text, nullable=False)

    doctors: Mapped[list["Doctor"]] = relationship(back_populates="clinic")
