"""ORM models. Importing this package registers all tables on Base.metadata."""

from app.models.base import Base
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.treatment import Treatment
from app.models.patient import Patient
from app.models.booking import PatientBooking
from app.models.history import PatientTreatmentHistory

__all__ = [
    "Base",
    "Clinic",
    "Doctor",
    "Treatment",
    "Patient",
    "PatientBooking",
    "PatientTreatmentHistory",
]
