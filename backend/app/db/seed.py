"""Idempotent CSV seed for the six ClinicPilot tables."""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import PROJECT_DIR
from app.db.session import SessionLocal, init_db
from app.models import (
    Clinic,
    Doctor,
    Patient,
    PatientBooking,
    PatientTreatmentHistory,
    Treatment,
)

CSV_DIR = PROJECT_DIR / "mock-data"


class SeedError(RuntimeError):
    """Raised when the seed inputs are missing or malformed."""


def _read_csv(filename: str) -> list[dict[str, str]]:
    path = CSV_DIR / filename
    if not path.exists():
        raise SeedError(f"Missing seed file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _upsert(session: Session, model, values: dict) -> object:
    primary_key = next(iter(model.__mapper__.primary_key))
    identity = values[primary_key.key]
    instance = session.get(model, identity)
    if instance is None:
        instance = model(**values)
        session.add(instance)
    else:
        for key, value in values.items():
            setattr(instance, key, value)
    return instance


def _seed_clinics(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        _upsert(
            session,
            Clinic,
            {
                "clinic_id": row["clinic_id"],
                "clinic_name": row["clinic_name"],
                "address": row["address"],
                "phone": row["phone"],
                "operating_hours": row["operating_hours"],
                "cancellation_policy": row["cancellation_policy"],
                "emergency_instructions": row["emergency_instructions"],
            },
        )


def _seed_doctors(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        _upsert(
            session,
            Doctor,
            {
                "doctor_id": row["doctor_id"],
                "clinic_id": row["clinic_id"],
                "full_name": row["full_name"],
                "specialty": row["specialty"],
                "languages_spoken": row["languages_spoken"],
                "consultation_fee_usd": Decimal(row["consultation_fee_usd"]),
                "bio": row["bio"],
            },
        )


def _seed_treatments(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        _upsert(
            session,
            Treatment,
            {
                "treatment_id": row["treatment_id"],
                "treatment_name": row["treatment_name"],
                "category": row["category"],
                "duration_minutes": int(row["duration_minutes"]),
                "price_usd": Decimal(row["price_usd"]),
                "pre_treatment_instructions": row["pre_treatment_instructions"],
            },
        )


def _seed_patients(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        _upsert(
            session,
            Patient,
            {
                "patient_id": row["patient_id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "dob": date.fromisoformat(row["dob"]),
                "gender": row["gender"],
                "phone": row["phone"],
                "email": row["email"],
                "known_allergies": row["known_allergies"],
                "emergency_contact": row["emergency_contact"],
            },
        )


def _seed_bookings(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        values = {
            "booking_id": row["booking_id"],
            "patient_id": row["patient_id"],
            "doctor_id": row["doctor_id"],
            "treatment_id": row["treatment_id"],
            "booking_date": date.fromisoformat(row["booking_date"]),
            "booking_time": time.fromisoformat(row["booking_time"]),
            "status": row["status"],
            "notes": row["notes"] or None,
        }
        existing = session.get(PatientBooking, row["booking_id"])
        if existing is None:
            values["created_at"] = datetime.now(timezone.utc)
        _upsert(session, PatientBooking, values)


def _seed_history(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        _upsert(
            session,
            PatientTreatmentHistory,
            {
                "record_id": row["record_id"],
                "patient_id": row["patient_id"],
                "doctor_id": row["doctor_id"],
                "visit_date": date.fromisoformat(row["visit_date"]),
                "chief_complaint": row["chief_complaint"],
                "diagnosis": row["diagnosis"],
                "prescribed_medication": row["prescribed_medication"],
                "doctor_notes": row["doctor_notes"],
            },
        )


def database_counts(session: Session) -> dict[str, int]:
    """Return row counts for every seeded table."""

    models = {
        "clinics": Clinic,
        "doctors": Doctor,
        "treatments": Treatment,
        "patients": Patient,
        "patient_bookings": PatientBooking,
        "patient_treatment_history": PatientTreatmentHistory,
    }
    return {
        name: int(session.scalar(select(func.count()).select_from(model)) or 0)
        for name, model in models.items()
    }


def seed_database(session: Session) -> dict[str, int]:
    """Upsert all seeded rows and return table counts.

    The caller owns the transaction and must commit. Rows created by users are
    never deleted; only rows with seed IDs are inserted or updated.
    """

    clinics = _read_csv("clinics.csv")
    doctors = _read_csv("doctors.csv")
    treatments = _read_csv("treatments.csv")
    patients = _read_csv("patients.csv")
    bookings = _read_csv("patient_bookings.csv")
    history = _read_csv("patient_treatment_history.csv")

    _seed_clinics(session, clinics)
    _seed_doctors(session, doctors)
    _seed_treatments(session, treatments)
    _seed_patients(session, patients)
    _seed_bookings(session, bookings)
    _seed_history(session, history)
    session.flush()
    return database_counts(session)


def main() -> None:
    init_db()
    with SessionLocal() as session:
        try:
            counts = seed_database(session)
            session.commit()
        except Exception:
            session.rollback()
            raise

    print("Seed complete.")
    for table, count in counts.items():
        print(f"  {table}: {count}")


if __name__ == "__main__":
    main()
