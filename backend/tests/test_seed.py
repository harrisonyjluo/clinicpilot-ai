"""Seed contract tests for the six CSV-backed tables."""

from decimal import Decimal

from sqlalchemy import select

from app.db.seed import database_counts, seed_database
from app.models import (
    Clinic,
    Doctor,
    Patient,
    PatientBooking,
    PatientTreatmentHistory,
    Treatment,
)

# The dataset shape is the demo contract; the seed module must not own it.
EXPECTED_SEED_COUNTS = {
    "clinics": 2,
    "doctors": 4,
    "treatments": 5,
    "patients": 3,
    "patient_bookings": 4,
    "patient_treatment_history": 2,
}


def test_seed_counts_match_the_dataset(db_session):
    assert database_counts(db_session) == EXPECTED_SEED_COUNTS


def test_seed_is_idempotent(db_session):
    before = database_counts(db_session)
    seed_database(db_session)
    db_session.commit()
    assert database_counts(db_session) == before == EXPECTED_SEED_COUNTS


def test_p001_profile_and_upcoming_bookings(db_session):
    patient = db_session.get(Patient, "P001")
    assert patient is not None
    assert patient.first_name == "John"
    assert patient.last_name == "Doe"
    assert patient.known_allergies == "Penicillin, Peanuts"

    bookings = db_session.scalars(
        select(PatientBooking)
        .where(PatientBooking.patient_id == "P001")
        .order_by(PatientBooking.booking_date, PatientBooking.booking_time)
    ).all()
    assert [booking.booking_id for booking in bookings] == ["B1001", "B1004"]
    assert bookings[0].doctor.full_name == "Dr. Sarah Jenkins"
    assert bookings[0].treatment.treatment_name == "General Health Checkup"


def test_all_seeded_entities_are_queryable(db_session):
    assert db_session.get(Clinic, "C001") is not None
    assert db_session.get(Clinic, "C002") is not None

    doctors = db_session.scalars(select(Doctor).order_by(Doctor.doctor_id)).all()
    assert [doctor.doctor_id for doctor in doctors] == ["D101", "D102", "D103", "D104"]
    assert doctors[2].consultation_fee_usd == Decimal("220.00")

    treatments = db_session.scalars(select(Treatment).order_by(Treatment.treatment_id)).all()
    assert [treatment.treatment_id for treatment in treatments] == [
        "T001",
        "T002",
        "T003",
        "T004",
        "T005",
    ]

    history = db_session.get(PatientTreatmentHistory, "H9002")
    assert history is not None
    assert history.patient_id == "P001"
    assert history.diagnosis == "Healthy / Normal Range"
