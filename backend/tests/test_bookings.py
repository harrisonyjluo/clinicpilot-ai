"""Bookings API contract tests."""

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models import PatientBooking

NEW_BOOKING = {
    "patient_id": "P001",
    "doctor_id": "D103",
    "treatment_id": "T003",
    "booking_date": "2026-12-01",
    "booking_time": "10:00:00",
    "notes": "Skin rash on left arm",
}


def _booking_count(patient_id: str = "P001") -> int:
    with SessionLocal() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(PatientBooking)
                .where(PatientBooking.patient_id == patient_id)
            )
            or 0
        )


def _get_bookings(client, patient_id: str = "P001"):
    return client.get("/api/bookings", params={"patient_id": patient_id})


def test_get_bookings_returns_seeded_p001_bookings(client):
    response = _get_bookings(client)

    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == "P001"
    assert body["count"] == 2
    assert [booking["booking_id"] for booking in body["bookings"]] == ["B1001", "B1004"]

    first = body["bookings"][0]
    assert first["doctor_name"] == "Dr. Sarah Jenkins"
    assert first["treatment_name"] == "General Health Checkup"
    assert first["clinic_name"] == "Metro Care Health Clinic"
    assert first["booking_date"] == "2026-11-05"
    assert first["booking_time"] == "09:00:00"


def test_get_bookings_requires_patient_id(client):
    response = client.get("/api/bookings")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_get_bookings_returns_any_seeded_patient(client):
    response = _get_bookings(client, "P002")

    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == "P002"
    assert body["count"] == 1
    assert [booking["booking_id"] for booking in body["bookings"]] == ["B1002"]
    assert body["bookings"][0]["doctor_name"] == "Dr. Elena Rostova"


def test_create_booking_persists_and_appears_in_list(client):
    response = client.post("/api/bookings", json=NEW_BOOKING)

    assert response.status_code == 201
    created = response.json()
    assert created["booking_id"] == "B1005"
    assert created["patient_id"] == "P001"
    assert created["doctor_id"] == "D103"
    assert created["treatment_id"] == "T003"
    assert created["doctor_name"] == "Dr. Elena Rostova"
    assert created["treatment_name"] == "Dermatology Consultation"
    assert created["clinic_name"] == "Sunset Wellness Center"
    assert created["booking_date"] == "2026-12-01"
    assert created["booking_time"] == "10:00:00"
    assert created["status"] == "Confirmed"

    listing = _get_bookings(client).json()
    assert listing["count"] == 3
    assert [booking["booking_id"] for booking in listing["bookings"]] == [
        "B1001",
        "B1004",
        "B1005",
    ]
    assert _booking_count() == 3


def test_create_booking_is_idempotent_with_client_request_id(client):
    payload = {**NEW_BOOKING, "client_request_id": "idempotency-1"}

    first = client.post("/api/bookings", json=payload)
    second = client.post("/api/bookings", json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["booking_id"] == "B1005"
    assert second.json()["booking_id"] == "B1005"
    assert _booking_count() == 3


def test_create_booking_rejects_unknown_doctor_without_writing(client):
    before = _booking_count()
    response = client.post(
        "/api/bookings",
        json={**NEW_BOOKING, "doctor_id": "D999"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCTOR_NOT_FOUND"
    assert _booking_count() == before


def test_create_booking_rejects_unknown_treatment_without_writing(client):
    before = _booking_count()
    response = client.post(
        "/api/bookings",
        json={**NEW_BOOKING, "treatment_id": "T999"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TREATMENT_NOT_FOUND"
    assert _booking_count() == before


def test_create_booking_allows_any_existing_patient(client):
    response = client.post(
        "/api/bookings",
        json={**NEW_BOOKING, "patient_id": "P002"},
    )

    assert response.status_code == 201
    created = response.json()
    assert created["booking_id"] == "B1005"
    assert created["patient_id"] == "P002"
    assert _booking_count("P002") == 2

    listing = _get_bookings(client, "P002").json()
    assert listing["count"] == 2
    assert [booking["booking_id"] for booking in listing["bookings"]] == [
        "B1002",
        "B1005",
    ]


def test_create_booking_rejects_unknown_patient_without_writing(client):
    before = _booking_count("P001")
    response = client.post(
        "/api/bookings",
        json={**NEW_BOOKING, "patient_id": "P999"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PATIENT_NOT_FOUND"
    assert _booking_count("P001") == before


def test_create_booking_requires_patient_id(client):
    payload = {
        key: value for key, value in NEW_BOOKING.items() if key != "patient_id"
    }
    response = client.post("/api/bookings", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_create_booking_returns_structured_validation_error(client):
    before = _booking_count()
    response = client.post("/api/bookings", json={"patient_id": "P001"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["details"]
    assert _booking_count() == before
