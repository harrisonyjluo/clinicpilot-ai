"""Patient context API contract tests."""


def test_get_patient_context_returns_profile_allergies_and_history(client):
    response = client.get("/api/patients/P001/context")

    assert response.status_code == 200
    body = response.json()
    assert body["patient"]["patient_id"] == "P001"
    assert body["patient"]["first_name"] == "John"
    assert body["patient"]["last_name"] == "Doe"
    assert body["allergies"] == ["Penicillin", "Peanuts"]
    assert [booking["booking_id"] for booking in body["bookings"]] == [
        "B1001",
        "B1004",
    ]
    assert [record["record_id"] for record in body["history"]] == ["H9002"]
    assert body["history"][0]["doctor_name"] == "Dr. Sarah Jenkins"


def test_get_patient_context_returns_any_seeded_patient(client):
    response = client.get("/api/patients/P002/context")

    assert response.status_code == 200
    body = response.json()
    assert body["patient"]["patient_id"] == "P002"
    assert body["patient"]["first_name"] == "Alice"
    assert body["allergies"] == ["None"]
    assert [booking["booking_id"] for booking in body["bookings"]] == ["B1002"]
    assert body["history"] == []


def test_get_patient_context_returns_404_for_unknown_patient(client):
    response = client.get("/api/patients/P999/context")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PATIENT_NOT_FOUND"


def test_get_patient_context_validates_patient_id_format(client):
    response = client.get("/api/patients/P99/context")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
