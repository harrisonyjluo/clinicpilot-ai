"""Health endpoint contract tests."""

from app.services.health_service import HealthService


def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["app"] == "ClinicPilot AI"
    assert body["version"]


def test_health_returns_503_when_database_is_down(client, monkeypatch):
    monkeypatch.setattr(HealthService, "database_is_ready", lambda self: False)

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "database": "error",
        "app": "ClinicPilot AI",
        "version": "0.1.0",
    }


def test_openapi_exposes_health(client):
    schema = client.get("/openapi.json").json()
    assert "/api/health" in schema["paths"]


def test_root_points_to_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"


def test_root_lists_the_per_booking_summary_routes(client):
    endpoints = client.get("/").json()["endpoints"]

    assert "GET /api/bookings/{booking_id}/summary" in endpoints
    assert "POST /api/bookings/{booking_id}/summary" in endpoints
