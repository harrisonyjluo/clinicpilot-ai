"""Cross-cutting API contract tests: OpenAPI and error middleware."""

from fastapi.testclient import TestClient

from app.api.dependencies import get_booking_service
from app.main import app


def test_api_routes_are_in_openapi(client):
    schema = client.get("/openapi.json").json()

    assert "/api/bookings" in schema["paths"]
    assert "/api/chat" in schema["paths"]
    assert "/api/chat/stream" in schema["paths"]
    assert "/api/patients/{patient_id}/context" in schema["paths"]
    assert "/api/summarize" in schema["paths"]


def test_health_response_has_request_id_header(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")


def test_request_id_header_is_echoed(client):
    response = client.get("/api/health", headers={"X-Request-ID": "request-1"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-1"


def test_unknown_route_uses_structured_error(client):
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_unexpected_errors_keep_cors_headers_for_the_frontend(seeded_database):
    def explode():
        raise RuntimeError("boom")

    app.dependency_overrides[get_booking_service] = explode
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.get(
                "/api/bookings?patient_id=P001",
                headers={"Origin": "http://localhost:5173"},
            )
    finally:
        app.dependency_overrides.pop(get_booking_service, None)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert (
        response.headers.get("access-control-allow-origin")
        == "http://localhost:5173"
    )
    assert response.headers.get("X-Request-ID")
