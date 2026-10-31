"""Chat endpoint contract tests."""


def test_chat_rejects_empty_message(client):
    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_chat_accepts_any_patient_id(client):
    response = client.post(
        "/api/chat",
        json={"message": "Hello", "patient_id": "P002"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_NOT_CONFIGURED"


def test_chat_allows_omitted_patient_id(client):
    response = client.post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_NOT_CONFIGURED"


def test_chat_returns_503_when_api_key_is_missing(client):
    response = client.post(
        "/api/chat",
        json={"message": "Hello", "patient_id": "P001"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "AI_NOT_CONFIGURED"
    assert "ANTHROPIC_API_KEY" in body["error"]["message"]
