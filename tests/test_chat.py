from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data == {"response": "Echo: Hello"}
