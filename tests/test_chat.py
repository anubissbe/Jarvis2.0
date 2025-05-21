from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data == {"response": "Echo: Hello"}


def test_multiple_messages():
    first = client.post("/chat", json={"message": "Hi"})
    assert first.status_code == 200
    assert first.json() == {"response": "Echo: Hi"}

    second = client.post("/chat", json={"message": "How are you?"})
    assert second.status_code == 200
    assert second.json() == {"response": "Echo: How are you?"}
