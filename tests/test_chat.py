from fastapi.testclient import TestClient

import app.main
from app.main import app

client = TestClient(app)


def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data == {"response": "Echo: Hello"}


def test_chat_error_handling(monkeypatch):
    async def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app.main.chain, "apredict", raise_error)
    error_client = TestClient(app)
    resp = error_client.post("/chat", json={"message": "Hi"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "boom"}
