from fastapi.testclient import TestClient

import app.main as jarvis_main
from app.main import app as fastapi_app

client = TestClient(fastapi_app)


def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data == {"response": "Echo: Hello"}


def test_chat_error_handling(monkeypatch):
    async def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(jarvis_main.chain, "apredict", raise_error)
    error_client = TestClient(fastapi_app)
    resp = error_client.post("/chat", json={"message": "Hi"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "boom"}
