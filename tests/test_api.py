import asyncio
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.app.main import BodyLimitMiddleware, MAX_FILE, create_app

TOKEN = "test-only-key-012345678901234567890123456789"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    def factory():
        return lambda image: [{"class_id": 0, "class_name": "item", "confidence": 0.9, "xyxy": [0, 0, 8, 8]}]
    with TestClient(create_app(factory)) as test_client:
        yield test_client


def image_bytes(fmt="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16)).save(stream, format=fmt)
    return stream.getvalue()


def test_health_and_readiness(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"status": "ready"}


@pytest.mark.parametrize("key", [None, "wrong"])
def test_auth_required(client, key):
    response = client.post("/predict", headers={"X-API-Key": key} if key else {},
                           files={"file": ("a.png", image_bytes(), "image/png")})
    assert response.status_code == 401


def test_valid_prediction(client):
    response = client.post("/predict", headers={"X-API-Key": TOKEN},
                           files={"file": ("a.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["detections"][0]["class_name"] == "item"


@pytest.mark.parametrize("content,mime,status", [(b"not an image", "image/png", 400), (b"bad", "text/plain", 415), (b"", "image/jpeg", 400)])
def test_reject_invalid_uploads(client, content, mime, status):
    response = client.post("/predict", headers={"X-API-Key": TOKEN}, files={"file": ("x", content, mime)})
    assert response.status_code == status


def test_reject_mime_spoof(client):
    response = client.post("/predict", headers={"X-API-Key": TOKEN},
                           files={"file": ("fake.png", image_bytes("GIF"), "image/png")})
    assert response.status_code == 415


def test_reject_large_length_before_parsing(client):
    response = client.post("/predict", content=b"x", headers={"Content-Length": str(MAX_FILE * 2)})
    assert response.status_code == 413


def test_chunked_body_limit_without_content_length():
    calls, messages = [], []
    async def downstream(scope, receive, send):
        calls.append("called")
    chunks = iter([{"type": "http.request", "body": b"12345", "more_body": True},
                   {"type": "http.request", "body": b"67890", "more_body": False}])
    async def receive():
        return next(chunks)
    async def send(message):
        messages.append(message)
    asyncio.run(BodyLimitMiddleware(downstream, max_bytes=8)(
        {"type": "http", "method": "POST", "headers": []}, receive, send))
    assert not calls
    assert messages[0]["status"] == 413


def test_fail_closed_on_missing_token(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="API_TOKEN"):
        with TestClient(create_app(lambda: lambda image: [])):
            pass


def test_fail_closed_on_model_load_error(monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    def broken():
        raise ValueError("Model checksum mismatch")
    with pytest.raises(ValueError, match="checksum"):
        with TestClient(create_app(broken)):
            pass
