"""Tests for the FastAPI inference service, against a freshly-initialized
(untrained) checkpoint — verifies the API mechanics (routing, image
decoding, response schema, error handling), not prediction quality."""

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import src.api.main as api_main
from src.models.resnet import build_resnet


@pytest.fixture
def client(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "test_model.pt"
    model = build_resnet("resnet18", num_classes=8, pretrained=False)
    torch.save(model.state_dict(), checkpoint_path)

    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    api_main._model = None  # reset the module-level cache between tests
    return TestClient(api_main.app)


def _fake_image_bytes() -> bytes:
    im = Image.new("L", (227, 227), color=128)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_no_model_loaded_before_first_request(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is False


def test_predict_returns_valid_response_shape(client):
    resp = client.post("/predict", files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in api_main.CLASS_NAMES.values()
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 5
    assert sum(item["confidence"] for item in body["top_k"]) <= 1.0 + 1e-4


def test_predict_rejects_invalid_image(client):
    resp = client.post("/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 400


def test_predict_returns_503_when_no_checkpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(tmp_path / "does_not_exist.pt"))
    api_main._model = None
    resp = client.post("/predict", files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")})
    assert resp.status_code == 503


def test_explain_returns_heatmap(client):
    resp = client.post("/explain", files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in api_main.CLASS_NAMES.values()
    assert len(body["heatmap_png_base64"]) > 0

    import base64

    png_bytes = base64.b64decode(body["heatmap_png_base64"])
    decoded = Image.open(io.BytesIO(png_bytes))
    assert decoded.size == (227, 227)
