"""API tests use an untrained model checkpoint saved in a temp dir — they
verify the endpoint mechanics (status codes, response shape), not any
real prediction accuracy. There is no trained model yet (see
docs/DATASET.md)."""

import io

import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main
from src.models.factory import build_model


def _make_test_image_bytes() -> bytes:
    img = Image.new("L", (32, 32), color=100)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_without_model():
    main._state["model"] = None
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_without_model_returns_503():
    main._state["model"] = None
    client = TestClient(main.app)
    response = client.post(
        "/predict", files={"file": ("a.jpg", _make_test_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 503


def test_predict_with_loaded_model_returns_valid_shape(tmp_path):
    num_classes = 8
    config = {
        "model": {"name": "simple_cnn", "num_classes": num_classes, "pretrained": False},
        "data": {"image_size": 32},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    model = build_model("simple_cnn", num_classes=num_classes, in_channels=3)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    main.load_model(config_path=config_path, checkpoint_path=checkpoint_path)
    client = TestClient(main.app)

    response = client.post(
        "/predict", files={"file": ("a.jpg", _make_test_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in main._state["class_names"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["class_probabilities"]) == num_classes
    assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-4

    main._state["model"] = None  # reset global state for other tests
