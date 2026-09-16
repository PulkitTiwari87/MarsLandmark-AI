"""FastAPI inference service for MarsLandmark-AI.

Only two endpoints are implemented, per the project's "only what's
needed" guidance: `/health` and `/predict`. `/explain` (Grad-CAM) is
deferred to Phase 09, since it requires a trained model to be meaningful.

If no trained checkpoint is available yet, `/predict` returns HTTP 503
with an explicit message rather than fabricating a prediction — there is
currently no trained model (see docs/DATASET.md), so this is the honest
behavior today.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.data.labels import load_class_map
from src.data.transforms import build_transforms
from src.models.factory import build_model

DEFAULT_CONFIG_PATH = Path(os.environ.get("MARSLANDMARK_CONFIG", "configs/config.yaml"))
DEFAULT_CHECKPOINT_PATH = Path(os.environ.get("MARSLANDMARK_CHECKPOINT", "checkpoints/best.pt"))

_state: dict = {"model": None, "class_names": [], "image_size": 227}


def load_model(config_path: Path = DEFAULT_CONFIG_PATH, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    """Load a trained checkpoint into memory. Leaves the model as None
    (rather than raising) if no checkpoint exists yet, so the API can
    start up and report its real status via /health instead of crashing.
    """
    if not Path(checkpoint_path).exists():
        _state["model"] = None
        return
    config = yaml.safe_load(Path(config_path).read_text())
    model = build_model(config["model"]["name"], config["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    class_map = load_class_map(None)
    _state["model"] = model
    _state["image_size"] = config["data"]["image_size"]
    _state["class_names"] = [class_map[i] for i in range(config["model"]["num_classes"])]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="MarsLandmark-AI Inference API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoint is available yet. "
                "See docs/DATASET.md for acquisition/training status."
            ),
        )
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    transform = build_transforms(train=False, image_size=_state["image_size"])
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = _state["model"](tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probabilities, dim=0)
    return {
        "predicted_class": _state["class_names"][top_idx.item()],
        "confidence": top_prob.item(),
        "class_probabilities": {
            name: probabilities[i].item() for i, name in enumerate(_state["class_names"])
        },
    }
