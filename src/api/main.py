"""Phase 11: FastAPI inference service.

Only the endpoints the project actually needs (project rule §11: "Only
implement the endpoints that are actually needed"):
    GET  /health   - liveness + whether a model checkpoint is loaded
    POST /predict  - classify an uploaded image
    POST /explain  - classify + return a Grad-CAM heatmap overlay (base64 PNG)

The checkpoint path is configurable via the MODEL_CHECKPOINT_PATH env var
(default: checkpoints/exp_resnet18_finetuned.pt, matching the experiment
that was actually selected — see docs/EXPERIMENTS.md). If no checkpoint
is present, /predict and /explain return 503 rather than fabricating a
prediction from an untrained model.

Run: uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import base64
import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.data.dataset import build_transform
from src.explainability.gradcam import GradCAM
from src.models.resnet import build_resnet

CLASS_NAMES = {
    0: "other",
    1: "crater",
    2: "dark dune",
    3: "slope streak",
    4: "bright dune",
    5: "impact ejecta",
    6: "swiss cheese",
    7: "spider",
}

app = FastAPI(
    title="MarsLandmark-AI Inference API",
    description="Classifies HiRISE landmark crops. See docs/MODEL_CARD.md for intended use and known limitations (e.g. measured 0% recall on 'spider').",
)

# Frontend (frontend/index.html) runs from a local file/different origin
# than the API; this is a local research demo, not a service handling
# sensitive data, so permissive CORS is acceptable here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model: torch.nn.Module | None = None


def _checkpoint_path() -> str:
    return os.environ.get("MODEL_CHECKPOINT_PATH", "checkpoints/exp_resnet18_finetuned.pt")


def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        path = _checkpoint_path()
        if not os.path.exists(path):
            raise HTTPException(
                status_code=503,
                detail=f"No model checkpoint at {path}. Set MODEL_CHECKPOINT_PATH or place a checkpoint there.",
            )
        model = build_resnet("resnet18", num_classes=len(CLASS_NAMES), pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _model = model
    return _model


def _load_image(raw_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from None


class ClassScore(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k: list[ClassScore]


class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float
    heatmap_png_base64: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None, "checkpoint_path": _checkpoint_path()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    model = get_model()
    image = _load_image(await file.read())
    tensor = build_transform()(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top = torch.topk(probs, k=min(5, len(CLASS_NAMES)))

    return PredictionResponse(
        predicted_class=CLASS_NAMES[int(top.indices[0])],
        confidence=float(top.values[0]),
        top_k=[
            ClassScore(class_name=CLASS_NAMES[int(i)], confidence=float(p))
            for p, i in zip(top.values, top.indices)
        ],
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)) -> ExplainResponse:
    model = get_model()
    image = _load_image(await file.read())
    tensor = build_transform()(image).unsqueeze(0)

    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    try:
        heatmap, predicted_class = cam.generate(tensor)
    finally:
        cam.remove_hooks()

    with torch.no_grad():
        confidence = float(torch.softmax(model(tensor), dim=1)[0, predicted_class])

    heatmap_img = Image.fromarray((heatmap.numpy() * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    return ExplainResponse(
        predicted_class=CLASS_NAMES[predicted_class],
        confidence=confidence,
        heatmap_png_base64=encoded,
    )
