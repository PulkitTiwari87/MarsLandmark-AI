"""Training loop functions, reusable from the CLI or from a notebook
(e.g. the Colab training notebook, notebooks/colab_train.ipynb).

Usage (CLI):
    python -m src.training.train --config configs/config.yaml --model simple_cnn
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import HiRISELandmarkDataset
from src.models.baseline import SimpleCNN
from src.models.resnet import build_resnet
from src.training.imbalance import compute_class_weights
from src.training.metrics import compute_metrics

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

EXPERIMENT_CSV_FIELDS = [
    "experiment_id",
    "date",
    "git_commit",
    "dataset_version",
    "dataset_checksum",
    "model",
    "image_size",
    "batch_size",
    "optimizer",
    "learning_rate",
    "scheduler",
    "epochs",
    "seed",
    "augmentation",
    "loss_function",
    "hardware",
    "training_time_min",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "status",
    "notes",
]


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names: dict[int, str] = CLASS_NAMES):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, class_names)
    return total_loss / n, metrics


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs: int,
    class_names: dict[int, str] = CLASS_NAMES,
    early_stopping_patience: int | None = None,
) -> dict:
    """Trains for up to `epochs`, tracking the best val macro_f1 checkpoint
    in memory (state_dict), with optional early stopping. Returns a history
    dict — never claims a result it did not measure."""
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, class_names)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            history["stopped_early_at_epoch"] = epoch
            break

    return {"history": history, "best_val_macro_f1": best_macro_f1, "best_state_dict": best_state}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def log_experiment(csv_path: Path, **fields) -> None:
    """Append one row to the experiment registry. Missing fields are
    written as empty (never fabricated) — the caller must not pass
    invented values for anything not actually measured/configured."""
    row = {k: fields.get(k, "") for k in EXPERIMENT_CSV_FIELDS}
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def build_model(name: str, num_classes: int = 8):
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name in ("resnet18", "resnet50"):
        return build_resnet(name, num_classes=num_classes, pretrained=True)
    raise ValueError(f"Unknown model name: {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", type=str, default=None, help="Overrides model.name in config")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/split_manifest.csv")
    )
    parser.add_argument("--epochs", type=int, default=None, help="Overrides training.epochs in config")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])

    model_name = args.model or config["model"]["name"]
    epochs = args.epochs or config["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HiRISELandmarkDataset(args.data_dir, args.manifest, "train")
    val_ds = HiRISELandmarkDataset(args.data_dir, args.manifest, "val")
    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config["data"].get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config["data"].get("num_workers", 0))

    model = build_model(model_name, num_classes=config["model"]["num_classes"]).to(device)
    if config["training"].get("class_weighted_loss"):
        train_label_counts: dict[int, int] = {}
        for _, label in train_ds.samples:
            train_label_counts[label] = train_label_counts.get(label, 0) + 1
        class_weights = compute_class_weights(train_label_counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    start = time.time()
    result = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=epochs,
        early_stopping_patience=config["training"].get("early_stopping_patience"),
    )
    training_time_min = (time.time() - start) / 60

    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log_experiment(
        config["logging"]["experiment_registry"],
        experiment_id=args.experiment_id,
        date=datetime.date.today().isoformat(),
        git_commit=_git_commit(),
        dataset_version="hirise-map-proj-v3",
        model=model_name,
        image_size=config["data"]["image_size"],
        batch_size=batch_size,
        optimizer="adamw",
        learning_rate=config["training"]["learning_rate"],
        scheduler=config["training"].get("scheduler", ""),
        epochs=len(result["history"]["train_loss"]),
        seed=config["seed"],
        augmentation="none (dataset pre-augmented 6x, see docs/EDA.md)",
        loss_function="cross_entropy_weighted" if config["training"].get("class_weighted_loss") else "cross_entropy",
        hardware=hardware,
        training_time_min=round(training_time_min, 2),
        val_accuracy=round(result["history"]["val_accuracy"][-1], 4),
        val_macro_f1=round(result["best_val_macro_f1"], 4),
        status="COMPLETED",
        notes=args.notes,
    )

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["best_state_dict"], checkpoint_dir / f"{args.experiment_id}.pt")
    print(json.dumps({"best_val_macro_f1": result["best_val_macro_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
