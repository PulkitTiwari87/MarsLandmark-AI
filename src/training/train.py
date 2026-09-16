"""Training entry point.

Trains on TRAIN, selects/early-stops on VALIDATION. Never touches TEST —
final test evaluation is a separate, single-use script
(src/evaluation/evaluate.py), per the project's test-set integrity rule.

Usage:
    python -m src.training.train --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import HiRISELandmarkDataset
from src.data.transforms import build_transforms
from src.models.factory import build_model
from src.training.metrics import compute_classification_metrics
from src.training.seed import set_seed


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_preds = [], []
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), all_labels, all_preds


def train(config: dict) -> dict:
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    image_size = data_cfg["image_size"]

    train_ds = HiRISELandmarkDataset(
        image_dir=raw_dir, label_file=raw_dir / "train-labels.txt",
        transform=build_transforms(train=True, image_size=image_size),
    )
    val_ds = HiRISELandmarkDataset(
        image_dir=raw_dir, label_file=raw_dir / "val-labels.txt",
        transform=build_transforms(train=False, image_size=image_size),
    )
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])

    model_cfg = config["model"]
    model = build_model(model_cfg["name"], model_cfg["num_classes"], model_cfg["pretrained"]).to(device)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_val_metrics: dict | None = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(train_cfg["epochs"]):
        train_loss, train_labels, train_preds = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_labels, val_preds = run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_metrics = compute_classification_metrics(val_labels, val_preds, train_ds.class_names)
        scheduler.step()

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time_min = (time.time() - start_time) / 60
    return {
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_macro_f1": best_val_metrics["macro_f1"],
        "training_time_min": training_time_min,
        "epochs_run": epoch + 1,
    }


def log_experiment(config: dict, result: dict) -> None:
    registry_path = Path(config["logging"]["experiment_registry"])
    with open(registry_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"exp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            datetime.now(timezone.utc).date().isoformat(),
            get_git_commit(),
            config["data"]["dataset_name"],
            "UNKNOWN",  # dataset checksum — fill in once verified, see docs/DATASET.md
            config["model"]["name"],
            config["data"]["image_size"],
            config["data"]["batch_size"],
            config["training"]["optimizer"],
            config["training"]["learning_rate"],
            config["training"]["scheduler"],
            result["epochs_run"],
            config["seed"],
            "flip+90rot",
            "cross_entropy",
            "UNKNOWN",  # hardware — fill in from the machine that actually ran this
            round(result["training_time_min"], 2),
            round(result["best_val_accuracy"], 4),
            round(result["best_val_macro_f1"], 4),
            "N/A",  # test_accuracy — only filled by evaluate.py, once, at the end
            "N/A",
            "COMPLETED",
            "",
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    result = train(config)
    log_experiment(config, result)


if __name__ == "__main__":
    main()
