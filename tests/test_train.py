"""Fast CPU smoke tests for the training loop, against a tiny synthetic
dataset (not the real archive) — verifies the loop runs end-to-end and
that the experiment registry is written correctly, not that any
particular accuracy is achieved."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import HiRISELandmarkDataset
from src.data.split import write_split_manifest
from src.models.baseline import SimpleCNN
from src.training.train import EXPERIMENT_CSV_FIELDS, evaluate, fit, log_experiment


def _make_tiny_dataset(root: Path, n_per_split=6):
    images_dir = root / "map-proj-v3"
    images_dir.mkdir(parents=True)
    labels, manifest = [], {}
    for split_idx, split in enumerate(["train", "val"]):
        for i in range(n_per_split):
            cls = i % 2
            fname = f"ESP_{split_idx:06d}_0001_RED-{i:04d}.jpg"
            Image.new("L", (32, 32), color=(i * 20) % 256).save(images_dir / fname)
            labels.append(f"{fname} {cls}")
            manifest[fname] = split
    (root / "labels-map-proj-v3.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    manifest_path = root / "split_manifest.csv"
    write_split_manifest(manifest, manifest_path)
    return manifest_path


def test_fit_runs_and_improves_or_completes_without_error(tmp_path):
    manifest_path = _make_tiny_dataset(tmp_path)
    train_ds = HiRISELandmarkDataset(tmp_path, manifest_path, "train")
    val_ds = HiRISELandmarkDataset(tmp_path, manifest_path, "val")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = SimpleCNN(num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    class_names = {0: "a", 1: "b"}

    result = fit(
        model, train_loader, val_loader, optimizer, criterion,
        torch.device("cpu"), epochs=2, class_names=class_names,
    )

    assert len(result["history"]["train_loss"]) == 2
    assert len(result["history"]["val_macro_f1"]) == 2
    assert result["best_state_dict"] is not None
    assert 0.0 <= result["best_val_macro_f1"] <= 1.0


def test_evaluate_returns_loss_and_metrics(tmp_path):
    manifest_path = _make_tiny_dataset(tmp_path)
    val_ds = HiRISELandmarkDataset(tmp_path, manifest_path, "val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    model = SimpleCNN(num_classes=2)
    loss, metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), torch.device("cpu"), {0: "a", 1: "b"})
    assert loss >= 0
    assert "accuracy" in metrics and "macro_f1" in metrics


def test_log_experiment_writes_header_and_row(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    log_experiment(csv_path, experiment_id="exp001", model="simple_cnn", status="COMPLETED")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert reader[0]["experiment_id"] == "exp001"
    assert reader[0]["model"] == "simple_cnn"
    assert reader[0]["val_accuracy"] == ""  # not fabricated, left empty when not passed
    assert list(reader[0].keys()) == EXPERIMENT_CSV_FIELDS
