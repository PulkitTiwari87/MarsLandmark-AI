"""Final test-set evaluation. Run this exactly once, after the model is
frozen and no further tuning will happen — see docs/DATASET_SELECTION.md
and project rule 0.6 (test set integrity). This script deliberately only
reads the TEST split; it has no code path that touches train/val.

Usage:
    python -m src.evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/best.pt --output reports/final_test_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import HiRISELandmarkDataset
from src.data.transforms import build_transforms
from src.models.factory import build_model
from src.training.metrics import compute_classification_metrics


def evaluate(config: dict, checkpoint_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])

    test_ds = HiRISELandmarkDataset(
        image_dir=raw_dir,
        label_file=raw_dir / "test-labels.txt",
        transform=build_transforms(train=False, image_size=data_cfg["image_size"]),
    )
    test_loader = DataLoader(test_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])

    model_cfg = config["model"]
    model = build_model(model_cfg["name"], model_cfg["num_classes"], pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_labels.extend(labels.tolist())
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())

    return compute_classification_metrics(all_labels, all_preds, test_ds.class_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    metrics = evaluate(config, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
