"""End-to-end smoke test of the training loop against synthetic random
images/labels. This checks the pipeline runs and produces well-formed
metrics — it says nothing about real model performance, since there is
nothing learnable in random noise labels."""

import random

from PIL import Image

from src.training.train import train


def _make_synthetic_split(root, name, n, num_classes):
    lines = []
    for i in range(n):
        fname = f"{name}_{i:03d}.jpg"
        Image.new("L", (48, 48), color=random.randint(0, 255)).save(root / fname)
        lines.append(f"{fname} {random.randint(0, num_classes - 1)}")
    (root / f"{name}-labels.txt").write_text("\n".join(lines) + "\n")


def test_train_runs_end_to_end(tmp_path):
    random.seed(0)
    num_classes = 8
    for split, n in (("train", 16), ("val", 8)):
        _make_synthetic_split(tmp_path, split, n, num_classes)

    config = {
        "seed": 42,
        "data": {
            "raw_dir": str(tmp_path),
            "image_size": 16,
            "batch_size": 4,
            "num_workers": 0,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 5,
        },
        "model": {"name": "simple_cnn", "pretrained": False, "num_classes": num_classes},
        "logging": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }

    result = train(config)
    assert result["epochs_run"] == 1
    assert 0.0 <= result["best_val_accuracy"] <= 1.0
    assert 0.0 <= result["best_val_macro_f1"] <= 1.0
    assert (tmp_path / "checkpoints" / "best.pt").exists()
