"""Tests for HiRISELandmarkDataset, against a small synthetic fixture."""

from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import HiRISELandmarkDataset
from src.data.split import write_split_manifest


def _make_dataset(root: Path):
    images_dir = root / "map-proj-v3"
    images_dir.mkdir(parents=True)
    samples = [
        ("ESP_000001_0001_RED-0001.jpg", "0", "train"),
        ("ESP_000001_0001_RED-0001-r90.jpg", "0", "train"),
        ("ESP_000002_0002_RED-0005.jpg", "1", "val"),
        ("ESP_000003_0003_RED-0007.jpg", "2", "test"),
    ]
    lines = []
    manifest = {}
    for name, cls, split in samples:
        Image.new("L", (227, 227), color=100).save(images_dir / name)
        lines.append(f"{name} {cls}")
        manifest[name] = split
    (root / "labels-map-proj-v3.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = root / "split_manifest.csv"
    write_split_manifest(manifest, manifest_path)
    return manifest_path


def test_dataset_filters_by_split_and_returns_tensor_label(tmp_path):
    manifest_path = _make_dataset(tmp_path)

    train_ds = HiRISELandmarkDataset(tmp_path, manifest_path, "train")
    val_ds = HiRISELandmarkDataset(tmp_path, manifest_path, "val")
    test_ds = HiRISELandmarkDataset(tmp_path, manifest_path, "test")

    assert len(train_ds) == 2
    assert len(val_ds) == 1
    assert len(test_ds) == 1

    image, label = train_ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 227, 227)  # replicated to 3 channels
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_dataset_rejects_invalid_split(tmp_path):
    manifest_path = _make_dataset(tmp_path)
    try:
        HiRISELandmarkDataset(tmp_path, manifest_path, "bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
