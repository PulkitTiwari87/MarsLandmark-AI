"""PyTorch Dataset for the HiRISE landmark dataset, using the grouped split.

Images are 227x227 single-channel grayscale (measured, see docs/DATASET.md).
This dataset replicates to 3 channels and applies ImageNet normalization
stats so a pretrained torchvision backbone (Phase 06) can be used directly
without modifying its first conv layer. No additional geometric
augmentation is applied: the archive already ships 6x augmentation per
original landmark (rotations/flips/brightness — see docs/DATASET.md), and
adding more was judged unnecessary rather than assumed beneficial (every
augmentation must have a reason, project rule §8). This can be revisited
in Phase 07 if overfitting is measured.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.split import read_split_manifest

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class HiRISELandmarkDataset(Dataset):
    def __init__(self, data_dir: Path, split_manifest_path: Path, split: str, transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "map-proj-v3"
        self.transform = transform or build_transform()

        labels_path = self.data_dir / "labels-map-proj-v3.txt"
        label_map = dict(
            line.split() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assignment = read_split_manifest(Path(split_manifest_path))

        self.samples: list[tuple[str, int]] = [
            (fname, int(label_map[fname]))
            for fname, s in assignment.items()
            if s == split and fname in label_map
        ]
        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} — check the manifest/data_dir.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fname, label = self.samples[idx]
        with Image.open(self.images_dir / fname) as im:
            im = im.convert("L")
            tensor = self.transform(im)
        return tensor, label
