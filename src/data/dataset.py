"""PyTorch Dataset for the HiRISE landmark crops."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.data.labels import DEFAULT_CLASS_NAMES, load_class_map, parse_label_file


class HiRISELandmarkDataset(Dataset):
    def __init__(self, image_dir: Path, label_file: Path, class_map_file: Path | None = None, transform=None):
        self.image_dir = Path(image_dir)
        self.samples = parse_label_file(label_file)
        self.class_map = load_class_map(class_map_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, label = self.samples[idx]
        image = Image.open(self.image_dir / filename).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> list[str]:
        return [self.class_map.get(i, name) for i, name in enumerate(DEFAULT_CLASS_NAMES)]
