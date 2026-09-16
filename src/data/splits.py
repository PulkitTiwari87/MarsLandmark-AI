"""Split loading and leakage checks for the HiRISE landmark dataset.

The dataset is expected to ship its own official train/val/test split
(reported in docs/DATASET_SELECTION.md as 6,997/2,025/1,793, source-image
grouped) — this module loads whatever split label files exist rather than
re-deriving a split, per the project's preference for preserving official
splits (see docs/DATA_SPLIT.md).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.labels import parse_label_file

SPLIT_FILENAMES = {
    "train": "train-labels.txt",
    "val": "val-labels.txt",
    "test": "test-labels.txt",
}


def load_split(data_dir: Path, split: str) -> list[tuple[str, int]]:
    if split not in SPLIT_FILENAMES:
        raise ValueError(f"Unknown split {split!r}, expected one of {list(SPLIT_FILENAMES)}")
    label_file = Path(data_dir) / SPLIT_FILENAMES[split]
    if not label_file.exists():
        raise FileNotFoundError(
            f"Expected split label file at {label_file}. If the real dataset "
            "uses different filenames, update SPLIT_FILENAMES after inspecting "
            "the downloaded archive (see docs/DATASET.md)."
        )
    return parse_label_file(label_file)


def check_filename_leakage(
    splits: dict[str, list[tuple[str, int]]],
) -> dict[str, list[str]]:
    """Return {filename: [splits it appears in]} for any file in >1 split.

    An empty dict means no exact-filename duplication across splits. This
    does NOT by itself prove there is no leakage from near-duplicate crops
    of the same source image landing in different splits — that requires
    grouping by source image ID, which needs the real filename convention
    to be confirmed first (see docs/DATA_SPLIT.md).
    """
    seen: dict[str, list[str]] = {}
    for split_name, samples in splits.items():
        for filename, _label in samples:
            seen.setdefault(filename, []).append(split_name)
    return {fname: where for fname, where in seen.items() if len(set(where)) > 1}


def class_distribution(samples: list[tuple[str, int]]) -> Counter:
    return Counter(label for _filename, label in samples)
