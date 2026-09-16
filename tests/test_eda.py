"""Tests for EDA computation logic, against a small synthetic fixture."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.eda import run_eda


def _make_dataset(root: Path):
    images_dir = root / "map-proj-v3"
    images_dir.mkdir(parents=True)
    samples = [
        ("ESP_000001_0001_RED-0001.jpg", "0", 50),
        ("ESP_000001_0001_RED-0001-r90.jpg", "0", 60),
        ("ESP_000002_0002_RED-0005.jpg", "1", 200),
        ("ESP_000002_0002_RED-0006.jpg", "1", 210),
    ]
    lines = []
    for name, cls, fill in samples:
        Image.new("L", (10, 10), color=fill).save(images_dir / name)
        lines.append(f"{name} {cls}")
    (root / "labels-map-proj-v3.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return images_dir


def test_run_eda_measures_class_distribution_and_brightness(tmp_path):
    _make_dataset(tmp_path)
    figures_dir = tmp_path / "figures"
    summary = run_eda(tmp_path, figures_dir, seed=1, brightness_sample_size=100)

    assert summary["class_distribution"] == {"0": 2, "1": 2}
    assert summary["strips_total"] == 2
    assert summary["crops_per_strip_min"] == 2
    assert summary["crops_per_strip_max"] == 2
    # class 1 (fills 200, 210) should measure brighter than class 0 (fills 50, 60)
    assert (
        summary["brightness_mean_by_class"]["1"]["mean"]
        > summary["brightness_mean_by_class"]["0"]["mean"]
    )
    assert np.isclose(summary["brightness_mean_by_class"]["0"]["mean"], 55.0)
    assert (figures_dir / "class_distribution.png").exists()
    assert (figures_dir / "crops_per_source_strip.png").exists()
    assert (figures_dir / "brightness_distribution.png").exists()
