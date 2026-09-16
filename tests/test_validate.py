"""Tests for dataset validation logic, against a small synthetic fixture
(not the real 940MB archive, which is git-ignored and not present in CI)."""

from pathlib import Path

from PIL import Image

from src.data.validate import validate_dataset


def _make_dataset(root: Path, *, corrupt=False, duplicate=False, missing_label=False):
    images_dir = root / "map-proj-v3"
    images_dir.mkdir(parents=True)

    labels = [
        ("ESP_000001_0001_RED-0001.jpg", "0"),
        ("ESP_000001_0001_RED-0001-r90.jpg", "0"),
        ("ESP_000002_0002_RED-0005.jpg", "1"),
    ]
    for i, (name, _) in enumerate(labels):
        Image.new("L", (227, 227), color=10 + i).save(images_dir / name)

    if duplicate:
        # byte-identical copy of an existing image under a new name
        (images_dir / "ESP_000002_0002_RED-0006.jpg").write_bytes(
            (images_dir / "ESP_000002_0002_RED-0005.jpg").read_bytes()
        )
        labels.append(("ESP_000002_0002_RED-0006.jpg", "1"))

    if corrupt:
        (images_dir / "broken.jpg").write_bytes(b"not an image")
        labels.append(("broken.jpg", "0"))

    label_lines = [f"{name} {cls}" for name, cls in labels]
    if missing_label:
        label_lines = label_lines[:-1]  # drop the last label, leaving its image unlabeled
    (root / "labels-map-proj-v3.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return images_dir


def test_clean_dataset_reports_zero_problems(tmp_path):
    _make_dataset(tmp_path)
    report = validate_dataset(tmp_path)

    assert report["total_images_on_disk"] == 3
    assert report["total_label_lines"] == 3
    assert report["labels_without_matching_image"] == []
    assert report["images_without_matching_label"] == []
    assert report["corrupted_or_unreadable_images"] == []
    assert report["exact_duplicate_content_groups"] == 0
    assert report["class_distribution"] == {"0": 2, "1": 1}
    assert report["unique_source_strip_ids"] == 2


def test_detects_corrupted_image(tmp_path):
    _make_dataset(tmp_path, corrupt=True)
    report = validate_dataset(tmp_path)

    assert len(report["corrupted_or_unreadable_images"]) == 1
    assert report["corrupted_or_unreadable_images"][0]["file"] == "broken.jpg"


def test_detects_exact_duplicate_content(tmp_path):
    _make_dataset(tmp_path, duplicate=True)
    report = validate_dataset(tmp_path)

    assert report["exact_duplicate_content_groups"] == 1
    assert report["files_in_exact_duplicate_groups"] == 2


def test_detects_image_without_label(tmp_path):
    _make_dataset(tmp_path, missing_label=True)
    report = validate_dataset(tmp_path)

    assert report["images_without_matching_label"] == ["ESP_000002_0002_RED-0005.jpg"]
