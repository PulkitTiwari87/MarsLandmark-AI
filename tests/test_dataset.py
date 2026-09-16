"""Dataset tests use synthetic solid-color images generated on the fly —
NOT real NASA/HiRISE imagery. They verify the loading code is correct, not
anything about the real dataset's content."""

from PIL import Image

from src.data.dataset import HiRISELandmarkDataset
from src.data.transforms import build_transforms


def _make_synthetic_dataset(tmp_path):
    for name in ("a.jpg", "b.jpg"):
        Image.new("L", (32, 32), color=128).save(tmp_path / name)
    label_file = tmp_path / "labels.txt"
    label_file.write_text("a.jpg 0\nb.jpg 1\n")
    return tmp_path, label_file


def test_dataset_length_and_getitem(tmp_path):
    image_dir, label_file = _make_synthetic_dataset(tmp_path)
    ds = HiRISELandmarkDataset(image_dir, label_file, transform=build_transforms(train=False, image_size=16))
    assert len(ds) == 2
    image, label = ds[0]
    assert image.shape == (3, 16, 16)  # 3-channel after grayscale replication
    assert label == 0


def test_dataset_class_names_default(tmp_path):
    image_dir, label_file = _make_synthetic_dataset(tmp_path)
    ds = HiRISELandmarkDataset(image_dir, label_file)
    assert len(ds.class_names) == 8
