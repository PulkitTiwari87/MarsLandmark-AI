from PIL import Image

from src.data.transforms import build_transforms


def test_eval_transform_shape_and_range():
    img = Image.new("L", (64, 64), color=200)
    transform = build_transforms(train=False, image_size=32)
    tensor = transform(img)
    assert tensor.shape == (3, 32, 32)


def test_train_transform_is_deterministic_in_shape():
    img = Image.new("L", (64, 64), color=50)
    transform = build_transforms(train=True, image_size=32)
    for _ in range(5):
        tensor = transform(img)
        assert tensor.shape == (3, 32, 32)


def test_single_channel_mode_skips_normalization_stats():
    img = Image.new("L", (32, 32), color=10)
    transform = build_transforms(train=False, image_size=16, three_channel=False)
    tensor = transform(img)
    assert tensor.shape == (1, 16, 16)
