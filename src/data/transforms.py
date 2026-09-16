"""Preprocessing and augmentation transforms.

Every augmentation here is chosen for a stated physical/scientific reason,
per the project's augmentation policy — none are added purely because they
"tend to help":

- Horizontal/vertical flip and 90-degree rotation: HiRISE crops are
  map-projected overhead imagery with no canonical "up" relative to the
  landmark itself (a crater or dune looks like a crater or dune from any
  cardinal orientation). These are label-preserving transformations of the
  physical scene, not just of the pixels.
- No color jitter: HiRISE crops are single-channel reflectance data, not
  RGB photography, so jitter would not model any real illumination
  physics without further justification. It is intentionally omitted.
- Channel replication (grayscale -> 3-channel) is applied only when using
  an ImageNet-pretrained backbone, which expects 3-channel input. This is
  a shape-compatibility step, not a source of new information — it must
  not be described as part of the original dataset.
"""

from __future__ import annotations

import random

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomNinetyRotation:
    """Rotate by a random multiple of 90 degrees (label-preserving for
    overhead imagery, unlike arbitrary-angle rotation which introduces
    padding artifacts)."""

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return T.functional.rotate(img, angle)


def build_transforms(train: bool, image_size: int, three_channel: bool = True) -> T.Compose:
    ops: list = []
    if three_channel:
        ops.append(T.Grayscale(num_output_channels=3))
    ops.append(T.Resize((image_size, image_size)))
    if train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            RandomNinetyRotation(),
        ]
    ops.append(T.ToTensor())
    if three_channel:
        # ImageNet stats are appropriate only because we reuse an
        # ImageNet-pretrained backbone; see docs/MODEL_ARCHITECTURE.md.
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)
