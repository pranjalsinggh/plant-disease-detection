"""Pillow/NumPy transforms shared by training, evaluation, and inference."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


def validate_image_size(image_size: int) -> int:
    if isinstance(image_size, bool) or not isinstance(image_size, int) or not 32 <= image_size <= 1024:
        raise ValueError("image_size must be an integer between 32 and 1024 pixels.")
    return image_size


class ImageTransform:
    """Convert to RGB, resize, and normalize [0,255] into [-1,1].

    Training additionally uses random horizontal flips, small rotations, and
    brightness/contrast jitter. Evaluation and prediction are deterministic.
    """

    def __init__(self, image_size: int = 128, training: bool = False) -> None:
        self.image_size = validate_image_size(image_size)
        self.training = training

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("Preprocessing expects a PIL.Image.Image.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        # Copy avoids non-writable NumPy buffers and ensures contiguous CHW data.
        return torch.from_numpy(array.transpose(2, 0, 1).copy())


def build_transform(image_size: int = 128, training: bool = False) -> ImageTransform:
    return ImageTransform(image_size=image_size, training=training)


def preprocess_image(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    return ImageTransform(image_size=image_size)(image)
