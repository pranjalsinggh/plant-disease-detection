"""CPU checks of the claimed architecture and image contract."""

import pytest
import torch
from PIL import Image
from torch import nn

from plant_disease.model import PlantDiseaseCNN
from plant_disease.transforms import ImageTransform


def test_four_layer_cnn_returns_logits_for_requested_classes():
    model = PlantDiseaseCNN(num_classes=38, dropout=0.4).eval()
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert len(convolutions) == 4
    assert sum(isinstance(module, nn.ReLU) for module in model.modules()) >= 4
    assert sum(isinstance(module, nn.MaxPool2d) for module in model.modules()) == 4
    assert any(isinstance(module, (nn.Dropout, nn.Dropout2d)) for module in model.modules())
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 32, 32))
    assert logits.shape == (2, 38)
    assert torch.isfinite(logits).all()


def test_rgb_preprocessing_resizes_and_normalizes():
    transform = ImageTransform(image_size=32, training=False)
    white = transform(Image.new("RGB", (61, 19), "white"))
    black = transform(Image.new("RGB", (61, 19), "black"))
    assert white.shape == black.shape == (3, 32, 32)
    assert white.dtype == torch.float32
    assert torch.allclose(white, torch.ones_like(white))
    assert torch.allclose(black, -torch.ones_like(black))


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_preprocessing_converts_non_rgb_images(mode):
    image = Image.new(mode, (21, 37), 128 if mode == "L" else (12, 34, 56, 127))
    tensor = ImageTransform(image_size=32, training=False)(image)
    assert tensor.shape == (3, 32, 32)
    assert torch.isfinite(tensor).all()
    assert tensor.min() >= -1 and tensor.max() <= 1


def test_inference_preprocessing_is_deterministic():
    image = Image.new("RGB", (21, 37), (12, 34, 56))
    transform = ImageTransform(image_size=32, training=False)
    assert torch.equal(transform(image), transform(image))
