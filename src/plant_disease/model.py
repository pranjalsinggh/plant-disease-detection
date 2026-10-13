"""A compact CNN trained from scratch with four convolutional layers."""

from __future__ import annotations

import torch
from torch import nn


class PlantDiseaseCNN(nn.Module):
    """Four Conv/ReLU/MaxPool blocks with dropout and an adaptive classifier.

    The default head has 38 outputs. A different class count supports datasets
    with their own categories and small synthetic integration tests.
    """

    def __init__(self, num_classes: int = 38, dropout: float = 0.4) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")
        self.num_classes = num_classes
        layers: list[nn.Module] = []
        channels = (3, 32, 64, 128, 256)
        for incoming, outgoing in zip(channels, channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout2d(p=dropout / 2),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected an image tensor shaped [batch, 3, height, width].")
        if images.shape[2] < 32 or images.shape[3] < 32:
            raise ValueError("Input height and width must each be at least 32 pixels.")
        if not images.is_floating_point():
            raise ValueError("Input images must be floating point tensors after preprocessing.")
        return self.classifier(self.pool(self.features(images)))
