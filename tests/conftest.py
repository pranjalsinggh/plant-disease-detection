"""Small, deterministic image fixtures; no external data is downloaded."""

from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def efficient_cpu_threads():
    """Small test images run faster without a large CPU thread pool."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def image_factory():
    def create(path: Path, *, seed: int = 0, mode: str = "RGB") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Every seed gets distinct bytes, including images of the same category.
        image = Image.new(mode, (39, 27), seed % 240 if mode == "L" else (20 + seed % 200, 120, 55))
        image.putpixel((seed % 39, seed % 27), 250 if mode == "L" else (250, seed % 255, 0))
        image.save(path)
        return path

    return create


@pytest.fixture
def tiny_dataset(tmp_path, image_factory):
    """A two-category official folder layout, with deliberately uneven batches."""
    root = tmp_path / "dataset"
    labels = ["Apple___Apple_scab", "Apple___healthy"]
    seed = 0
    for split, count in (("train", 3), ("valid", 2), ("test", 1)):
        for label in labels:
            for index in range(count):
                image_factory(root / split / label / f"leaf-{index}.png", seed=seed)
                seed += 1
    return root, labels
