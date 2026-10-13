"""Create tiny synthetic images to exercise indexing/training, not disease accuracy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from plant_disease.labels import DEFAULT_CLASSES


def create_smoke_dataset(output: str | Path, *, train_per_class: int = 3, valid_per_class: int = 1, size: int = 96, seed: int = 42) -> dict:
    """Create deterministic, distinct images and retain any unrelated files.

    Every image has a unique seed and an embedded pixel identifier to prevent
    accidental exact duplicates across train/valid. These are coloured shapes,
    not photographs of real leaves and cannot establish model performance.
    """
    if train_per_class < 1 or valid_per_class < 1 or size < 32:
        raise ValueError("Use at least one image per class/split and a size of at least 32.")
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    image_id = 0
    for split, count in (("train", train_per_class), ("valid", valid_per_class)):
        counts[split] = 0
        for class_id, label in enumerate(DEFAULT_CLASSES):
            directory = root / split / label
            directory.mkdir(parents=True, exist_ok=True)
            for item in range(count):
                rng = random.Random(seed + image_id * 997)
                color = ((class_id * 41 + 31) % 256, (class_id * 67 + 79) % 256, (class_id * 23 + 137) % 256)
                image = Image.new("RGB", (size, size), color)
                draw = ImageDraw.Draw(image)
                for _ in range(8):
                    x, y = rng.randrange(size), rng.randrange(size)
                    radius = rng.randrange(2, max(3, size // 5))
                    draw.ellipse((x, y, x + radius, y + radius), fill=tuple(rng.randrange(256) for _ in range(3)))
                # Lossless PNG preserves this identifier exactly.
                image.putpixel((0, 0), (image_id & 255, (image_id >> 8) & 255, (image_id >> 16) & 255))
                image.save(directory / f"synthetic_{item:03d}.png")
                image_id += 1
                counts[split] += 1
    manifest = {
        "synthetic": True,
        "purpose": "Smoke testing only: generated coloured shapes, not real plant images.",
        "root": str(root),
        "class_count": len(DEFAULT_CLASSES),
        "created_images": sum(counts.values()),
        "splits": counts,
        "seed": seed,
    }
    (root / "SYNTHETIC_DATASET.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--data-dir", dest="output", type=Path, default=Path("data/smoke"))
    parser.add_argument("--train-per-class", type=int, default=3)
    parser.add_argument("--valid-per-class", type=int, default=1)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = create_smoke_dataset(args.output, train_per_class=args.train_per_class, valid_per_class=args.valid_per_class, size=args.size, seed=args.seed)
    except ValueError as error:
        parser.exit(2, f"Cannot create synthetic fixture: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
