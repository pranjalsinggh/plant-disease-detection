"""Train the four-layer CNN from an indexed, pre-split image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import prepare_output, resolve_device, save_checkpoint, write_json
from .model import PlantDiseaseCNN
from .transforms import ImageTransform, validate_image_size


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    seed: int = 42
    num_workers: int = 0
    patience: int = 5
    weight_decay: float = 0.0001
    dropout: float = 0.4

    def __post_init__(self) -> None:
        validate_image_size(self.image_size)
        for name in ("batch_size", "epochs", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer between 0 and 4294967295.")
        for name in ("learning_rate", "weight_decay", "dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read training configuration {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training configuration must be valid JSON: {path}") from exc
    return config_from_mapping(raw)


def config_from_mapping(raw: Mapping[str, Any]) -> TrainConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("Training configuration must be a JSON object.")
    unknown = set(raw) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training settings: {', '.join(sorted(unknown))}.")
    return TrainConfig(**dict(raw))


def seed_everything(seed: int) -> None:
    # Required for deterministic CUDA matrix multiplication on supported builds.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Derive Python/NumPy augmentation seeds from the DataLoader worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class IndexedImageDataset(Dataset):
    """Load image pixels lazily; the index holds paths, not image blobs."""

    def __init__(self, samples: Sequence[Mapping[str, Any]], image_size: int = 128, training: bool = False) -> None:
        self.samples = list(samples)
        self.transform = ImageTransform(image_size=image_size, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[position]
        path = Path(sample["path"])
        try:
            with Image.open(path) as image:
                tensor = self.transform(image)
        except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(
                f"Cannot decode indexed image {path}. Restore the image or rebuild the "
                "index with image verification before training."
            ) from exc
        return tensor, int(sample["label_idx"])


def validate_samples(samples: Sequence[Mapping[str, Any]], classes: list[str], split: str) -> None:
    if not samples:
        raise ValueError(f"The index has no '{split}' images. Index a dataset with train/ and valid/ folders.")
    for sample in samples:
        label_idx = sample.get("label_idx")
        if (
            isinstance(label_idx, bool)
            or not isinstance(label_idx, int)
            or not 0 <= label_idx < len(classes)
            or sample.get("label") != classes[label_idx]
        ):
            raise ValueError("Index labels are inconsistent with its class order. Rebuild the image index.")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int]:
    """Compute per-image averages, including a partially filled final batch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss encountered. Check image data and reduce learning_rate.")
            if training:
                loss.backward()
                optimizer.step()
            batch_count = targets.size(0)
            total_loss += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += batch_count
    if count == 0:
        raise ValueError("Cannot calculate metrics for an empty image loader.")
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def train(
    index_path: str | Path,
    config: TrainConfig | Mapping[str, Any],
    output_dir: str | Path,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    from .index import get_summary, load_classes, load_samples

    if not isinstance(config, TrainConfig):
        config = config_from_mapping(config)
    target_device = resolve_device(device)
    dataset_summary = get_summary(index_path)
    if dataset_summary.get("synthetic_metadata", False):
        raise ValueError("This index contains benchmark metadata without image files. Index a real image dataset before training.")
    classes = load_classes(index_path)
    if len(classes) < 2:
        raise ValueError("Training requires at least two indexed classes.")
    train_samples = load_samples(index_path, "train")
    valid_samples = load_samples(index_path, "valid")
    validate_samples(train_samples, classes, "train")
    validate_samples(valid_samples, classes, "valid")
    seen_train_classes = {sample["label_idx"] for sample in train_samples}
    if len(seen_train_classes) != len(classes):
        raise ValueError("Each indexed class needs at least one training image. Rebuild the split or index.")
    output = prepare_output(output_dir, overwrite)
    # An explicitly overwritten run must never expose the previous run's model
    # if the new run fails before completing its first epoch.
    if overwrite:
        (output / "best.pt").unlink(missing_ok=True)
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": target_device.type == "cuda",
        "generator": generator,
    }
    train_loader = DataLoader(
        IndexedImageDataset(train_samples, config.image_size, training=True), shuffle=True, **loader_options
    )
    valid_loader = DataLoader(
        IndexedImageDataset(valid_samples, config.image_size), shuffle=False, **loader_options
    )
    model = PlantDiseaseCNN(len(classes), dropout=config.dropout).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    config_dict = asdict(config)
    run: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "four_layer_cnn",
        "index_path": str(Path(index_path).expanduser().resolve()),
        "dataset": dataset_summary,
        "class_names": classes,
        "config": config_dict,
        "device": str(target_device),
        "torch_version": str(torch.__version__),
        "metric_convention": "Accuracy values are fractions in [0,1]; loss is mean cross-entropy per image.",
        "training_metric_context": "Training metrics use augmented images and active dropout; validation uses neither.",
        "synthetic_images": bool(dataset_summary.get("synthetic_images", False)),
        "results_source": (
            "Synthetic-image smoke test only; these metrics do not measure plant disease detection."
            if dataset_summary.get("synthetic_images", False)
            else "Measured by this training run; no supplied reference metrics are substituted."
        ),
    }
    write_json(output / "run.json", run)
    start = time.perf_counter()
    best_accuracy, best_loss = -1.0, math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    columns = ("epoch", "train_loss", "train_accuracy", "valid_loss", "valid_accuracy", "learning_rate", "epoch_seconds")
    try:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for epoch in range(1, config.epochs + 1):
                epoch_start = time.perf_counter()
                learning_rate = optimizer.param_groups[0]["lr"]
                train_metrics = run_epoch(model, train_loader, criterion, target_device, optimizer)
                valid_metrics = run_epoch(model, valid_loader, criterion, target_device)
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "valid_loss": valid_metrics["loss"],
                    "valid_accuracy": valid_metrics["accuracy"],
                    "learning_rate": learning_rate,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
                writer.writerow(row)
                handle.flush()
                history.append(row)
                accuracy, loss = valid_metrics["accuracy"], valid_metrics["loss"]
                improved = accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss)
                if improved:
                    best_accuracy, best_loss, best_epoch = accuracy, loss, epoch
                    stale_epochs = 0
                    save_checkpoint(
                        output / "best.pt",
                        {
                            "format_version": 1,
                            "architecture": "four_layer_cnn",
                            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                            "classes": classes,
                            "image_size": config.image_size,
                            "epoch": epoch,
                            "metrics": {"train": train_metrics, "valid": valid_metrics},
                            "config": config_dict,
                            "synthetic_images": bool(dataset_summary.get("synthetic_images", False)),
                        },
                    )
                else:
                    stale_epochs += 1
                scheduler.step(loss)
                print(
                    f"Epoch {epoch:03d}/{config.epochs}: train loss={train_metrics['loss']:.4f}, "
                    f"accuracy={train_metrics['accuracy']:.2%}; valid loss={loss:.4f}, "
                    f"accuracy={accuracy:.2%}", flush=True
                )
                if stale_epochs >= config.patience:
                    print(f"Early stopping after {stale_epochs} epochs without an improved checkpoint.", flush=True)
                    break
    except (Exception, KeyboardInterrupt) as exc:
        run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        run["epochs_completed"] = len(history)
        run["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "run.json", run)
        raise
    run.update(
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        epochs_completed=len(history),
        stopped_early=len(history) < config.epochs,
        elapsed_seconds=time.perf_counter() - start,
        best_epoch=best_epoch,
        best_metrics=history[best_epoch - 1],
        final_metrics=history[-1],
        checkpoint="best.pt",
        history="history.csv",
    )
    write_json(output / "run.json", run)
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="SQLite image index path")
    parser.add_argument("--config", default="configs/train.json", help="JSON training configuration")
    parser.add_argument("--output", required=True, help="New directory for best.pt and experiment artifacts")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--epochs", type=int, help="Override configured epochs")
    parser.add_argument("--image-size", type=int, help="Override the configured square image size")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in an existing run directory")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        overrides = {name: value for name, value in {
            "epochs": args.epochs, "image_size": args.image_size, "batch_size": args.batch_size
        }.items() if value is not None}
        if overrides:
            config = replace(config, **overrides)
        run = train(args.index, config, args.output, args.device, args.overwrite)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(2, f"Training error: {exc}\n")
    print(f"Saved best epoch {run['best_epoch']} to {Path(args.output) / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
