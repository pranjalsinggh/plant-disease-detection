"""Evaluate a saved model on an indexed valid or held-out test split."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import prepare_output, write_json
from .predict import load_predictor
from .train import IndexedImageDataset, validate_samples


def classification_metrics(confusion: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Rows are true labels; columns are predictions. Undefined rates are zero."""
    confusion = np.asarray(confusion)
    if confusion.shape != (len(classes), len(classes)) or (confusion < 0).any():
        raise ValueError("Confusion matrix dimensions must match non-negative class counts.")
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    precision = np.divide(true_positive, predicted, out=np.zeros(len(classes)), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(classes)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes)), where=precision + recall != 0)
    total = int(confusion.sum())
    if total == 0:
        raise ValueError("Cannot evaluate an empty confusion matrix.")
    return {
        "samples": total,
        "accuracy": float(true_positive.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "per_class": [
            {"label": label, "support": int(support[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(classes)
        ],
    }


def evaluate(
    index_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "valid",
    batch_size: int = 32,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    from .index import get_summary, load_classes, load_samples

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test.")
    dataset_summary = get_summary(index_path)
    if dataset_summary.get("synthetic_metadata", False):
        raise ValueError("This index contains benchmark metadata without image files. Index real images before evaluation.")
    predictor = load_predictor(checkpoint, device=device)
    classes = load_classes(index_path)
    if classes != predictor.classes:
        raise ValueError("The index's class names/order do not match the checkpoint. Evaluate against its original dataset.")
    samples = load_samples(index_path, split)
    validate_samples(samples, classes, split)
    output = prepare_output(output_dir, overwrite)
    loader = DataLoader(IndexedImageDataset(samples, predictor.image_size), batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(predictor.device)
            targets = targets.to(predictor.device)
            logits = predictor.model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Checkpoint produced non-finite loss on this dataset.")
            total_loss += loss.item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            flattened = (targets * len(classes) + predictions).cpu().numpy()
            confusion += np.bincount(flattened, minlength=len(classes) ** 2).reshape(confusion.shape)
    metrics = classification_metrics(confusion, classes)
    metrics.update(
        loss=total_loss / len(samples),
        split=split,
        class_names=classes,
        checkpoint=str(Path(checkpoint).expanduser().resolve()),
        checkpoint_epoch=predictor.metadata["epoch"],
        index_path=str(Path(index_path).expanduser().resolve()),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        synthetic_images=bool(dataset_summary.get("synthetic_images", False)),
        synthetic_checkpoint=bool(predictor.metadata.get("synthetic_images", False)),
        results_source=(
            "Synthetic-image smoke test only; these metrics do not measure plant disease detection."
            if dataset_summary.get("synthetic_images", False) or predictor.metadata.get("synthetic_images", False)
            else "Measured evaluation on the indexed split."
        ),
        metric_convention="Rates are fractions in [0,1]; macro averages include every indexed class; undefined rates are zero.",
    )
    with (output / "confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual / predicted", *classes])
        for label, row in zip(classes, confusion.tolist()):
            writer.writerow([label, *row])
    write_json(output / "metrics.json", metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="SQLite image index path")
    parser.add_argument("--checkpoint", required=True, help="A trained best.pt checkpoint")
    parser.add_argument("--output", required=True, help="New directory for metrics.json and confusion.csv")
    parser.add_argument("--split", default="valid", choices=("valid", "test"), help="Split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing evaluation artifacts")
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(args.index, args.checkpoint, args.output, args.split, args.batch_size, args.device, args.overwrite)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
    print(f"{metrics['split']}: {metrics['samples']} images, accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.4f}")
    print(f"Saved evaluation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
