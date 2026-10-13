"""Unequal batch sizes must not distort reported losses or accuracy."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from plant_disease.evaluate import classification_metrics
from plant_disease.train import run_epoch


def test_epoch_metrics_are_weighted_by_images():
    logits = torch.tensor([[10.0, 0.0]] * 5)
    labels = torch.tensor([0, 0, 0, 0, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=4, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    expected_loss = criterion(logits, labels).item()
    metrics = run_epoch(nn.Identity(), loader, criterion, torch.device("cpu"))
    assert metrics["samples"] == 5
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)


def test_confusion_matrix_class_metrics_have_correct_orientation():
    metrics = classification_metrics(np.array([[3, 1], [2, 4]]), ["class-a", "class-b"])
    assert metrics["samples"] == 10
    assert metrics["accuracy"] == pytest.approx(0.7)
    assert metrics["per_class"][0]["precision"] == pytest.approx(3 / 5)
    assert metrics["per_class"][0]["recall"] == pytest.approx(3 / 4)
    assert metrics["per_class"][1]["precision"] == pytest.approx(4 / 5)
    assert metrics["per_class"][1]["recall"] == pytest.approx(4 / 6)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 8 / 11) / 2)


def test_missing_class_support_has_finite_zero_rates():
    metrics = classification_metrics(np.array([[2, 0], [0, 0]]), ["present", "absent"])
    assert metrics["per_class"][1] == {
        "label": "absent", "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }
    assert metrics["macro_f1"] == pytest.approx(0.5)
