"""Exercise the public commands against actual temporary images and weights."""

import csv
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def run_command(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    # Avoid expensive CPU thread startup for tiny integration fixtures.
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=PROJECT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_index_train_evaluate_predict_commands(tiny_dataset, tmp_path):
    dataset, labels = tiny_dataset
    database = tmp_path / "index.sqlite"
    indexed = run_command(
        "-m", "plant_disease.index", "--data-dir", dataset, "--database", database, "--allow-subset"
    )
    assert json.loads(indexed.stdout)["total_images"] == 12
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps({"image_size": 32, "batch_size": 4, "epochs": 1, "num_workers": 0, "seed": 123}),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    run_command(
        "-m", "plant_disease.train", "--index", database, "--config", config,
        "--output", output, "--device", "cpu",
    )
    checkpoint = output / "best.pt"
    assert checkpoint.is_file()
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["epochs_completed"] == report["best_epoch"] == 1
    assert report["class_names"] == sorted(labels)
    with (output / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 1
    assert 0 <= float(history[0]["valid_accuracy"]) <= 1
    assert float(history[0]["train_loss"]) > 0

    # Row insertion order must not alter the persisted numeric class mapping.
    with closing(sqlite3.connect(database)) as connection:
        classes = connection.execute("SELECT label_idx, label FROM classes ORDER BY label_idx").fetchall()
        connection.execute("DELETE FROM classes")
        connection.executemany("INSERT INTO classes(label_idx, label) VALUES (?, ?)", reversed(classes))
        connection.commit()
    evaluation = tmp_path / "evaluation"
    run_command(
        "-m", "plant_disease.evaluate", "--index", database, "--checkpoint", checkpoint,
        "--output", evaluation, "--batch-size", "3", "--device", "cpu",
    )
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["samples"] == 4
    assert metrics["accuracy"] == pytest.approx(float(history[0]["valid_accuracy"]))
    assert metrics["loss"] == pytest.approx(float(history[0]["valid_loss"]), abs=1e-6)
    assert metrics["class_names"] == sorted(labels)
    assert sum(row["support"] for row in metrics["per_class"]) == 4
    with (evaluation / "confusion.csv").open(newline="", encoding="utf-8") as handle:
        confusion = list(csv.reader(handle))
    assert confusion[0][1:] == sorted(labels)
    assert sum(int(value) for row in confusion[1:] for value in row[1:]) == 4

    predicted = run_command(
        "-m", "plant_disease.predict", "--checkpoint", checkpoint,
        "--image", dataset / "test" / labels[0] / "leaf-0.png", "--top-k", "2", "--device", "cpu",
    )
    predictions = json.loads(predicted.stdout)["predictions"]
    assert len(predictions) == 2
    assert {prediction["label"] for prediction in predictions} == set(labels)
    assert sum(prediction["probability"] for prediction in predictions) == pytest.approx(1.0)
    assert predictions[0]["probability"] >= predictions[1]["probability"]


def test_smoke_generator_creates_all_38_categories(tmp_path):
    dataset = tmp_path / "synthetic"
    run_command(
        PROJECT / "scripts" / "create_smoke_dataset.py", "--output", dataset,
        "--train-per-class", "1", "--valid-per-class", "1", "--size", "32",
    )
    manifest = json.loads((dataset / "SYNTHETIC_DATASET.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert manifest["class_count"] == 38
    result = run_command("-m", "plant_disease.index", "--data-dir", dataset, "--database", tmp_path / "index.sqlite")
    summary = json.loads(result.stdout)
    assert summary["total_images"] == 76
    assert summary["class_count"] == 38
    assert summary["same_split_duplicates"] == 0
    assert summary["synthetic_images"] is True
