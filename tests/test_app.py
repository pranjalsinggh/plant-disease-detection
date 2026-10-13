"""The UI must be honest and usable before a real model has been trained."""

import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from plant_disease.model import PlantDiseaseCNN

APP = Path(__file__).resolve().parents[1] / "app.py"


def memory_upload(color):
    buffer = io.BytesIO()
    Image.new("RGB", (47, 29), color).save(buffer, format="PNG")
    return SimpleNamespace(name="leaf.png", size=len(buffer.getvalue()), getvalue=buffer.getvalue)


@pytest.fixture
def synthetic_prediction_app(tmp_path, monkeypatch):
    """Exercise real inference with test weights, explicitly marked synthetic.

    AppTest does not implement file uploads. Only that widget is substituted;
    checkpoint validation, model loading, button handling and rendering are real.
    This fixture makes no assertion about disease-detection accuracy.
    """
    checkpoint = tmp_path / "synthetic.pt"
    torch.manual_seed(123)
    torch.save(
        {
            "format_version": 1,
            "architecture": "four_layer_cnn",
            "model_state": PlantDiseaseCNN(num_classes=2).state_dict(),
            "classes": ["Apple___Apple_scab", "Apple___healthy"],
            "image_size": 32,
            "epoch": 1,
            "metrics": {},
            "config": {"dropout": 0.4},
            "synthetic_images": True,
        },
        checkpoint,
    )
    monkeypatch.setenv("PLANT_DISEASE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PLANT_DISEASE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PLANT_DISEASE_INDEX", str(tmp_path / "missing.sqlite"))
    uploads = {"file": memory_upload((100, 180, 40))}
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploads["file"])
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert any("Synthetic smoke model" in warning.value for warning in app.warning)
    return app, uploads, checkpoint


def analyze_leaf(app):
    button = next(button for button in app.button if button.label == "Analyze leaf")
    assert not button.disabled
    button.click().run()
    assert not app.exception
    assert not app.error
    return next(metric.value for metric in app.metric if metric.label == "Model score")


def test_app_starts_without_weights_or_fabricated_live_results(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANT_DISEASE_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("PLANT_DISEASE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PLANT_DISEASE_INDEX", str(tmp_path / "missing.sqlite"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert not app.error
    assert any("Awaiting a trained checkpoint" in info.value for info in app.info)
    analyze = next(button for button in app.button if button.label == "Analyze leaf")
    assert analyze.disabled
    assert len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 38
    live_values = [metric.value for metric in app.metric]
    assert "94.03%" not in live_values
    assert "95.94%" not in live_values
    assert any(expander.label == "Reference figures from the project brief" for expander in app.expander)


def test_app_reports_invalid_checkpoint_and_disables_prediction(tmp_path, monkeypatch):
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"model_state": {}}, checkpoint)
    monkeypatch.setenv("PLANT_DISEASE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PLANT_DISEASE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PLANT_DISEASE_INDEX", str(tmp_path / "missing.sqlite"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert any("Could not load the checkpoint" in error.value for error in app.error)
    assert next(button for button in app.button if button.label == "Analyze leaf").disabled


def test_prediction_survives_reruns_and_download_avoids_rerun(synthetic_prediction_app):
    app, _, _ = synthetic_prediction_app
    score = analyze_leaf(app)
    predictions = app.session_state["leaf_predictions"]
    input_identity = app.session_state["leaf_prediction_input"]
    assert len(predictions) == 2
    assert app.get("download_button")[0].proto.ignore_rerun

    for _ in range(2):
        app.run()
        assert not app.exception
        assert app.session_state["leaf_prediction_input"] == input_identity
        assert app.session_state["leaf_predictions"] == predictions
        assert next(metric.value for metric in app.metric if metric.label == "Model score") == score
        assert len(app.get("progress")) == len(predictions)
        assert app.get("download_button")[0].proto.ignore_rerun


def test_new_upload_clears_previous_result_even_with_same_filename(synthetic_prediction_app):
    app, uploads, _ = synthetic_prediction_app
    analyze_leaf(app)
    first_identity = app.session_state["leaf_prediction_input"]

    uploads["file"] = memory_upload((40, 70, 190))
    app.run()
    assert not app.exception
    assert app.session_state["leaf_prediction_input"] != first_identity
    assert app.session_state["leaf_predictions"] is None
    assert all(metric.label != "Model score" for metric in app.metric)
    assert not app.get("download_button")

    analyze_leaf(app)
    assert app.session_state["leaf_predictions"]
    uploads["file"] = None
    app.run()
    assert app.session_state["leaf_prediction_input"] is None
    assert app.session_state["leaf_predictions"] is None
    assert all(metric.label != "Model score" for metric in app.metric)
    assert next(button for button in app.button if button.label == "Analyze leaf").disabled


def test_checkpoint_revision_clears_previous_prediction(synthetic_prediction_app):
    app, _, checkpoint = synthetic_prediction_app
    analyze_leaf(app)
    first_identity = app.session_state["leaf_prediction_input"]
    previous_stat = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["epoch"] = 2
    torch.save(payload, checkpoint)
    # Force a distinct timestamp even on filesystems with coarse clock precision.
    os.utime(checkpoint, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000))
    app.run()

    assert not app.exception
    assert not app.error
    assert app.session_state["leaf_prediction_input"] != first_identity
    assert app.session_state["leaf_predictions"] is None
    assert all(metric.label != "Model score" for metric in app.metric)
    assert not app.get("download_button")
