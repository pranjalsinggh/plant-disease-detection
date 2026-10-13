"""The UI must be honest and usable before a real model has been trained."""

from pathlib import Path

import torch
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


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
