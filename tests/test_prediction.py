"""Check that the deployed predictor uses the saved weights and label order."""

import pytest
import torch
from PIL import Image

from plant_disease.evaluate import evaluate
from plant_disease.index import build_index
from plant_disease.model import PlantDiseaseCNN
from plant_disease.predict import load_predictor
from plant_disease.transforms import ImageTransform


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(123)
    model = PlantDiseaseCNN(num_classes=2, dropout=0.4).eval()
    classes = ["Apple___Apple_scab", "Apple___healthy"]
    payload = {
        "format_version": 1,
        "architecture": "four_layer_cnn",
        "model_state": model.state_dict(),
        "classes": classes,
        "image_size": 32,
        "epoch": 1,
        "metrics": {"accuracy": 0.5},
        "config": {"dropout": 0.4},
    }
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path, model, classes


def test_saved_model_and_loaded_prediction_agree(checkpoint):
    path, model, classes = checkpoint
    image = Image.new("RGB", (47, 23), (130, 80, 200))
    tensor = ImageTransform(image_size=32, training=False)(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    order = torch.argsort(probabilities, descending=True).tolist()

    predictor = load_predictor(path, device="cpu")
    results = predictor.predict(image, top_k=2)
    assert predictor.classes == classes
    assert [result["label"] for result in results] == [classes[index] for index in order]
    assert [result["probability"] for result in results] == pytest.approx(
        [probabilities[index].item() for index in order], abs=1e-6
    )
    assert sum(result["probability"] for result in results) == pytest.approx(1.0)
    assert predictor.predict(image, top_k=2) == results


def test_invalid_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"classes": ["invented"], "model_state": {}}, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_predictor(path, device="cpu")


def test_duplicate_checkpoint_labels_are_rejected(checkpoint, tmp_path):
    path, _, _ = checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["classes"] = ["same", "same"]
    invalid = tmp_path / "duplicate-labels.pt"
    torch.save(payload, invalid)
    with pytest.raises((ValueError, RuntimeError)):
        load_predictor(invalid, device="cpu")


def test_top_k_is_bounded_by_available_classes(checkpoint):
    path, _, _ = checkpoint
    predictor = load_predictor(path, device="cpu")
    results = predictor.predict(Image.new("RGB", (32, 32), "green"), top_k=38)
    assert len(results) == 2


def test_evaluation_rejects_reordered_checkpoint_categories(checkpoint, tiny_dataset, tmp_path):
    path, _, _ = checkpoint
    dataset, _ = tiny_dataset
    database = tmp_path / "images.sqlite"
    build_index(dataset, database, allow_subset=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["classes"] = list(reversed(payload["classes"]))
    mismatched = tmp_path / "mismatched.pt"
    torch.save(payload, mismatched)
    with pytest.raises(ValueError, match="(?i)(class|order|match)"):
        evaluate(database, mismatched, tmp_path / "evaluation", device="cpu")
