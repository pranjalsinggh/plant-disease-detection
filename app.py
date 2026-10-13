"""Local Streamlit workspace for inspecting and using trained leaf classifiers."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from plant_disease.index import get_summary
from plant_disease.labels import DEFAULT_CLASSES
from plant_disease.predict import load_predictor

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("PLANT_DISEASE_CHECKPOINT", str(ROOT / "artifacts/run/best.pt")))
RUN_DIR = Path(os.getenv("PLANT_DISEASE_RUN_DIR", str(CHECKPOINT.parent)))
INDEX = Path(os.getenv("PLANT_DISEASE_INDEX", str(ROOT / "data/index.sqlite")))


def display_label(label: str) -> tuple[str, str]:
    crop, _, condition = label.partition("___")
    return crop.replace("_", " "), condition.replace("_", " ") or label


@st.cache_resource(max_entries=2)
def cached_predictor(path: str, modified_ns: int):
    """The timestamp invalidates the cache when a checkpoint is replaced."""
    return load_predictor(path, device="cpu")


def read_run() -> dict | None:
    path = RUN_DIR / "run.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Expected an object in run.json")
        return value
    except (OSError, ValueError) as exc:
        st.error(f"Unable to read this training report: {exc}")
        return None


def main() -> None:
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
    st.markdown(
        """<style>
        .block-container {max-width:1200px;padding-top:2.7rem;padding-bottom:3rem;}
        h1 {letter-spacing:-.045em;font-size:3.3rem!important;line-height:1.08!important;}
        h2,h3 {letter-spacing:-.025em;}
        [data-testid="stMetric"] {background:#edf2e7;border-radius:14px;padding:18px;}
        .eyebrow {font-size:.77rem;letter-spacing:.19em;font-weight:700;color:#267354;}
        .hero-copy {color:#52665d;font-size:1.1rem;max-width:630px;line-height:1.7;}
        .rule {height:1px;background:#dae4d5;margin:2rem 0 1rem;}
        </style>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("🌿 Leaf lab")
        st.caption("PLANT DISEASE DETECTION")
        st.divider()
        st.subheader("Model status")
        predictor = None
        checkpoint_revision = None
        if CHECKPOINT.is_file():
            try:
                checkpoint_revision = CHECKPOINT.stat().st_mtime_ns
                predictor = cached_predictor(str(CHECKPOINT), checkpoint_revision)
                st.success("Trained checkpoint loaded")
                st.caption(f"{len(predictor.classes)} categories · {predictor.image_size}px input")
                if predictor.metadata.get("synthetic_images"):
                    st.warning("Synthetic smoke model: generated patterns, not plant images.")
            except Exception as exc:
                st.error(f"Could not load the checkpoint: {exc}")
        else:
            st.info("Awaiting a trained checkpoint")
        st.divider()
        st.caption("Four convolutional blocks. One repeatable workflow.")
        st.markdown("[View source on GitHub](https://github.com/pranjalsinggh/plant-disease-detection)")

    st.markdown('<p class="eyebrow">COMPUTER VISION / AGRICULTURE</p>', unsafe_allow_html=True)
    st.title("A closer look at plant health.")
    st.markdown(
        '<p class="hero-copy">From a leaf image to a ranked prediction. '
        'Train, evaluate, and explore a compact CNN built for 38 plant health categories.</p>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Reference taxonomy", "38 categories")
    b.metric("CNN architecture", "4 conv layers")
    c.metric("Image catalogue", "SQLite")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    diagnose, results, dataset = st.tabs(["Diagnose a leaf", "Training results", "Dataset & method"])

    with diagnose:
        preview, analysis = st.columns([1.05, 1], gap="large")
        with preview:
            st.subheader("Start with a clear leaf image")
            st.caption("Use an evenly lit photograph with one leaf in focus. JPG, PNG, or WebP, up to 10 MB.")
            uploaded = st.file_uploader("Leaf photograph", type=["jpg", "jpeg", "png", "webp"])
            leaf = None
            if uploaded is not None:
                try:
                    if uploaded.size > 10 * 1024 * 1024:
                        raise ValueError("Please use an image smaller than 10 MB.")
                    with Image.open(io.BytesIO(uploaded.getvalue())) as raw:
                        raw.load()
                        leaf = ImageOps.exif_transpose(raw).convert("RGB")
                    st.image(leaf, caption="Your leaf photograph", width="stretch")
                except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                    st.error(f"This image could not be opened: {exc}")
        input_key = None
        if predictor is not None and leaf is not None:
            input_key = (
                str(CHECKPOINT), checkpoint_revision,
                hashlib.sha256(uploaded.getvalue()).hexdigest(),
            )
        if st.session_state.get("leaf_prediction_input") != input_key:
            st.session_state["leaf_prediction_input"] = input_key
            st.session_state["leaf_predictions"] = None

        with analysis:
            st.subheader("Prediction")
            if predictor is None:
                st.info("Train a model to unlock predictions. No pretrained weights are included.")
                with st.expander("Set up the model"):
                    st.code(
                        "python -m plant_disease.index --data-dir data/dataset --database data/index.sqlite\n"
                        "python -m plant_disease.train --index data/index.sqlite "
                        "--config configs/train.json --output artifacts/run",
                        language="bash",
                    )
                    st.caption("Restart or rerun the app after training creates artifacts/run/best.pt.")
            elif leaf is None:
                st.caption("Your top predictions will appear here after you upload an image.")
            if st.button("Analyze leaf", type="primary", disabled=predictor is None or leaf is None):
                try:
                    with st.spinner("Inspecting leaf features…"):
                        st.session_state["leaf_predictions"] = predictor.predict(leaf, top_k=3)
                except Exception as exc:
                    st.session_state["leaf_predictions"] = None
                    st.error(f"Prediction failed: {exc}")
            predictions = st.session_state.get("leaf_predictions")
            if predictions and input_key is not None:
                crop, condition = display_label(predictions[0]["label"])
                st.caption(crop.upper())
                st.subheader(condition.capitalize())
                score = predictions[0]["probability"]
                st.metric("Model score", f"{score:.1%}")
                if score < 0.65:
                    st.warning("Low model confidence. Try a clearer image and compare the alternatives.")
                st.caption("Scores are uncalibrated probabilities among known classes. "
                           "The model can still be confident on an unfamiliar image.")
                for item in predictions:
                    crop_name, disease = display_label(item["label"])
                    st.write(f"**{crop_name} · {disease}** — {item['probability']:.1%}")
                    st.progress(float(item["probability"]))
                st.download_button("Download prediction", json.dumps(predictions, indent=2),
                                   file_name="leaf-prediction.json", mime="application/json",
                                   on_click="ignore")

    with results:
        st.subheader("Measured results from your training run")
        run = read_run()
        history_path = RUN_DIR / "history.csv"
        if run and run.get("synthetic_images"):
            st.warning("This run used generated smoke images. Its scores do not measure disease detection.")
        if run is None:
            st.info("No training report yet. This tab will show measured results after training.")
        if history_path.is_file():
            try:
                history = pd.read_csv(history_path)
                if "epoch" not in history:
                    raise ValueError("Missing epoch column")
                history = history.set_index("epoch")
                accuracy_cols = [c for c in history if "accuracy" in c]
                loss_cols = [c for c in history if "loss" in c]
                left, right = st.columns(2)
                with left:
                    st.write("**Accuracy by epoch**")
                    if accuracy_cols:
                        st.line_chart(history[accuracy_cols])
                with right:
                    st.write("**Loss by epoch**")
                    if loss_cols:
                        st.line_chart(history[loss_cols])
                st.dataframe(history, width="stretch")
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Unable to display training history: {exc}")
        if run is not None:
            with st.expander("Run metadata"):
                st.json(run)
        with st.expander("Reference figures from the project brief"):
            st.caption("Supplied by the repository owner; not reproduced or verified by this implementation.")
            st.write("Training accuracy: **95.94%** · Validation accuracy: **94.03%**")
            st.write("Training loss: **1.8891 → 0.1929** · Approximately **79,000** training images")
            st.caption("These figures are never used as live model metrics. Actual file counts and results "
                       "are collected by the indexer and training pipeline.")

    with dataset:
        st.subheader("A dataset you can inspect")
        if INDEX.is_file():
            try:
                summary = get_summary(INDEX)
                x, y, z = st.columns(3)
                x.metric("Indexed images", f"{summary['total_images']:,}")
                y.metric("Training split", f"{summary['splits'].get('train', 0):,}")
                z.metric("Validation split", f"{summary['splits'].get('valid', 0):,}")
            except Exception as exc:
                st.error(f"Unable to read the dataset index: {exc}")
        else:
            st.caption("Index a dataset to see its actual image counts here.")
        st.write("The catalogue stores image paths, class labels, dimensions, file sizes, and SHA-256 hashes. "
                 "Images are loaded on demand for training. Exact duplicate content across splits is rejected.")
        st.markdown("[Dataset setup guide](https://github.com/pranjalsinggh/plant-disease-detection/blob/main/docs/DATASET.md)")
        st.subheader("How the model learns")
        st.code("RGB image → resize & normalize\n"
                "Conv 32 → ReLU → MaxPool → Dropout\n"
                "Conv 64 → ReLU → MaxPool → Dropout\n"
                "Conv 128 → ReLU → MaxPool → Dropout\n"
                "Conv 256 → ReLU → MaxPool → Dropout\n"
                "Adaptive pooling → Dense → class logits", language="text")
        st.caption("Training uses augmentation, cross-entropy loss, and Adam. "
                   "Validation uses the same preprocessing as inference without random augmentation.")
        st.subheader("Reference categories")
        st.dataframe(pd.DataFrame([
            {"Crop": display_label(label)[0], "Condition": display_label(label)[1]}
            for label in DEFAULT_CLASSES
        ]), hide_index=True, width="stretch")
        st.caption("Clean-background training images may generalize poorly to field photographs. "
                   "Keep related leaves and augmented copies in the same split, and evaluate on separate field data.")


if __name__ == "__main__":
    main()
