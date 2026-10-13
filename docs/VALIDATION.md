# Recorded verification

Verified on **13 October 2026** during repository creation. These checks establish that the software works; they do not reproduce the supplied plant-disease accuracy figures.

## Environment

Windows, Python 3.11, PyTorch 2.14.0+cpu, Streamlit 1.55.0, Pillow 12.1.1, NumPy 2.3.5, and pandas 2.3.3. The project installed successfully in editable mode, and `pip check` found no broken requirements. CUDA was unavailable; model verification used CPU.

## Automated checks

`python -m pytest -q`: **27 passed in 38.24 seconds**.

The suite covers the four-convolution architecture, tensor shapes, RGB conversion and normalization, weighted metrics for uneven batches, safe checkpoint loading, label-order preservation, and the public training/evaluation/prediction commands. Dataset checks cover corrupt files, cross-split duplicates, same-split duplicate reporting, class completeness, atomic rebuilds, and closing SQLite file handles on Windows. App checks exercise missing and invalid weights and ensure supplied reference accuracy is not shown as live model performance.

`python -m ruff check .`: **passed**. Python source compilation also passed. GitHub Actions runs lint and tests on Python 3.11/Linux after pushes.

## All-category smoke workflow

Generated **152 synthetic PNGs**, with **114 training** and **38 validation** images across the **38 reference categories**. The image index verified all images without corrupt files or duplicate hashes. A single CPU training epoch at 32×32 resolution produced `best.pt`, `history.csv`, and `run.json`.

Evaluation produced class metrics for all 38 categories and a 38×38 confusion matrix. Prediction returned three ordered class scores. The checkpoint preserved the index's class order. Synthetic provenance was recorded in the dataset, checkpoint, run, evaluation, and CLI prediction output.

These images contain generated coloured shapes, not plant leaves. Their accuracy has no disease-detection meaning. Smoke data and weights were retained outside the published source and are not bundled as a usable disease model.

## Metadata capacity

The same SQLite schema accepted **100,000 synthetic metadata rows** in **3.5739 seconds**, approximately **27,980 rows/second**. The resulting catalogue was **43,225,088 bytes**. A split/class count query took **0.001638 seconds** and used the covering `images_split_class` index.

See [the recorded benchmark JSON](index_benchmark.json). This was one local run with hardware-dependent timings. No image files were created or decoded. It does not measure storage or training performance on a 100,000-image dataset. The benchmark catalogue is marked as synthetic metadata and is rejected by the training data loader.

## Browser check

The Streamlit interface was inspected in a local browser. Its upload controls, missing-model state, disabled prediction button, reference taxonomy, and initial layout rendered correctly. A [preview](app-preview.jpg) is included in the README.

## Still needed for a disease model

A labelled real-image dataset and a full training run are required before making disease predictions. To validate the supplied 94.03% validation and 95.94% training accuracy, retain the original dataset/splits, hyperparameters, epoch history, and trained checkpoint, then evaluate through a documented protocol. No such artifacts were available for this repository creation.
