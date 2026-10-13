# Dataset setup and catalogue

No plant photographs or trained weights are distributed with this repository. Download the [New Plant Diseases Dataset from Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) separately, inspect its current terms, and retain its source attribution. This is an augmented derivative with 38 directory labels. The [original PlantVillage project](https://github.com/spMohanty/PlantVillage-Dataset) is a distinct source; do not assume the two releases have interchangeable file counts or splits.

The project brief describes approximately 79,000 training images. That is a user-supplied reference target, not a verified inventory. Release contents and your chosen subset determine the actual number: the index counts every successfully decoded file and saves counts per split and class. Likewise, the supplied 94.03% validation accuracy, 95.94% training accuracy, and training loss of 1.8891 → 0.1929 are reference results, not measurements reproduced by this repository.

## Layout

Extract the download and point `--data-dir` at the directory directly containing `train` and `valid`, rather than the outer Kaggle archive wrapper. Keep the exact class directory names, including spaces and punctuation.

```text
data/dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   └── image.jpg
│   └── ...37 other classes...
├── valid/
│   ├── Apple___Apple_scab/
│   │   └── another-image.jpg
│   └── ...37 other classes...
└── test/                         # optional labelled holdout
    ├── Apple___Apple_scab/
    │   └── held-out-image.jpg
    └── ...
```

`val/` or `validation/` can replace `valid/`; keep exactly one of them. Validation must contain the same class directories as training. A labelled test split may contain a subset of training classes. If the download also includes a flat, unlabelled test-image folder, keep that outside this root; it is useful for visual predictions but cannot measure class accuracy.

## Build the real image index

Install the package as explained in the README, then run:

```bash
python -m plant_disease.index --data-dir data/dataset --database data/index.sqlite --expected-classes 38
```

The SQLite database stores an absolute file path, the sorted class index, split, width, height, file size, and SHA-256 for each real image. It supports JPEG, PNG, BMP, TIFF, and WebP. Images remain on disk and are decoded during indexing to detect corrupt files. Class directories determine labels; `DEFAULT_CLASSES` in `plant_disease.labels` records the canonical 38-class roster for fixtures and reference. The expected-class check verifies the count; use the original source names to preserve label meaning.

Indexes on `(split, label_idx)` and `sha256` support training selection and duplicate checks. Inserts use batches to support large catalogues. The entire rebuild takes place in a temporary SQLite file; only a successful scan replaces the previous database. Moving a dataset invalidates its stored absolute paths, so rebuild the index after moving it.

Unreadable images stop indexing by default. To inspect and exclude them deliberately:

```bash
python -m plant_disease.index --data-dir data/dataset --database data/index.sqlite --skip-invalid
```

The printed and saved summary lists invalid paths and reasons. Every training and validation class still needs at least one valid image. Exact duplicate file hashes across splits always stop the rebuild. Duplicate files within one split are retained and counted in `same_split_duplicates`; remove redundant images if you want each photograph weighted only once.

## Avoid leakage before splitting

Hash checking detects byte-for-byte copies, but cannot detect a rotated, cropped, recoloured, resized, or re-encoded version of the same photograph. Group all variants of a source photograph, leaf, plant, capture session, and field **before** assigning groups to training, validation, and test. Augment only the training split after the split is fixed. Review any prepackaged augmented validation split for related source images before interpreting accuracy.

For a meaningful deployment evaluation, hold out real photographs from independent plants and capture conditions. Curated leaf backgrounds can differ sharply from phone photographs in a field. Validation results do not establish diagnostic reliability or support pesticide decisions on their own.

## Tiny synthetic smoke fixture

```bash
python scripts/create_smoke_dataset.py --output data/smoke --train-per-class 3 --valid-per-class 1
python -m plant_disease.index --data-dir data/smoke --database data/smoke.sqlite
```

This creates 152 small coloured-shape PNGs across all 38 canonical class names and writes `SYNTHETIC_DATASET.json`. They are deliberately distinct across splits, exercise the pipeline, and have no scientific disease-detection value. Default generation is deterministic; use a fresh output directory when changing image counts so old files are not mixed with a new fixture. `--allow-subset` is available on the index CLI for smaller custom fixtures.

## 100,000-row metadata capacity benchmark

```bash
python scripts/benchmark_index.py --rows 100000 --database work/metadata-benchmark.sqlite --output work/metadata-benchmark.json
```

This inserts at least 100,000 **synthetic metadata rows** into the same schema and reports insertion throughput, database size, a split/class query time, and its query plan. Paths, checksums, dimensions, and file sizes in this benchmark are fabricated; it creates and decodes zero image files. This checks catalogue capacity, not real-image I/O speed or model performance. The benchmark database is marked `synthetic_metadata: true` and must never be used for training. Choose a fresh database path for every run; the script refuses to overwrite an existing catalogue.

Keep downloads, image files, local SQLite databases, and trained checkpoints out of Git. Reproducible code and measured experiment reports belong in the repository; redistribution of datasets requires following their own current terms.
