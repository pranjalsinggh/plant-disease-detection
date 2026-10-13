"""Verify that ingestion catches corruption and prevents validation leakage."""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from plant_disease.index import DatasetError, build_index, get_summary, load_classes, load_samples


def test_index_round_trip_and_class_alignment(tiny_dataset, tmp_path):
    root, labels = tiny_dataset
    database = tmp_path / "images.sqlite"
    build_index(root, database, allow_subset=True)

    assert load_classes(database) == sorted(labels)
    summary = get_summary(database)
    assert summary["total_images"] == 12
    assert summary["splits"] == {"train": 6, "valid": 4, "test": 2}
    for split, count in (("train", 6), ("valid", 4), ("test", 2)):
        samples = load_samples(database, split)
        assert len(samples) == count
        for sample in samples:
            assert sample["split"] == split
            assert Path(sample["path"]).is_file()
            assert sample["label"] == load_classes(database)[sample["label_idx"]]


def test_full_dataset_class_count_is_checked(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    with pytest.raises(DatasetError):
        build_index(root, tmp_path / "images.sqlite")


def test_duplicate_image_across_splits_is_rejected(tiny_dataset, tmp_path):
    root, labels = tiny_dataset
    shutil.copyfile(root / "train" / labels[0] / "leaf-0.png", root / "valid" / labels[0] / "leak.png")
    with pytest.raises(DatasetError, match="(?i)(duplicate|leak)"):
        build_index(root, tmp_path / "images.sqlite", allow_subset=True)


def test_corrupt_image_does_not_replace_existing_index(tiny_dataset, tmp_path):
    root, labels = tiny_dataset
    database = tmp_path / "images.sqlite"
    build_index(root, database, allow_subset=True)
    previous = database.read_bytes()
    (root / "train" / labels[0] / "corrupt.png").write_bytes(b"this is not an image")
    with pytest.raises(DatasetError):
        build_index(root, database, allow_subset=True)
    assert database.read_bytes() == previous
    assert get_summary(database)["total_images"] == 12


def test_invalid_images_can_be_explicitly_skipped(tiny_dataset, tmp_path):
    root, labels = tiny_dataset
    (root / "train" / labels[0] / "corrupt.png").write_bytes(b"not a png")
    database = tmp_path / "images.sqlite"
    build_index(root, database, allow_subset=True, skip_invalid=True)
    assert get_summary(database)["total_images"] == 12


def test_validation_categories_must_match_training(tiny_dataset, tmp_path):
    root, labels = tiny_dataset
    shutil.move(str(root / "valid" / labels[1]), str(root / "valid" / "Unrecognized___category"))
    with pytest.raises(DatasetError):
        build_index(root, tmp_path / "images.sqlite", allow_subset=True)


def test_validation_alias_is_canonicalized(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    (root / "valid").rename(root / "validation")
    database = tmp_path / "images.sqlite"
    build_index(root, database, allow_subset=True)
    assert len(load_samples(database, "valid")) == 4
    assert all(sample["split"] == "valid" for sample in load_samples(database, "valid"))


def test_reading_an_index_does_not_lock_out_a_rebuild(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    database = tmp_path / "images.sqlite"
    build_index(root, database, allow_subset=True)
    load_samples(database, "train")
    load_classes(database)
    get_summary(database)
    build_index(root, database, allow_subset=True)
    assert get_summary(database)["total_images"] == 12


def test_metadata_only_benchmark_is_refused_as_training_samples(tiny_dataset, tmp_path):
    root, _ = tiny_dataset
    database = tmp_path / "images.sqlite"
    build_index(root, database, allow_subset=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE metadata SET value = 'true' WHERE key = 'synthetic_metadata'")
        connection.commit()
    assert get_summary(database)["synthetic_metadata"] is True
    with pytest.raises(DatasetError, match="(?i)metadata"):
        load_samples(database, "train")


def test_duplicate_within_split_is_visible_in_summary(tiny_dataset, tmp_path):
    root, labels = tiny_dataset
    shutil.copyfile(root / "train" / labels[0] / "leaf-0.png", root / "train" / labels[0] / "duplicate.png")
    summary = build_index(root, tmp_path / "images.sqlite", allow_subset=True)
    assert summary["total_images"] == 13
    assert summary["same_split_duplicates"] == 1
    assert summary["duplicates_by_split"]["train"] == 1
