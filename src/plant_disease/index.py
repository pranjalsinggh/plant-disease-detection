"""Validate an image dataset and build a queryable SQLite catalogue.

Images remain on disk; the catalogue stores absolute paths and image metadata.
Rebuilding uses a temporary database, so a failed scan preserves the old index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Iterator

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


class DatasetError(ValueError):
    """The requested dataset cannot be indexed safely."""


def _resolve_splits(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {root}")
    if not (root / "train").is_dir():
        raise DatasetError(f"Missing train/ directory under {root}")
    validation = [root / name for name in ("valid", "val", "validation") if (root / name).is_dir()]
    if len(validation) != 1:
        raise DatasetError("Use exactly one validation directory: valid/, val/, or validation/.")
    result = {"train": root / "train", "valid": validation[0]}
    if (root / "test").is_dir():
        result["test"] = root / "test"
    return result


def _image_files(root: Path) -> Iterator[Path]:
    # Sorting makes a scan reproducible; only one directory is held at a time.
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs.sort()
        for filename in sorted(files):
            path = Path(directory) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE classes (
            label_idx INTEGER PRIMARY KEY,
            label TEXT NOT NULL UNIQUE
        );
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            label_idx INTEGER NOT NULL REFERENCES classes(label_idx),
            split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
            sha256 TEXT NOT NULL,
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0),
            byte_size INTEGER NOT NULL CHECK (byte_size > 0)
        );
        CREATE INDEX images_split_class ON images(split, label_idx);
        CREATE INDEX images_sha256 ON images(sha256);
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )


def build_index(
    data_dir: str | Path,
    database: str | Path,
    *,
    expected_classes: int = 38,
    allow_subset: bool = False,
    skip_invalid: bool = False,
    batch_size: int = 500,
) -> dict:
    """Decode real image files and atomically replace the SQLite catalogue.

    ``allow_subset`` disables the class-count check for development fixtures.
    Directory labels determine the sorted mapping; they are never inferred from
    a filename. Exact duplicate file hashes across splits are always rejected.
    """
    if expected_classes < 1 or batch_size < 1:
        raise DatasetError("expected_classes and batch_size must be positive integers.")
    root = Path(data_dir).expanduser().resolve()
    destination = Path(database).expanduser().resolve()
    splits = _resolve_splits(root)
    labels = sorted(path.name for path in splits["train"].iterdir() if path.is_dir() and not path.name.startswith("."))
    if not labels:
        raise DatasetError("No class directories found in train/.")
    if not allow_subset and len(labels) != expected_classes:
        raise DatasetError(f"Expected {expected_classes} classes, found {len(labels)}. Use --allow-subset for a small fixture.")
    label_to_idx = {label: index for index, label in enumerate(labels)}
    for split, split_dir in splits.items():
        unlabelled = [path.name for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if unlabelled:
            raise DatasetError(f"{split}/ contains unlabelled images; place images inside class directories.")
        split_labels = {path.name for path in split_dir.iterdir() if path.is_dir() and not path.name.startswith(".")}
        unknown = split_labels - set(labels)
        if unknown:
            raise DatasetError(f"Unknown classes in {split}/: {', '.join(sorted(unknown))}")
        if split != "test" and split_labels != set(labels):
            raise DatasetError(f"{split}/ must contain every training class; missing: {', '.join(sorted(set(labels) - split_labels))}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    invalid: list[dict[str, str]] = []
    candidate_images = 0
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            _create_schema(connection)
            connection.executemany("INSERT INTO classes(label_idx, label) VALUES (?, ?)", enumerate(labels))
            batch: list[tuple] = []
            for split, split_dir in splits.items():
                for label in labels:
                    class_dir = split_dir / label
                    if not class_dir.is_dir():
                        continue
                    for image_path in _image_files(class_dir):
                        candidate_images += 1
                        resolved = image_path.resolve()
                        if resolved == destination:
                            raise DatasetError("The database destination must not overwrite a dataset image.")
                        try:
                            with Image.open(resolved) as decoded:
                                decoded.load()
                                width, height = decoded.size
                            size = resolved.stat().st_size
                            checksum = _sha256(resolved)
                        except (OSError, ValueError, Image.DecompressionBombError) as error:
                            if not skip_invalid:
                                raise DatasetError(f"Invalid image {resolved}: {error}") from error
                            invalid.append({"path": str(resolved), "error": str(error)})
                            continue
                        batch.append((str(resolved), label_to_idx[label], split, checksum, width, height, size))
                        if len(batch) >= batch_size:
                            connection.executemany("INSERT INTO images(path, label_idx, split, sha256, width, height, byte_size) VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                            batch.clear()
            if batch:
                connection.executemany("INSERT INTO images(path, label_idx, split, sha256, width, height, byte_size) VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            leakage = connection.execute(
                "SELECT sha256, GROUP_CONCAT(DISTINCT split) FROM images GROUP BY sha256 HAVING COUNT(DISTINCT split) > 1 LIMIT 1"
            ).fetchone()
            if leakage:
                example = connection.execute("SELECT path FROM images WHERE sha256 = ? LIMIT 1", (leakage[0],)).fetchone()[0]
                raise DatasetError(f"Cross-split duplicate detected ({leakage[1]}): {example}. Remove duplicate copies before indexing.")
            for split in ("train", "valid"):
                empty_classes = [
                    label for label in labels
                    if connection.execute("SELECT 1 FROM images WHERE split = ? AND label_idx = ? LIMIT 1", (split, label_to_idx[label])).fetchone() is None
                ]
                if empty_classes:
                    raise DatasetError(f"No valid images for classes in {split}/: {', '.join(empty_classes)}")
            metadata = {
                "schema_version": 1,
                "dataset_root": str(root),
                "candidate_images": candidate_images,
                "invalid_images": invalid,
                "synthetic_metadata": False,
                "synthetic_images": (root / "SYNTHETIC_DATASET.json").is_file(),
            }
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", [(key, json.dumps(value)) for key, value in metadata.items()])
            connection.commit()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return get_summary(destination)


def _read_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise DatasetError(f"Index database does not exist: {path}. Run python -m plant_disease.index first.")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_classes(db_path: str | Path) -> list[str]:
    """Return the deterministic sorted class mapping saved in the database."""
    with closing(_read_connection(db_path)) as connection:
        return [row["label"] for row in connection.execute("SELECT label FROM classes ORDER BY label_idx")]


def load_samples(db_path: str | Path, split: str) -> list[dict]:
    """Load one split as path/label records suitable for a torch Dataset."""
    split = {"val": "valid", "validation": "valid"}.get(split, split)
    if split not in SPLITS:
        raise DatasetError(f"Unknown split {split!r}; choose train, valid, or test.")
    with closing(_read_connection(db_path)) as connection:
        marker = connection.execute("SELECT value FROM metadata WHERE key = 'synthetic_metadata'").fetchone()
        if marker and json.loads(marker["value"]):
            raise DatasetError("This database contains synthetic metadata only, with no image files. Build a real image index before loading samples.")
        return [dict(row) for row in connection.execute(
            "SELECT images.path, classes.label, images.label_idx, images.split FROM images JOIN classes USING(label_idx) WHERE split = ? ORDER BY images.id",
            (split,),
        )]


def get_summary(db_path: str | Path) -> dict:
    """Summarize actual indexed rows; missing optional test data has count zero."""
    with closing(_read_connection(db_path)) as connection:
        classes = [row["label"] for row in connection.execute("SELECT label FROM classes ORDER BY label_idx")]
        counts = {split: 0 for split in SPLITS}
        for row in connection.execute("SELECT split, COUNT(*) AS count FROM images GROUP BY split"):
            counts[row["split"]] = row["count"]
        duplicate_counts = {split: 0 for split in SPLITS}
        for row in connection.execute("SELECT split, COUNT(*) - COUNT(DISTINCT sha256) AS count FROM images GROUP BY split"):
            duplicate_counts[row["split"]] = row["count"]
        metadata = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT key, value FROM metadata")}
        per_class = {label: {split: 0 for split in SPLITS} for label in classes}
        for row in connection.execute("SELECT label, split, COUNT(*) AS count FROM images JOIN classes USING(label_idx) GROUP BY label, split"):
            per_class[row["label"]][row["split"]] = row["count"]
    return {
        "database": str(Path(db_path).expanduser().resolve()),
        "total_images": sum(counts.values()),
        "class_count": len(classes),
        "classes": classes,
        "splits": counts,
        "per_class": per_class,
        "same_split_duplicates": sum(duplicate_counts.values()),
        "duplicates_by_split": duplicate_counts,
        "invalid_count": len(metadata.get("invalid_images", [])),
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--database", type=Path, default=Path("data/index.sqlite"))
    parser.add_argument("--expected-classes", type=int, default=38)
    parser.add_argument("--allow-subset", action="store_true", help="Disable expected class count for development fixtures.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip unreadable images and list them in the saved summary.")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    try:
        summary = build_index(args.data_dir, args.database, expected_classes=args.expected_classes, allow_subset=args.allow_subset, skip_invalid=args.skip_invalid, batch_size=args.batch_size)
    except (DatasetError, sqlite3.Error) as error:
        parser.exit(2, f"Dataset indexing failed: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
