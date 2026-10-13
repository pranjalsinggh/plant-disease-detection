"""Measure SQLite catalogue throughput with 100,000+ synthetic METADATA rows.

No image files are created or decoded. This measures metadata operations only,
not real dataset indexing time, CNN accuracy, or available image inventory.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from time import perf_counter

from plant_disease.index import _create_schema
from plant_disease.labels import DEFAULT_CLASSES


def benchmark(database: str | Path, *, rows: int = 100_000, batch_size: int = 1000) -> dict:
    if rows < 100_000 or batch_size < 1:
        raise ValueError("Use at least 100,000 metadata rows and a positive batch size.")
    destination = Path(database).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite an existing database: {destination}. Choose a fresh benchmark path.")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".benchmark-", suffix=".sqlite", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    classes = sorted(DEFAULT_CLASSES)
    started = perf_counter()
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            _create_schema(connection)
            connection.executemany("INSERT INTO classes(label_idx, label) VALUES (?, ?)", enumerate(classes))
            for offset in range(0, rows, batch_size):
                batch = [
                    (str(destination.parent / "METADATA_ONLY_NO_IMAGE_FILES" / f"record_{index:08d}.png"), index % len(classes), "train" if index % 5 else "valid", f"{index:064x}", 224, 224, 4096)
                    for index in range(offset, min(offset + batch_size, rows))
                ]
                connection.executemany("INSERT INTO images(path, label_idx, split, sha256, width, height, byte_size) VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            metadata = {
                "schema_version": 1,
                "synthetic_metadata": True,
                "dataset_root": None,
                "candidate_images": 0,
                "invalid_images": [],
                "warning": "Synthetic METADATA only. Paths, hashes, dimensions and byte sizes are fabricated. No image files exist.",
            }
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", [(key, json.dumps(value)) for key, value in metadata.items()])
            connection.commit()
            insert_seconds = perf_counter() - started
            query_started = perf_counter()
            queried_rows = connection.execute("SELECT COUNT(*) FROM images WHERE split = ? AND label_idx = ?", ("train", 0)).fetchone()[0]
            query_seconds = perf_counter() - query_started
            query_plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM images WHERE split = 'train' AND label_idx = 0")]
            actual_rows = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "benchmark_type": "synthetic_metadata_only",
        "warning": "No images were created or decoded; this is not a real 100,000-image dataset benchmark.",
        "database": str(destination),
        "metadata_rows": actual_rows,
        "classes": len(classes),
        "image_files_created": 0,
        "insert_seconds": round(insert_seconds, 4),
        "rows_per_second": round(actual_rows / insert_seconds, 1),
        "filtered_count_query_seconds": round(query_seconds, 6),
        "filtered_rows": queried_rows,
        "query_plan": query_plan,
        "database_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--database", type=Path, default=Path("work/metadata-benchmark.sqlite"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, help="Optional JSON benchmark results path.")
    args = parser.parse_args()
    try:
        result = benchmark(args.database, rows=args.rows, batch_size=args.batch_size)
    except (ValueError, sqlite3.Error) as error:
        parser.exit(2, f"Metadata benchmark failed: {error}\n")
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
