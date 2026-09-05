#!/usr/bin/env python
"""Rebuild the official Food-101 layout from the Hugging Face Parquet mirror.

The canonical archive at data.vision.ee.ethz.ch served ~0.25 MB/s when this was
written — over five hours for one 4.7 GB file. The `ethz/food101` mirror on
Hugging Face holds the same images and the same official 75,750 / 25,250 split,
and downloads in a couple of minutes.

The catch is that the mirror ships Parquet, not the `images/` + `meta/` tree
that `prepare_food101.py` expects. This script bridges the two, writing:

    <output>/
    ├── images/<class>/<stem>.jpg   # all 101,000 images
    └── meta/
        ├── train.txt              # 75,750 lines: "apple_pie/1005649"
        ├── test.txt               # 25,250 lines
        └── classes.txt            # 101 class names

The official split survives the round trip because the mirror preserves each
original filename in the Parquet `image.path` field, so the `<class>/<stem>`
identity that meta/train.txt is built from is fully recoverable. Class names
come from the Parquet schema metadata, so the label ordering matches upstream
rather than being inferred.

Typical use:

    pip install -e ".[food101]"
    python scripts/food101_from_parquet.py --output ~/data/food-101
    python scripts/prepare_food101.py --source ~/data/food-101 --output data/food-101

Pass --parquet-dir to reuse shards already on disk instead of downloading.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "ethz/food101"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/data"
TRAIN_SHARDS = 8
VAL_SHARDS = 3

EXPECTED_CLASSES = 101
EXPECTED_TRAIN = 75_750
EXPECTED_TEST = 25_250
EXPECTED_PER_CLASS = {"train": 750, "test": 250}

# The mirror calls the held-out split "validation"; Food-101 calls it "test".
# prepare_food101.py reads meta/test.txt, so we write the upstream name.
SPLITS = (("train", "train", TRAIN_SHARDS), ("test", "validation", VAL_SHARDS))


def _shard_names(prefix: str, total: int) -> list[str]:
    return [f"{prefix}-{i:05d}-of-{total:05d}.parquet" for i in range(total)]


def _download_one(name: str, dest: Path) -> str:
    """Fetch a single shard unless it is already complete on disk."""
    url = f"{BASE_URL}/{name}"
    target = dest / name
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            expected = int(response.headers.get("Content-Length") or 0)
            if target.exists() and expected and target.stat().st_size == expected:
                return f"  {name}: already complete, skipped"
            with open(target, "wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {name}: {exc}") from exc

    size = target.stat().st_size
    if expected and size != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: got {size} bytes, expected {expected}")
    return f"  {name}: {size / 1e6:.0f} MB"


def download_shards(dest: Path, *, jobs: int = 6) -> None:
    """Download every shard, in parallel — the CDN is much faster that way."""
    dest.mkdir(parents=True, exist_ok=True)
    names = _shard_names("train", TRAIN_SHARDS) + _shard_names(
        "validation", VAL_SHARDS
    )
    print(f"downloading {len(names)} shards from {REPO} into {dest}")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for line in pool.map(lambda n: _download_one(n, dest), names):
            print(line, flush=True)


def _class_names(shard: Path) -> list[str]:
    """Read the label ordering from the Parquet schema metadata."""
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(shard).schema_arrow.metadata or {}
    blob = metadata.get(b"huggingface")
    if not blob:
        raise ValueError(f"{shard} has no huggingface schema metadata")
    names = json.loads(blob.decode())["info"]["features"]["label"]["names"]
    if not names:
        raise ValueError(f"{shard} lists no class names")
    if len(names) != EXPECTED_CLASSES:
        # Not fatal: the count is reported again per split, and a genuine
        # mismatch shows up there as a wrong per-class total.
        print(
            f"warning: expected {EXPECTED_CLASSES} classes, found {len(names)}",
            file=sys.stderr,
        )
    return names


def convert(parquet_dir: Path, output: Path) -> dict[str, int]:
    """Unpack shards into images/ + meta/. Returns ``{split: count}``."""
    import pyarrow.parquet as pq

    images_dir = output / "images"
    meta_dir = output / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    first = parquet_dir / _shard_names("validation", VAL_SHARDS)[0]
    if not first.exists():
        raise FileNotFoundError(
            f"{first} not found. Run without --parquet-dir to download the "
            "shards, or point --parquet-dir at a directory that has them."
        )
    names = _class_names(first)
    (meta_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for name in names:
        (images_dir / name).mkdir(exist_ok=True)

    counts: dict[str, int] = {}
    for split, prefix, total in SPLITS:
        entries: list[str] = []
        per_class: collections.Counter[str] = collections.Counter()
        collisions = 0

        for shard_name in _shard_names(prefix, total):
            shard = parquet_dir / shard_name
            if not shard.exists():
                raise FileNotFoundError(f"missing shard: {shard}")
            handle = pq.ParquetFile(shard)
            for batch in handle.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    label = names[row["label"]]
                    stem = Path(row["image"]["path"]).stem
                    destination = images_dir / label / f"{stem}.jpg"
                    if destination.exists():
                        collisions += 1
                    destination.write_bytes(row["image"]["bytes"])
                    entries.append(f"{label}/{stem}")
                    per_class[label] += 1
            print(f"  {shard_name}: {len(entries)} images so far", flush=True)

        (meta_dir / f"{split}.txt").write_text(
            "\n".join(entries) + "\n", encoding="utf-8"
        )
        counts[split] = len(entries)

        low, high = min(per_class.values()), max(per_class.values())
        want = EXPECTED_PER_CLASS[split]
        print(
            f"{split}: {len(entries)} images, {len(per_class)} classes, "
            f"per-class {low}-{high}"
        )
        if collisions:
            print(
                f"warning: {collisions} filename collision(s) in {split}; "
                "images were overwritten and the split may be short",
                file=sys.stderr,
            )
        if low != want or high != want:
            print(
                f"warning: {split} should hold {want} images per class",
                file=sys.stderr,
            )

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write images/ and meta/ (the food-101 root).",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        help="Reuse shards from here instead of downloading.",
    )
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep downloaded shards (~4.8 GB) instead of deleting them.",
    )
    parser.add_argument(
        "--jobs", type=int, default=6, help="Parallel downloads (default: 6)."
    )
    args = parser.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print(
            'error: pyarrow is required. Install it with: pip install -e ".[food101]"',
            file=sys.stderr,
        )
        return 1

    downloaded_here = args.parquet_dir is None
    parquet_dir = args.parquet_dir or (args.output / "_parquet")

    try:
        if downloaded_here:
            download_shards(parquet_dir, jobs=args.jobs)
        print(f"\nconverting into {args.output}")
        counts = convert(parquet_dir, args.output)
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if counts.get("train") == EXPECTED_TRAIN and counts.get("test") == EXPECTED_TEST:
        print(f"\nOK: {EXPECTED_TRAIN} train + {EXPECTED_TEST} test images")
    else:
        print(
            f"\nwarning: expected {EXPECTED_TRAIN}/{EXPECTED_TEST} images but got "
            f"{counts.get('train')}/{counts.get('test')}",
            file=sys.stderr,
        )

    if downloaded_here and not args.keep_parquet:
        for shard in parquet_dir.glob("*.parquet"):
            shard.unlink()
        parquet_dir.rmdir()
        print(f"removed shards from {parquet_dir} (--keep-parquet to retain)")

    print("\nNext:")
    print(
        f"  python scripts/prepare_food101.py --source {args.output} "
        "--output data/food-101"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
