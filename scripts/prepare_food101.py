#!/usr/bin/env python
"""Convert the official Food-101 release into this project's directory layout.

Food-101 ships as a flat class tree plus text files listing the split members:

    food-101/
    ├── images/<class>/<hash>.jpg      # all 101,000 images together
    └── meta/
        ├── train.txt                  # 75,750 lines: "apple_pie/1005649"
        └── test.txt                   # 25,250 lines

The training pipeline expects each split in its own directory tree, so this
script materialises:

    <output>/
    ├── training/labeled/<class>/...   # from meta/train.txt
    └── validation/<class>/...         # from meta/test.txt

Symlinks are used by default, so this costs almost no disk space and takes
seconds instead of copying ~5GB. Use --copy for real files (e.g. when the
output must survive the source being deleted, or land on another volume).

    python scripts/prepare_food101.py --source ~/data/food-101 --output data/food-101

Download the archive from:
    https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

EXPECTED_CLASSES = 101
EXPECTED_TRAIN = 75_750
EXPECTED_TEST = 25_250

SPLITS = {"train": "training/labeled", "test": "validation"}


def _read_split(meta_file: Path) -> dict[str, list[str]]:
    """Parse a meta file into ``{class_name: [image_stem, ...]}``."""
    if not meta_file.exists():
        raise FileNotFoundError(f"meta file not found: {meta_file}")

    per_class: dict[str, list[str]] = {}
    for lineno, raw in enumerate(
        meta_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        entry = raw.strip()
        if not entry:
            continue
        if "/" not in entry:
            raise ValueError(
                f"{meta_file}:{lineno}: expected '<class>/<image>', got {entry!r}"
            )
        class_name, stem = entry.rsplit("/", 1)
        per_class.setdefault(class_name, []).append(stem)

    if not per_class:
        raise ValueError(f"{meta_file} contained no entries")
    return per_class


def _link_or_copy(source: Path, target: Path, *, copy: bool, relative: bool) -> None:
    """Materialise ``target`` from ``source``, skipping work already done."""
    if target.exists() or target.is_symlink():
        return

    if copy:
        shutil.copy2(source, target)
        return

    # Relative symlinks keep the tree valid if the whole thing is moved.
    link_target = (
        Path(os.path.relpath(source, target.parent)) if relative else source
    )
    try:
        target.symlink_to(link_target)
    except OSError as exc:
        raise OSError(
            f"failed to create symlink {target} -> {link_target}: {exc}. "
            "Re-run with --copy if this filesystem disallows symlinks."
        ) from exc


def prepare(
    source: Path,
    output: Path,
    *,
    copy: bool = False,
    relative: bool = True,
    limit_per_class: int | None = None,
) -> dict[str, int]:
    """Build the split trees. Returns ``{split: image_count}``."""
    images_dir = source / "images"
    meta_dir = source / "meta"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"{images_dir} not found. --source must point at the extracted "
            "food-101 directory containing images/ and meta/."
        )
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"{meta_dir} not found (expected alongside images/)")

    counts: dict[str, int] = {}
    missing: list[str] = []

    for split, relative_dest in SPLITS.items():
        per_class = _read_split(meta_dir / f"{split}.txt")
        dest_root = output / relative_dest
        written = 0

        for class_name, stems in sorted(per_class.items()):
            class_dir = dest_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            selected = stems[:limit_per_class] if limit_per_class else stems
            for stem in selected:
                src = images_dir / class_name / f"{stem}.jpg"
                if not src.exists():
                    missing.append(f"{class_name}/{stem}.jpg")
                    continue
                _link_or_copy(
                    src, class_dir / f"{stem}.jpg", copy=copy, relative=relative
                )
                written += 1

        counts[split] = written
        print(
            f"  {split:5s} -> {dest_root}  "
            f"({len(per_class)} classes, {written} images)"
        )

    if missing:
        print(
            f"\nwarning: {len(missing)} image(s) listed in meta/ were missing "
            f"from images/, e.g. {missing[:3]}",
            file=sys.stderr,
        )

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted food-101 directory (contains images/ and meta/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/food-101"),
        help="Where to write the training/validation trees.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (slower, uses ~5GB).",
    )
    parser.add_argument(
        "--absolute-links",
        action="store_true",
        help="Use absolute symlink targets instead of relative ones.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Only take N images per class per split (for quick experiments).",
    )
    args = parser.parse_args()

    print(f"source: {args.source}")
    print(f"output: {args.output}")
    print(f"mode  : {'copy' if args.copy else 'symlink'}\n")

    try:
        counts = prepare(
            args.source,
            args.output,
            copy=args.copy,
            relative=not args.absolute_links,
            limit_per_class=args.limit_per_class,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Only assert official totals for a full, unlimited conversion.
    if args.limit_per_class is None:
        if counts.get("train") != EXPECTED_TRAIN or counts.get("test") != EXPECTED_TEST:
            print(
                f"\nwarning: expected {EXPECTED_TRAIN} train / {EXPECTED_TEST} test "
                f"images but wrote {counts.get('train')} / {counts.get('test')}. "
                "The source may be incomplete.",
                file=sys.stderr,
            )
        else:
            print(f"\nOK: {EXPECTED_TRAIN} train + {EXPECTED_TEST} test images")

    print("\nNext:")
    print("  food-recognition-train --config configs/food101_efficientnet_cbam.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
