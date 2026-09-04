#!/usr/bin/env python
"""Download the Food-11 dataset from Kaggle via kagglehub.

Requires the optional download extra and Kaggle credentials:

    pip install -e ".[download]"
    python scripts/download_dataset.py

kagglehub reads credentials from ~/.kaggle/kaggle.json or the KAGGLE_USERNAME /
KAGGLE_KEY environment variables. The download lands in kagglehub's cache; this
script prints the path so you can point a config's train_dir/val_dir at it or
symlink it under data/.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_DATASET = "zhaopang/ml2021springhw3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug to download.",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError:
        print(
            "kagglehub is not installed. Install the optional extra:\n"
            '    pip install -e ".[download]"',
            file=sys.stderr,
        )
        return 1

    try:
        path = kagglehub.dataset_download(args.dataset)
    except Exception as exc:  # noqa: BLE001 - surface any kagglehub/auth error
        print(f"download failed: {exc}", file=sys.stderr)
        print(
            "\nCheck that Kaggle credentials are configured "
            "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).",
            file=sys.stderr,
        )
        return 1

    print(f"dataset downloaded to: {path}")
    print(
        "\nPoint your config at the extracted food-11 directories, e.g.\n"
        "  train_dir: <path>/food-11/training/labeled\n"
        "  val_dir:   <path>/food-11/validation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
