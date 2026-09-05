#!/usr/bin/env python
"""Generate a small synthetic dataset for smoke tests and CI.

Creates the exact directory layout the training pipeline expects, including a
flat ``training/unlabeled`` directory, so the semi-supervised path is exercised
without downloading anything.

Each class gets a distinct base hue plus noise, so a model can actually learn
the task and accuracy is a meaningful signal rather than luck.

    python scripts/make_sample_data.py --output data/sample
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

# Visually separable base colours, one per class.
CLASS_COLOURS = [
    (220, 60, 60),
    (60, 180, 90),
    (70, 110, 220),
    (240, 200, 70),
    (170, 90, 200),
    (60, 200, 200),
    (240, 140, 60),
    (140, 140, 140),
    (200, 80, 150),
    (110, 200, 60),
    (80, 80, 190),
]


def _make_image(rgb: tuple[int, int, int], size: int, rng: random.Random) -> Image.Image:
    """Build a noisy solid-colour image around ``rgb``."""
    jitter = 45
    pixels = []
    for _ in range(size * size):
        pixels.append(
            tuple(
                max(0, min(255, channel + rng.randint(-jitter, jitter)))
                for channel in rgb
            )
        )
    image = Image.new("RGB", (size, size))
    image.putdata(pixels)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sample"))
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--val-per-class", type=int, default=4)
    parser.add_argument("--unlabeled", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_classes > len(CLASS_COLOURS):
        parser.error(f"--num-classes must be <= {len(CLASS_COLOURS)}")

    rng = random.Random(args.seed)
    root = args.output

    for class_index in range(args.num_classes):
        colour = CLASS_COLOURS[class_index]
        name = f"{class_index:02d}"

        for split, count in (
            ("training/labeled", args.train_per_class),
            ("validation", args.val_per_class),
        ):
            directory = root / split / name
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                _make_image(colour, args.image_size, rng).save(directory / f"{i:03d}.jpg")

    # Flat directory with no class sub-folders - the layout that broke
    # ImageFolder in the original implementation.
    unlabeled = root / "training/unlabeled"
    unlabeled.mkdir(parents=True, exist_ok=True)
    for i in range(args.unlabeled):
        colour = CLASS_COLOURS[i % args.num_classes]
        _make_image(colour, args.image_size, rng).save(unlabeled / f"u{i:03d}.jpg")

    total = args.num_classes * (args.train_per_class + args.val_per_class) + args.unlabeled
    print(f"created {total} images under {root}")
    print(f"  train     : {root / 'training/labeled'} "
          f"({args.num_classes} classes x {args.train_per_class})")
    print(f"  val       : {root / 'validation'} "
          f"({args.num_classes} classes x {args.val_per_class})")
    print(f"  unlabeled : {unlabeled} ({args.unlabeled} images, flat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
