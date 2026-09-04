"""Shared pytest fixtures."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

CLASS_COLOURS = [(220, 60, 60), (60, 180, 90), (70, 110, 220)]


def _write_image(path: Path, rgb, size: int, rng: random.Random) -> None:
    jitter = 30
    data = [
        tuple(max(0, min(255, c + rng.randint(-jitter, jitter))) for c in rgb)
        for _ in range(size * size)
    ]
    image = Image.new("RGB", (size, size))
    image.putdata(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """A 3-class dataset with labelled train/val plus a flat unlabelled dir."""
    rng = random.Random(0)
    root = tmp_path / "data"

    for index, colour in enumerate(CLASS_COLOURS):
        name = f"{index:02d}"
        for split, count in (("training/labeled", 6), ("validation", 3)):
            for i in range(count):
                _write_image(root / split / name / f"{i}.jpg", colour, 64, rng)

    for i in range(4):
        colour = CLASS_COLOURS[i % len(CLASS_COLOURS)]
        _write_image(root / "training/unlabeled" / f"u{i}.jpg", colour, 64, rng)

    return root


@pytest.fixture
def nonpadded_dataset(tmp_path: Path) -> Path:
    """Class dirs named 0..10 without zero padding, to test numeric ordering."""
    rng = random.Random(1)
    root = tmp_path / "nonpadded"
    for index in range(11):
        _write_image(
            root / str(index) / "a.jpg",
            CLASS_COLOURS[index % len(CLASS_COLOURS)],
            32,
            rng,
        )
    return root
