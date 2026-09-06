"""Tests for the demo app's checkpoint resolution and inference wiring.

The app is an optional extra, so everything here skips cleanly when gradio is
absent -- which is the case in the default CI environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio", reason="app.py needs the [app] extra")

APP = Path(__file__).resolve().parents[1] / "app.py"


def _load_app():
    """Import app.py, which sits at the repository root rather than in src/."""
    spec = importlib.util.spec_from_file_location("food_app", APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["food_app"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load_app()


def test_resolve_checkpoint_accepts_an_existing_path(app, tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not a real checkpoint")

    assert app._resolve_checkpoint(str(checkpoint), None) == checkpoint


def test_resolve_checkpoint_rejects_a_missing_path(app, tmp_path: Path):
    """Fail at startup with the path, rather than deep inside torch.load."""
    with pytest.raises(SystemExit, match="checkpoint not found"):
        app._resolve_checkpoint(str(tmp_path / "absent.pt"), None)


def test_resolve_checkpoint_requires_a_source(app):
    """No path and no repo is a configuration error worth naming."""
    with pytest.raises(SystemExit, match="no checkpoint given"):
        app._resolve_checkpoint(None, None)


def test_interface_serves_predictions_for_a_real_checkpoint(app, tmp_path: Path):
    """End-to-end: train briefly, then drive the same callback the UI button uses."""
    pytest.importorskip("torch")
    from PIL import Image

    from food_recognition import TrainingConfig, train_model

    # A tiny two-class problem: solid red vs solid blue, 32px, 1 epoch.
    for split in ("train", "val"):
        for index, colour in enumerate([(220, 30, 30), (30, 30, 220)]):
            folder = tmp_path / split / f"{index:02d}"
            folder.mkdir(parents=True)
            for n in range(4):
                Image.new("RGB", (40, 40), colour).save(folder / f"{n}.png")

    cfg = TrainingConfig(
        model_name="simple_cnn",
        num_classes=2,
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        image_size=32,
        epochs=1,
        batch_size=4,
        device="cpu",
        output_dir=str(tmp_path / "run"),
    )
    train_model(cfg)

    checkpoint = tmp_path / "run" / "checkpoints" / "best.pt"
    assert checkpoint.is_file()

    demo = app.build_interface(checkpoint, device="cpu")
    classify = next(
        entry.fn for entry in demo.fns.values() if entry.fn.__name__ == "classify"
    )

    scores, overlay = classify(Image.new("RGB", (40, 40), (220, 30, 30)), 2, True)

    assert set(scores) <= {"00", "01"}
    assert len(scores) == 2
    assert abs(sum(scores.values()) - 1.0) < 1e-3
    # Grad-CAM requested, so an overlay must come back at the model's resolution.
    assert overlay is not None
    assert overlay.size == (32, 32)

    # Toggling the explanation off must skip the backward pass, not return a blank.
    scores_only, no_overlay = classify(Image.new("RGB", (40, 40), (30, 30, 220)), 1, False)
    assert len(scores_only) == 1
    assert no_overlay is None

    # An empty upload is a no-op rather than a traceback in the UI.
    assert classify(None, 3, True) == ({}, None)
