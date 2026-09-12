"""Tests for scripts/dump_predictions.py.

Runs the real script against a real (tiny) checkpoint on a real image directory,
so a broken correctness vector cannot pass. Per CONTRIBUTING.md this exercises
the actual behaviour rather than mocking the model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from food_recognition.models import initialize_model
from food_recognition.utils import save_checkpoint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dump_predictions.py"


def _load():
    spec = importlib.util.spec_from_file_location("dump_predictions", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dump = _load()


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    model, _ = initialize_model("resnet18", 3, use_pretrained=False)
    path = tmp_path / "ck" / "best.pt"
    save_checkpoint(
        path,
        model,
        epoch=1,
        metrics={"val_acc": 0.5},
        config={"model_name": "resnet18", "image_size": 32, "num_classes": 3},
        classes=["00", "01", "02"],
    )
    return path


def _dump(tmp_path: Path, checkpoint: Path, data_dir: Path, name: str = "v.json") -> dict:
    out = tmp_path / name
    rc = dump.main(
        [
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_dir),
            "--json-out", str(out),
            "--device", "cpu",
            "--num-workers", "0",
            "--batch-size", "4",
        ]
    )
    assert rc == 0
    return json.loads(out.read_text())


def test_dump_writes_one_entry_per_image(
    tmp_path: Path, checkpoint: Path, sample_dataset: Path
) -> None:
    val = sample_dataset / "validation"
    payload = _dump(tmp_path, checkpoint, val)
    # 3 classes x 3 validation images.
    assert payload["n_images"] == 9
    assert len(payload["correct"]) == 9


def test_dump_vector_is_binary_and_consistent_with_accuracy(
    tmp_path: Path, checkpoint: Path, sample_dataset: Path
) -> None:
    payload = _dump(tmp_path, checkpoint, sample_dataset / "validation")
    assert set(payload["correct"]) <= {0, 1}
    assert payload["n_correct"] == sum(payload["correct"])
    assert payload["accuracy"] == pytest.approx(payload["n_correct"] / payload["n_images"])


def test_dump_is_deterministic_across_invocations(
    tmp_path: Path, checkpoint: Path, sample_dataset: Path
) -> None:
    # Two dumps of the same checkpoint must agree exactly, otherwise pairing two
    # different checkpoints' vectors would be meaningless.
    first = _dump(tmp_path, checkpoint, sample_dataset / "validation", "a.json")
    second = _dump(tmp_path, checkpoint, sample_dataset / "validation", "b.json")
    assert first["correct"] == second["correct"]
    assert first["order_sha256_16"] == second["order_sha256_16"]


def test_dump_order_hash_differs_between_different_directories(
    tmp_path: Path, checkpoint: Path, sample_dataset: Path, nonpadded_dataset: Path
) -> None:
    # The hash exists so that pairing vectors from different sets fails loudly.
    a = _dump(tmp_path, checkpoint, sample_dataset / "validation", "a.json")
    b = _dump(tmp_path, checkpoint, nonpadded_dataset, "b.json")
    assert a["order_sha256_16"] != b["order_sha256_16"]


def test_dump_records_provenance(
    tmp_path: Path, checkpoint: Path, sample_dataset: Path
) -> None:
    payload = _dump(tmp_path, checkpoint, sample_dataset / "validation")
    assert payload["checkpoint"] == str(checkpoint)
    assert Path(payload["data_dir"]).name == "validation"
    assert payload["image_size"] == 32


def test_dump_agrees_with_a_direct_forward_pass(
    tmp_path: Path, checkpoint: Path, sample_dataset: Path
) -> None:
    """The vector must match what the model actually predicts, image by image."""
    from torch.utils.data import DataLoader

    from food_recognition.data import LabeledImageDataset, build_transform
    from food_recognition.predict import load_predictor

    val = sample_dataset / "validation"
    payload = _dump(tmp_path, checkpoint, val)

    predictor = load_predictor(checkpoint, device="cpu")
    dataset = LabeledImageDataset(
        val, transform=build_transform(predictor.image_size, is_train=False)
    )
    expected: list[int] = []
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=4, shuffle=False):
            preds = predictor.model(images).argmax(dim=1)
            expected.extend((preds == labels).to(torch.int64).tolist())

    assert payload["correct"] == expected
