"""Tests for the Food-101 preparation script.

The real dataset is 5GB, so these build a miniature copy of the *official*
layout (images/<class>/<hash>.jpg + meta/train.txt + meta/test.txt) and verify
the conversion against it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_food101.py"


def _load_script():
    """Import prepare_food101.py, which lives outside the installed package."""
    spec = importlib.util.spec_from_file_location("prepare_food101", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_food101"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture
def fake_food101(tmp_path: Path) -> Path:
    """A miniature Food-101 in the official on-disk layout."""
    root = tmp_path / "food-101"
    images = root / "images"
    meta = root / "meta"
    meta.mkdir(parents=True)

    classes = ["apple_pie", "baby_back_ribs", "waffles"]
    train_lines: list[str] = []
    test_lines: list[str] = []

    for class_index, name in enumerate(classes):
        class_dir = images / name
        class_dir.mkdir(parents=True)
        for i in range(5):
            stem = f"{class_index}{i:04d}"
            Image.new("RGB", (16, 16), (class_index * 60, 100, 150)).save(
                class_dir / f"{stem}.jpg"
            )
            (train_lines if i < 3 else test_lines).append(f"{name}/{stem}")

    (meta / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (meta / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    return root


def test_creates_expected_layout(script, fake_food101: Path, tmp_path: Path):
    output = tmp_path / "out"
    counts = script.prepare(fake_food101, output)

    assert counts == {"train": 9, "test": 6}  # 3 classes x 3 / x 2

    train_root = output / "training/labeled"
    val_root = output / "validation"
    assert sorted(p.name for p in train_root.iterdir()) == [
        "apple_pie",
        "baby_back_ribs",
        "waffles",
    ]
    assert len(list((train_root / "apple_pie").glob("*.jpg"))) == 3
    assert len(list((val_root / "apple_pie").glob("*.jpg"))) == 2


def test_split_membership_follows_meta_files(script, fake_food101: Path, tmp_path: Path):
    """No image may leak from the test split into training."""
    output = tmp_path / "out"
    script.prepare(fake_food101, output)

    train_stems = {
        p.stem for p in (output / "training/labeled").rglob("*.jpg")
    }
    val_stems = {p.stem for p in (output / "validation").rglob("*.jpg")}

    assert not (train_stems & val_stems), "train/val overlap - split is leaking"

    expected_train = {
        line.split("/")[1]
        for line in (fake_food101 / "meta/train.txt")
        .read_text()
        .strip()
        .splitlines()
    }
    assert train_stems == expected_train


def test_symlinks_by_default_and_resolve(script, fake_food101: Path, tmp_path: Path):
    output = tmp_path / "out"
    script.prepare(fake_food101, output)

    linked = next((output / "training/labeled/apple_pie").glob("*.jpg"))
    assert linked.is_symlink()
    assert linked.resolve().exists()
    # The link must point back into the source tree.
    assert "images" in str(linked.resolve())


def test_copy_mode_produces_real_files(script, fake_food101: Path, tmp_path: Path):
    output = tmp_path / "out"
    script.prepare(fake_food101, output, copy=True)

    copied = next((output / "training/labeled/apple_pie").glob("*.jpg"))
    assert not copied.is_symlink()
    assert copied.stat().st_size > 0


def test_copied_tree_survives_source_deletion(
    script, fake_food101: Path, tmp_path: Path
):
    """--copy must be genuinely independent of the source."""
    import shutil

    output = tmp_path / "out"
    script.prepare(fake_food101, output, copy=True)
    shutil.rmtree(fake_food101)

    copied = next((output / "training/labeled/apple_pie").glob("*.jpg"))
    with Image.open(copied) as img:
        assert img.size == (16, 16)


def test_relative_symlinks_survive_a_move(script, fake_food101: Path, tmp_path: Path):
    """Relative links keep working when source and output move together."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "food-101"
    import shutil

    shutil.move(str(fake_food101), str(source))

    output = bundle / "prepared"
    script.prepare(source, output, relative=True)

    moved = tmp_path / "relocated"
    shutil.move(str(bundle), str(moved))

    linked = next((moved / "prepared/training/labeled/apple_pie").glob("*.jpg"))
    assert linked.resolve().exists(), "relative symlink broke after moving the tree"


def test_limit_per_class(script, fake_food101: Path, tmp_path: Path):
    output = tmp_path / "out"
    counts = script.prepare(fake_food101, output, limit_per_class=2)

    assert counts["train"] == 6  # 3 classes x 2
    assert len(list((output / "training/labeled/apple_pie").glob("*.jpg"))) == 2


def test_rerun_is_idempotent(script, fake_food101: Path, tmp_path: Path):
    output = tmp_path / "out"
    first = script.prepare(fake_food101, output)
    second = script.prepare(fake_food101, output)

    assert first == second
    assert len(list((output / "training/labeled/apple_pie").glob("*.jpg"))) == 3


def test_missing_images_dir_raises(script, tmp_path: Path):
    bad = tmp_path / "not-food101"
    (bad / "meta").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="images"):
        script.prepare(bad, tmp_path / "out")


def test_missing_meta_dir_raises(script, tmp_path: Path):
    bad = tmp_path / "not-food101"
    (bad / "images").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="meta"):
        script.prepare(bad, tmp_path / "out")


def test_malformed_meta_line_raises(script, fake_food101: Path, tmp_path: Path):
    (fake_food101 / "meta/train.txt").write_text(
        "apple_pie/123\nno_slash_here\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected"):
        script.prepare(fake_food101, tmp_path / "out")


def test_empty_meta_file_raises(script, fake_food101: Path, tmp_path: Path):
    (fake_food101 / "meta/train.txt").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no entries"):
        script.prepare(fake_food101, tmp_path / "out")


def test_image_listed_but_absent_is_skipped(
    script, fake_food101: Path, tmp_path: Path, capsys
):
    """A meta entry with no file must warn, not crash."""
    meta = fake_food101 / "meta/train.txt"
    meta.write_text(meta.read_text() + "apple_pie/does_not_exist\n", encoding="utf-8")

    counts = script.prepare(fake_food101, tmp_path / "out")
    assert counts["train"] == 9  # the missing one is skipped, not counted
    assert "missing" in capsys.readouterr().err


def test_prepared_output_loads_with_the_dataset_class(
    script, fake_food101: Path, tmp_path: Path
):
    """The whole point: the output must be readable by the training pipeline."""
    from food_recognition.data import LabeledImageDataset

    output = tmp_path / "out"
    script.prepare(fake_food101, output)

    dataset = LabeledImageDataset(output / "training/labeled")
    assert dataset.classes == ["apple_pie", "baby_back_ribs", "waffles"]
    assert len(dataset) == 9

    image, label = dataset[0]
    assert label in {0, 1, 2}
    assert image.size == (16, 16)  # PIL image, no transform applied


def test_prepared_output_trains_end_to_end(script, fake_food101: Path, tmp_path: Path):
    """Run real training on the converted tree."""
    from food_recognition.config import EarlyStoppingConfig, TrainingConfig
    from food_recognition.training import train_model

    output = tmp_path / "out"
    script.prepare(fake_food101, output)

    cfg = TrainingConfig(
        model_name="simple_cnn",
        num_classes=3,
        use_pretrained=False,
        train_dir=output / "training/labeled",
        val_dir=output / "validation",
        image_size=32,
        batch_size=2,
        num_workers=0,
        epochs=1,
        device="cpu",
        use_amp=False,
        output_dir=tmp_path / "run",
        early_stopping=EarlyStoppingConfig(enabled=False),
    )
    summary = train_model(cfg)

    assert summary.classes == ["apple_pie", "baby_back_ribs", "waffles"]
    assert summary.best_checkpoint is not None


def test_cli_reports_missing_source(script, tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_food101.py",
            "--source", str(tmp_path / "absent"),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert script.main() == 1
    assert "error" in capsys.readouterr().err


def test_cli_succeeds_on_valid_source(
    script, fake_food101: Path, tmp_path: Path, capsys, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_food101.py",
            "--source", str(fake_food101),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert script.main() == 0

    out = capsys.readouterr()
    assert "training/labeled" in out.out
    # The miniature fixture is not the full dataset, so it must warn about totals.
    assert "warning" in out.err
