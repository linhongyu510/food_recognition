"""Tests for the console entry points, invoked exactly as users invoke them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from food_recognition.cli import eval_main, gradcam_main, predict_main, train_main


def _write_config(path: Path, dataset: Path, output: Path, **extra) -> Path:
    payload = {
        "model_name": "simple_cnn",
        "num_classes": 3,
        "use_pretrained": False,
        "train_dir": str(dataset / "training/labeled"),
        "val_dir": str(dataset / "validation"),
        "image_size": 32,
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 1,
        "device": "cpu",
        "use_amp": False,
        "output_dir": str(output),
        "early_stopping": {"enabled": False},
    }
    payload.update(extra)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def test_train_cli_succeeds(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")

    assert train_main(["--config", str(config)]) == 0

    out = capsys.readouterr().out
    assert "best val accuracy" in out
    assert "weighted avg" in out
    assert (tmp_path / "run/checkpoints/best.pt").exists()


def test_train_cli_overrides_config(sample_dataset: Path, tmp_path: Path):
    config = _write_config(
        tmp_path / "c.yaml", sample_dataset, tmp_path / "run", epochs=1
    )

    assert train_main(["--config", str(config), "--epochs", "2"]) == 0

    history = json.loads((tmp_path / "run/history.json").read_text())
    assert len(history) == 2, "CLI --epochs did not override the config"


def test_train_cli_accepts_dir_overrides(sample_dataset: Path, tmp_path: Path):
    """Run with no config at all, driving everything from flags."""
    code = train_main(
        [
            "--model-name", "simple_cnn",
            "--num-classes", "3",
            "--no-pretrained",
            "--train-dir", str(sample_dataset / "training/labeled"),
            "--val-dir", str(sample_dataset / "validation"),
            "--output-dir", str(tmp_path / "run"),
            "--image-size", "32",
            "--batch-size", "4",
            "--num-workers", "0",
            "--epochs", "1",
            "--device", "cpu",
        ]
    )
    assert code == 0
    assert (tmp_path / "run/checkpoints/best.pt").exists()


def test_train_cli_missing_config_returns_2(tmp_path: Path):
    assert train_main(["--config", str(tmp_path / "nope.yaml")]) == 2


def test_train_cli_invalid_config_returns_2(sample_dataset: Path, tmp_path: Path):
    config = _write_config(
        tmp_path / "c.yaml", sample_dataset, tmp_path / "run", scheduler="bogus"
    )
    assert train_main(["--config", str(config)]) == 2


def test_train_cli_missing_data_returns_1(tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_name": "simple_cnn",
                "num_classes": 3,
                "train_dir": str(tmp_path / "absent"),
                "val_dir": None,
                "device": "cpu",
                "epochs": 1,
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )
    assert train_main(["--config", str(config)]) == 1


def test_train_cli_semi_supervised_without_dir_returns_2(
    sample_dataset: Path, tmp_path: Path
):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    assert train_main(["--config", str(config), "--semi-supervised"]) == 2


# ----------------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------------
def test_eval_cli_reports_metrics(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    assert train_main(["--config", str(config)]) == 0
    capsys.readouterr()

    code = eval_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--data-dir", str(sample_dataset / "validation"),
            "--num-workers", "0",
        ]
    )
    assert code == 0

    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "macro avg" in out


def test_eval_cli_writes_json(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    out_path = tmp_path / "metrics.json"
    code = eval_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--data-dir", str(sample_dataset / "validation"),
            "--num-workers", "0",
            "--json-out", str(out_path),
        ]
    )
    assert code == 0

    data = json.loads(out_path.read_text())
    assert "accuracy" in data
    assert len(data["per_class"]) == 3


def test_eval_cli_missing_checkpoint_returns_1(sample_dataset: Path, tmp_path: Path):
    code = eval_main(
        [
            "--checkpoint", str(tmp_path / "nope.pt"),
            "--data-dir", str(sample_dataset / "validation"),
        ]
    )
    assert code == 1


# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------
def test_predict_cli_single_image(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    image = next((sample_dataset / "validation/00").glob("*.jpg"))
    code = predict_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(image),
        ]
    )
    assert code == 0
    assert "->" in capsys.readouterr().out


def test_predict_cli_directory_and_json(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    out_path = tmp_path / "preds.json"
    code = predict_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(sample_dataset / "training/unlabeled"),
            "--topk", "2",
            "--json-out", str(out_path),
        ]
    )
    assert code == 0

    data = json.loads(out_path.read_text())
    assert len(data) == 4
    assert all(len(item["topk"]) == 2 for item in data)
    assert all("label" in item for item in data)


def test_predict_cli_missing_input_returns_1(sample_dataset: Path, tmp_path: Path):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])

    code = predict_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(tmp_path / "absent.jpg"),
        ]
    )
    assert code == 1


# ----------------------------------------------------------------------------
# gradcam
# ----------------------------------------------------------------------------
def test_gradcam_cli_writes_overlay(sample_dataset: Path, tmp_path: Path, capsys):
    from PIL import Image

    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    image = next((sample_dataset / "validation/00").glob("*.jpg"))
    out_dir = tmp_path / "cams"
    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(image),
            "--output-dir", str(out_dir),
        ]
    )
    assert code == 0

    written = list(out_dir.glob("*_gradcam.png"))
    assert len(written) == 1
    with Image.open(written[0]) as img:
        assert img.size == (32, 32)  # matches the config's image_size


def test_gradcam_cli_side_by_side_doubles_width(
    sample_dataset: Path, tmp_path: Path, capsys
):
    from PIL import Image

    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    image = next((sample_dataset / "validation/00").glob("*.jpg"))
    out_dir = tmp_path / "cams"
    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(image),
            "--output-dir", str(out_dir),
            "--side-by-side",
        ]
    )
    assert code == 0

    with Image.open(next(out_dir.glob("*_gradcam.png"))) as img:
        assert img.size == (64, 32)


def test_gradcam_cli_processes_directory(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    out_dir = tmp_path / "cams"
    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(sample_dataset / "training/unlabeled"),
            "--output-dir", str(out_dir),
        ]
    )
    assert code == 0
    assert len(list(out_dir.glob("*_gradcam.png"))) == 4


def test_gradcam_cli_does_not_overwrite_duplicate_stems(
    sample_dataset: Path, tmp_path: Path, capsys
):
    """Regression: class dirs reuse file names (00/000.jpg, 01/000.jpg, ...).

    A flat output directory collapsed 12 images down to 4 files, silently
    discarding two thirds of the results.
    """
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    val_root = sample_dataset / "validation"
    n_inputs = len(list(val_root.rglob("*.jpg")))
    assert n_inputs > len({p.stem for p in val_root.rglob("*.jpg")}), (
        "fixture must contain colliding stems for this test to be meaningful"
    )

    out_dir = tmp_path / "cams"
    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(val_root),
            "--output-dir", str(out_dir),
        ]
    )
    assert code == 0

    written = list(out_dir.rglob("*_gradcam.png"))
    assert len(written) == n_inputs, (
        f"expected one overlay per input image ({n_inputs}), got {len(written)}"
    )
    # Structure must mirror the input so results stay attributable to a class.
    assert (out_dir / "00").is_dir()


def test_gradcam_cli_specific_class(sample_dataset: Path, tmp_path: Path, capsys):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])
    capsys.readouterr()

    image = next((sample_dataset / "validation/00").glob("*.jpg"))
    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(image),
            "--output-dir", str(tmp_path / "cams"),
            "--class-index", "2",
        ]
    )
    assert code == 0
    assert "02" in capsys.readouterr().out


def test_gradcam_cli_rejects_bad_alpha(sample_dataset: Path, tmp_path: Path):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])

    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(next((sample_dataset / "validation/00").glob("*.jpg"))),
            "--alpha", "2.0",
        ]
    )
    assert code == 2


def test_gradcam_cli_missing_input_returns_1(sample_dataset: Path, tmp_path: Path):
    config = _write_config(tmp_path / "c.yaml", sample_dataset, tmp_path / "run")
    train_main(["--config", str(config)])

    code = gradcam_main(
        [
            "--checkpoint", str(tmp_path / "run/checkpoints/best.pt"),
            "--input", str(tmp_path / "absent.jpg"),
        ]
    )
    assert code == 1


# ----------------------------------------------------------------------------
# --help must work for every entry point
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("main", [train_main, eval_main, predict_main, gradcam_main])
def test_help_exits_zero(main, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out
