"""Tests for the console entry points, invoked exactly as users invoke them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from food_recognition.cli import eval_main, predict_main, train_main


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
# --help must work for every entry point
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("main", [train_main, eval_main, predict_main])
def test_help_exits_zero(main, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out
