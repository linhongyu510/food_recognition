"""Tests for config validation and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from food_recognition.config import (
    EarlyStoppingConfig,
    SemiSupervisedConfig,
    TrainingConfig,
    dump_training_config,
    load_training_config,
)


def test_defaults_are_valid():
    TrainingConfig().validate()


def test_paths_are_coerced_from_strings():
    cfg = TrainingConfig(train_dir="a/b", output_dir="out")
    assert isinstance(cfg.train_dir, Path)
    assert isinstance(cfg.output_dir, Path)


def test_derived_paths():
    cfg = TrainingConfig(output_dir="runs/demo", checkpoint_name="best.pt")
    assert cfg.checkpoint_path() == Path("runs/demo/checkpoints/best.pt")
    assert cfg.last_checkpoint_path() == Path("runs/demo/checkpoints/last.pt")
    assert cfg.history_path() == Path("runs/demo/history.json")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_classes": 1}, "num_classes"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"image_size": 8}, "image_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"val_every_n_epochs": 0}, "val_every_n_epochs"),
        ({"label_smoothing": 1.0}, "label_smoothing"),
        ({"dropout": 1.5}, "dropout"),
        ({"grad_clip_norm": 0}, "grad_clip_norm"),
        ({"scheduler": "magic"}, "scheduler"),
        ({"warmup_epochs": -1}, "warmup_epochs"),
    ],
)
def test_invalid_values_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs).validate()


def test_warmup_must_be_shorter_than_training():
    with pytest.raises(ValueError, match="warmup_epochs"):
        TrainingConfig(epochs=5, warmup_epochs=5).validate()


def test_class_names_length_must_match_num_classes():
    with pytest.raises(ValueError, match="class_names"):
        TrainingConfig(num_classes=3, class_names=["a", "b"]).validate()


def test_semi_supervised_requires_unlabeled_dir():
    cfg = TrainingConfig(semi_supervised=SemiSupervisedConfig(enabled=True))
    cfg.unlabeled_dir = None
    with pytest.raises(ValueError, match="unlabeled_dir"):
        cfg.validate()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"refresh_interval": 0}, "refresh_interval"),
        ({"confidence_threshold": 0.0}, "confidence_threshold"),
        ({"confidence_threshold": 1.5}, "confidence_threshold"),
        ({"activation_threshold": 1.5}, "activation_threshold"),
        ({"max_ratio": 0}, "max_ratio"),
    ],
)
def test_semi_supervised_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SemiSupervisedConfig(**kwargs).validate()


def test_early_stopping_validation():
    with pytest.raises(ValueError, match="patience"):
        EarlyStoppingConfig(patience=0).validate()
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStoppingConfig(min_delta=-1).validate()


def test_nested_dicts_become_dataclasses():
    cfg = TrainingConfig(
        early_stopping={"enabled": False, "patience": 3},
        semi_supervised={"enabled": False, "confidence_threshold": 0.5},
    )
    assert isinstance(cfg.early_stopping, EarlyStoppingConfig)
    assert cfg.early_stopping.patience == 3
    assert isinstance(cfg.semi_supervised, SemiSupervisedConfig)
    assert cfg.semi_supervised.confidence_threshold == 0.5


def test_load_yaml_config(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_name": "resnet50",
                "num_classes": 5,
                "epochs": 3,
                "train_dir": "data/train",
                "val_dir": "data/val",
                "early_stopping": {"enabled": False, "patience": 2},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_training_config(path)

    assert cfg.model_name == "resnet50"
    assert cfg.num_classes == 5
    assert cfg.early_stopping.enabled is False


def test_relative_paths_resolve_against_cwd_not_config_dir(tmp_path: Path):
    """Regression: the old loader resolved data paths against the config's own
    directory, so 'configs/default.yaml' looked for data under 'configs/'.
    """
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"num_classes": 3, "train_dir": "data/train"}), encoding="utf-8"
    )

    cfg = load_training_config(path)
    assert "configs" not in str(cfg.train_dir)
    assert cfg.train_dir == Path("data/train")


def test_unknown_config_key_raises(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"num_classes": 3, "epochz": 10}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        load_training_config(path)


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_training_config(tmp_path / "nope.yaml")


def test_non_mapping_config_raises(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_training_config(path)


def test_overrides_are_applied(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"num_classes": 3, "epochs": 10}), encoding="utf-8")
    cfg = load_training_config(path, epochs=1)
    assert cfg.epochs == 1


def test_roundtrip_dump_and_load(tmp_path: Path):
    original = TrainingConfig(
        model_name="efficientnet_b0_cbam", num_classes=7, epochs=4, dropout=0.3
    )
    path = dump_training_config(original, tmp_path / "out.yaml")
    reloaded = load_training_config(path)

    assert reloaded.model_name == original.model_name
    assert reloaded.num_classes == original.num_classes
    assert reloaded.dropout == original.dropout


def test_to_dict_is_json_safe():
    import json

    data = TrainingConfig().to_dict()
    json.dumps(data)  # must not raise
    assert isinstance(data["train_dir"], str)
    assert isinstance(data["semi_supervised"], dict)


def test_all_shipped_configs_load():
    """Every YAML in configs/ must be valid, so no shipped example is broken."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(config_dir.glob("*.yaml"))
    assert files, "no shipped configs found"

    for path in files:
        cfg = load_training_config(path)
        assert cfg.model_name
        assert cfg.num_classes >= 2
