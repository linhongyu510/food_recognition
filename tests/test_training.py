"""End-to-end training, checkpointing and inference tests.

These actually run the training loop on tiny synthetic data rather than
asserting on mocks, so a broken loop cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from food_recognition.config import (
    EarlyStoppingConfig,
    SemiSupervisedConfig,
    TrainingConfig,
)
from food_recognition.predict import load_predictor
from food_recognition.training import Trainer, train_model
from food_recognition.utils import EarlyStopper, load_checkpoint


def _tiny_cfg(sample_dataset: Path, output: Path, **kwargs) -> TrainingConfig:
    params = {
        "model_name": "simple_cnn",
        "num_classes": 3,
        "use_pretrained": False,
        "train_dir": sample_dataset / "training/labeled",
        "val_dir": sample_dataset / "validation",
        "unlabeled_dir": None,
        "image_size": 32,
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 2,
        "learning_rate": 1e-3,
        "device": "cpu",
        "use_amp": False,
        "output_dir": output,
        "early_stopping": EarlyStoppingConfig(enabled=False),
    }
    params.update(kwargs)
    return TrainingConfig(**params)


def test_training_runs_and_saves_checkpoint(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run")
    summary = train_model(cfg)

    assert len(summary.history) == 2
    assert summary.best_checkpoint is not None
    assert summary.best_checkpoint.exists()
    assert 0.0 <= summary.best_accuracy <= 1.0
    assert summary.classes == ["00", "01", "02"]


def test_training_actually_reduces_loss(sample_dataset: Path, tmp_path: Path):
    """The classes are visually separable, so loss must fall over 6 epochs."""
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", epochs=6, learning_rate=3e-3)
    summary = train_model(cfg)

    first = summary.history[0].train_loss
    last = summary.history[-1].train_loss
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_history_and_metrics_files_are_written(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run")
    train_model(cfg)

    history = json.loads(cfg.history_path().read_text())
    assert len(history) == 2
    assert {"epoch", "train_loss", "val_acc"} <= set(history[0])

    metrics = json.loads(cfg.metrics_path().read_text())
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics


def test_last_checkpoint_is_saved(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", save_last=True)
    train_model(cfg)
    assert cfg.last_checkpoint_path().exists()


def test_checkpoint_is_self_describing(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run")
    summary = train_model(cfg)

    payload = load_checkpoint(summary.best_checkpoint)
    assert "model_state" in payload
    assert payload["classes"] == ["00", "01", "02"]
    assert payload["config"]["model_name"] == "simple_cnn"
    assert payload["config"]["num_classes"] == 3


def test_evaluate_returns_full_report(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", epochs=1)
    trainer = Trainer(cfg)
    trainer.train()

    report = trainer.evaluate()
    assert 0.0 <= report.accuracy <= 1.0
    assert len(report.per_class_f1) == 3
    assert sum(report.support) == 9
    assert report.loss is not None


def test_scheduler_changes_learning_rate(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(
        sample_dataset, tmp_path / "run", epochs=4, scheduler="cosine"
    )
    summary = train_model(cfg)

    rates = [record.learning_rate for record in summary.history]
    assert rates[0] != rates[-1], "cosine schedule did not change the LR"


def test_scheduler_none_keeps_learning_rate_constant(
    sample_dataset: Path, tmp_path: Path
):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", epochs=3, scheduler="none")
    summary = train_model(cfg)

    rates = {record.learning_rate for record in summary.history}
    assert len(rates) == 1


def test_warmup_ramps_learning_rate(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(
        sample_dataset,
        tmp_path / "run",
        epochs=4,
        warmup_epochs=2,
        learning_rate=1e-2,
        scheduler="none",
    )
    summary = train_model(cfg)
    assert summary.history[0].learning_rate < cfg.learning_rate


def test_early_stopping_halts_training(sample_dataset: Path, tmp_path: Path):
    """With patience=1 and a frozen model, training must stop before max epochs."""
    cfg = _tiny_cfg(
        sample_dataset,
        tmp_path / "run",
        epochs=20,
        learning_rate=1e-12,  # effectively no learning -> no improvement
        early_stopping=EarlyStoppingConfig(enabled=True, patience=1, min_delta=0.5),
    )
    summary = train_model(cfg)

    assert summary.stopped_early
    assert len(summary.history) < 20


def test_semi_supervised_path_runs(sample_dataset: Path, tmp_path: Path):
    """Regression: this whole path used to crash on the flat unlabelled dir."""
    cfg = _tiny_cfg(
        sample_dataset,
        tmp_path / "run",
        epochs=2,
        unlabeled_dir=sample_dataset / "training/unlabeled",
        semi_supervised=SemiSupervisedConfig(
            enabled=True,
            refresh_interval=1,
            confidence_threshold=0.1,  # low, so samples are accepted
            activation_threshold=0.0,
            max_ratio=1.0,
        ),
    )
    summary = train_model(cfg)

    assert len(summary.history) == 2
    assert summary.history[-1].pseudo_samples > 0


def test_semi_supervised_respects_high_threshold(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(
        sample_dataset,
        tmp_path / "run",
        epochs=1,
        unlabeled_dir=sample_dataset / "training/unlabeled",
        semi_supervised=SemiSupervisedConfig(
            enabled=True,
            refresh_interval=1,
            confidence_threshold=0.999999,
            activation_threshold=0.0,
        ),
    )
    summary = train_model(cfg)
    assert summary.history[0].pseudo_samples == 0


def test_training_without_validation_set(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", val_dir=None)
    summary = train_model(cfg)

    assert len(summary.history) == 2
    assert summary.history[0].val_acc is None
    assert summary.best_checkpoint is None


def test_grad_clipping_does_not_break_training(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", grad_clip_norm=1.0)
    summary = train_model(cfg)
    assert torch.isfinite(torch.tensor(summary.history[-1].train_loss))


def test_label_smoothing_keeps_loss_finite(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", label_smoothing=0.1)
    summary = train_model(cfg)
    assert torch.isfinite(torch.tensor(summary.history[-1].train_loss))


def test_seeding_makes_runs_reproducible(sample_dataset: Path, tmp_path: Path):
    losses = []
    for index in range(2):
        cfg = _tiny_cfg(
            sample_dataset, tmp_path / f"run{index}", epochs=2, seed=1234
        )
        losses.append(train_model(cfg).history[-1].train_loss)

    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_predictor_roundtrip_from_checkpoint(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run")
    summary = train_model(cfg)

    predictor = load_predictor(summary.best_checkpoint, device="cpu")
    assert predictor.classes == ["00", "01", "02"]
    assert predictor.image_size == 32

    image = next((sample_dataset / "validation/01").glob("*.jpg"))
    prediction = predictor.predict(image, topk=3)

    assert prediction.label in {"00", "01", "02"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.topk) == 3
    assert sum(prob for _, prob in prediction.topk) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("dropout", [0.0, 0.2, 0.5])
def test_predictor_roundtrip_survives_dropout(
    sample_dataset: Path, tmp_path: Path, dropout: float
):
    """Regression: checkpoints trained with dropout could not be reloaded.

    dropout>0 wraps the head in Sequential(Dropout, Linear), moving the
    state_dict keys from "fc.weight" to "fc.1.weight". load_predictor rebuilt
    the model without dropout, so every such checkpoint failed with
    'Missing key(s) in state_dict: "fc.weight"'. This affected the default
    benchmark configs, i.e. the documented happy path.
    """
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run")
    cfg.model_name = "resnet18"
    cfg.dropout = dropout
    summary = train_model(cfg)

    predictor = load_predictor(summary.best_checkpoint, device="cpu")
    image = next((sample_dataset / "validation/01").glob("*.jpg"))
    assert predictor.predict(image).label in {"00", "01", "02"}


def test_reloaded_model_reproduces_training_accuracy(
    sample_dataset: Path, tmp_path: Path
):
    """A checkpoint that loads but predicts differently is worse than one that fails."""
    from food_recognition.data import build_transform
    from food_recognition.metrics import compute_metrics

    cfg = _tiny_cfg(sample_dataset, tmp_path / "run")
    cfg.model_name = "resnet18"
    cfg.dropout = 0.3
    summary = train_model(cfg)

    predictor = load_predictor(summary.best_checkpoint, device="cpu")

    val_dir = sample_dataset / "validation"
    transform = build_transform(cfg.image_size, is_train=False)
    images, targets = [], []
    for class_index, class_name in enumerate(sorted(p.name for p in val_dir.iterdir())):
        for path in sorted((val_dir / class_name).glob("*.jpg")):
            with Image.open(path) as img:
                images.append(transform(img.convert("RGB")))
            targets.append(class_index)

    with torch.no_grad():
        logits = predictor.model(torch.stack(images))
    reloaded = compute_metrics(
        torch.tensor(targets), logits.argmax(dim=1), cfg.num_classes
    ).accuracy

    assert reloaded == pytest.approx(summary.best_accuracy, abs=0.02), (
        f"reloaded model scored {reloaded:.4f} but training reported "
        f"{summary.best_accuracy:.4f}"
    )


def test_predictor_handles_flat_directory(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(sample_dataset, tmp_path / "run", epochs=1)
    summary = train_model(cfg)

    predictor = load_predictor(summary.best_checkpoint, device="cpu")
    results = predictor.predict_directory(sample_dataset / "training/unlabeled")
    assert len(results) == 4


def test_predictor_rejects_non_checkpoint(tmp_path: Path):
    bogus = tmp_path / "bogus.pt"
    torch.save({"something_else": 1}, bogus)

    with pytest.raises(ValueError, match="not a food_recognition checkpoint"):
        load_predictor(bogus)


def test_linear_probe_trains_only_the_head(sample_dataset: Path, tmp_path: Path):
    cfg = _tiny_cfg(
        sample_dataset,
        tmp_path / "run",
        model_name="resnet18",
        linear_probe=True,
        epochs=1,
    )
    trainer = Trainer(cfg)
    frozen_before = trainer.model.conv1.weight.detach().clone()
    trainer.train()

    assert torch.allclose(frozen_before, trainer.model.conv1.weight)


# ----------------------------------------------------------------------------
# EarlyStopper unit behaviour
# ----------------------------------------------------------------------------
def test_early_stopper_max_mode():
    stopper = EarlyStopper(patience=2, mode="max")

    assert stopper.update(0.5, 1) is True
    assert stopper.update(0.6, 2) is True
    assert stopper.update(0.55, 3) is False
    assert stopper.should_stop is False
    assert stopper.update(0.40, 4) is False
    assert stopper.should_stop is True
    assert stopper.best == pytest.approx(0.6)
    assert stopper.best_epoch == 2


def test_early_stopper_min_mode():
    stopper = EarlyStopper(patience=1, mode="min")

    assert stopper.update(1.0, 1) is True
    assert stopper.update(0.5, 2) is True
    assert stopper.update(0.7, 3) is False
    assert stopper.should_stop is True


def test_early_stopper_min_delta_ignores_tiny_gains():
    stopper = EarlyStopper(patience=5, mode="max", min_delta=0.1)

    stopper.update(0.50, 1)
    assert stopper.update(0.55, 2) is False  # +0.05 < min_delta
    assert stopper.update(0.65, 3) is True   # +0.15 > min_delta


def test_early_stopper_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        EarlyStopper(mode="sideways")
