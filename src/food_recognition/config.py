"""Configuration dataclasses and YAML loading for food_recognition.

All training behaviour is driven by :class:`TrainingConfig`. Configs can be
built in Python or loaded from YAML via :func:`load_training_config`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "SemiSupervisedConfig",
    "EarlyStoppingConfig",
    "TrainingConfig",
    "load_training_config",
    "dump_training_config",
]


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


@dataclass
class SemiSupervisedConfig:
    """Pseudo-labelling (self-training) options.

    The unlabeled pool is only consulted once validation accuracy reaches
    ``activation_threshold``; predictions below ``confidence_threshold`` are
    discarded. ``refresh_interval`` controls how often pseudo-labels are
    regenerated (in epochs).
    """

    enabled: bool = False
    refresh_interval: int = 5
    confidence_threshold: float = 0.95
    activation_threshold: float = 0.70
    max_ratio: float = 2.0
    """Cap pseudo-labelled samples at ``max_ratio`` x labelled set size."""

    def validate(self) -> None:
        if self.refresh_interval < 1:
            raise ValueError("semi_supervised.refresh_interval must be >= 1")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "semi_supervised.confidence_threshold must be in (0, 1]"
            )
        if not 0.0 <= self.activation_threshold <= 1.0:
            raise ValueError(
                "semi_supervised.activation_threshold must be in [0, 1]"
            )
        if self.max_ratio <= 0:
            raise ValueError("semi_supervised.max_ratio must be > 0")


@dataclass
class EarlyStoppingConfig:
    """Stop training when the monitored metric stops improving."""

    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    def validate(self) -> None:
        if self.patience < 1:
            raise ValueError("early_stopping.patience must be >= 1")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be >= 0")


@dataclass
class TrainingConfig:
    """Single source of truth for a training run."""

    # --- model ---
    model_name: str = "resnet18"
    num_classes: int = 11
    use_pretrained: bool = True
    linear_probe: bool = False
    dropout: float = 0.0

    # --- data ---
    train_dir: Path = field(default_factory=lambda: Path("data/food-11/training/labeled"))
    val_dir: Path | None = field(
        default_factory=lambda: Path("data/food-11/validation")
    )
    unlabeled_dir: Path | None = None
    class_names: list[str] | None = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    use_autoaugment: bool = True

    # --- optimisation ---
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = None
    scheduler: str = "cosine"  # one of: none | cosine | step | plateau
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    warmup_epochs: int = 0

    # --- runtime ---
    device: str = "auto"
    seed: int = 0
    use_amp: bool = True
    val_every_n_epochs: int = 1
    deterministic: bool = True

    # --- output ---
    output_dir: Path = field(default_factory=lambda: Path("runs/exp"))
    checkpoint_name: str = "best.pt"
    save_last: bool = True

    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    semi_supervised: SemiSupervisedConfig = field(default_factory=SemiSupervisedConfig)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.train_dir = _coerce_path(self.train_dir)
        self.val_dir = _coerce_path(self.val_dir)
        self.unlabeled_dir = _coerce_path(self.unlabeled_dir)
        self.output_dir = _coerce_path(self.output_dir)
        if isinstance(self.early_stopping, Mapping):
            self.early_stopping = EarlyStoppingConfig(**self.early_stopping)
        if isinstance(self.semi_supervised, Mapping):
            self.semi_supervised = SemiSupervisedConfig(**self.semi_supervised)

    # ------------------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name

    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    def history_path(self) -> Path:
        return self.output_dir / "history.json"

    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Fail fast on invalid combinations before any heavy work starts."""
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.image_size < 32:
            raise ValueError("image_size must be >= 32")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.val_every_n_epochs < 1:
            raise ValueError("val_every_n_epochs must be >= 1")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0 when set")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be < epochs")

        valid_schedulers = {"none", "cosine", "step", "plateau"}
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"scheduler must be one of {sorted(valid_schedulers)}, "
                f"got {self.scheduler!r}"
            )

        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but "
                f"num_classes={self.num_classes}"
            )

        if self.semi_supervised.enabled and self.unlabeled_dir is None:
            raise ValueError(
                "semi_supervised.enabled=true requires unlabeled_dir to be set"
            )

        self.early_stopping.validate()
        self.semi_supervised.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain types suitable for YAML/JSON."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if dataclasses.is_dataclass(value):
                return {f.name: _convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return {f.name: _convert(getattr(self, f.name)) for f in fields(self)}


def load_training_config(path: Path | str, **overrides: Any) -> TrainingConfig:
    """Load a :class:`TrainingConfig` from YAML.

    Relative ``train_dir`` / ``val_dir`` / ``output_dir`` values are interpreted
    relative to the **current working directory**, matching how every other CLI
    tool behaves. (The previous implementation resolved them against the config
    file's own directory, so ``configs/default.yaml`` looked for data under
    ``configs/`` and could never find it.)

    Unknown keys raise ``ValueError`` rather than being silently ignored, so a
    typo in a config file is reported instead of quietly changing nothing.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

    raw.update(overrides)

    known = {f.name for f in fields(TrainingConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown config keys: {sorted(unknown)}. Valid keys: {sorted(known)}"
        )

    cfg = TrainingConfig(**raw)
    cfg.validate()
    return cfg


def dump_training_config(cfg: TrainingConfig, path: Path | str) -> Path:
    """Write ``cfg`` to ``path`` as YAML and return the path."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
