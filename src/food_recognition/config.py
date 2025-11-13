"""配置相关的数据类与工具函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class SemiSupervisedConfig:
    """半监督训练的配置。"""

    enabled: bool = True
    refresh_interval: int = 5
    confidence_threshold: float = 0.99
    activation_threshold: float = 0.7


@dataclass
class TrainingConfig:
    """训练流程的统一配置。"""

    model_name: str = "resnet18"
    num_classes: int = 11

    train_dir: Path = field(
        default_factory=lambda: Path("food-11_sample/training/labeled")
    )
    val_dir: Path = field(default_factory=lambda: Path("food-11_sample/validation"))
    unlabeled_dir: Optional[Path] = field(
        default_factory=lambda: Path("food-11_sample/training/unlabeled")
    )

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    linear_probe: bool = False
    use_pretrained: bool = False

    device: str = "auto"
    seed: int = 0

    val_every_n_epochs: int = 1

    checkpoint_dir: Path = field(default_factory=lambda: Path("model_save"))
    checkpoint_name: str = "model.pth"
    save_best_only: bool = True

    semi_supervised: SemiSupervisedConfig = field(
        default_factory=SemiSupervisedConfig
    )

    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name


def load_training_config(path: Path) -> TrainingConfig:
    """从 YAML 文件加载训练配置。"""
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    semi_cfg_data = data.pop("semi_supervised", None)
    semi_cfg = SemiSupervisedConfig(**semi_cfg_data) if semi_cfg_data else SemiSupervisedConfig()

    return TrainingConfig(semi_supervised=semi_cfg, **data)

