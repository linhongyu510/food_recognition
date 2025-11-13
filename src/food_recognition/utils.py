"""通用工具函数。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 的随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """根据配置返回 torch.device。

    参数:
        device: 指定设备，可选值为 "cpu"、"cuda"、"mps" 或 "auto"/None。
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "mps" and not getattr(torch.backends, "mps", None):
        return torch.device("cpu")
    return torch.device(device)


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型中可训练参数的数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class EarlyStopperState:
    """用于跟踪早停状态的数据类。"""

    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0

