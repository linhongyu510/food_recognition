"""Shared utilities: seeding, device resolution, early stopping, checkpoints."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "resolve_device",
    "ensure_dir",
    "EarlyStopper",
    "save_checkpoint",
    "load_checkpoint",
    "write_json",
    "configure_logging",
    "format_duration",
]

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a concise format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    With ``deterministic=True`` cuDNN autotuning is disabled for
    reproducibility at some cost in throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string to an available :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. An explicit request for
    an unavailable backend falls back to CPU with a warning rather than
    crashing partway through training.
    """
    def _mps_available() -> bool:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    requested = device.strip().lower()

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested == "mps" and not _mps_available():
        logger.warning("MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")

    return torch.device(requested)


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class EarlyStopper:
    """Track a monitored metric and signal when to stop.

    ``mode="max"`` treats larger values as better (accuracy);
    ``mode="min"`` treats smaller as better (loss).
    """

    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"

    best: Optional[float] = None
    best_epoch: int = 0
    counter: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        """Record ``value``; return ``True`` if it is a new best."""
        if self._is_better(value):
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    *,
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    classes: Optional[list[str]] = None,
) -> Path:
    """Save a self-describing checkpoint.

    Bundling config and class names means :mod:`predict` can restore a model
    without being told the architecture again.
    """
    path = Path(path)
    ensure_dir(path.parent)

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "classes": classes,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(path: Path | str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    # weights_only=True is the safe default on torch>=2.6 but is not accepted
    # by older releases, so fall back when the kwarg is unsupported.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - torch < 1.13
        return torch.load(path, map_location=map_location)


def write_json(path: Path | str, data: Any) -> Path:
    """Write ``data`` as UTF-8 JSON with stable indentation."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def format_duration(seconds: float) -> str:
    """Format a duration as ``1h02m03s`` / ``2m03s`` / ``3.4s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"
