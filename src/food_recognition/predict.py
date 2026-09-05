"""Inference on single images or directories.

Checkpoints saved by this project embed their own config and class names, so a
predictor can be restored from a checkpoint alone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import IMAGE_EXTENSIONS, _load_rgb, build_transform
from .models import initialize_model
from .utils import load_checkpoint, resolve_device

__all__ = ["Prediction", "Predictor", "load_predictor"]

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single image's prediction."""

    path: str
    label_index: int
    label: str
    confidence: float
    topk: list[tuple[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "label_index": self.label_index,
            "label": self.label,
            "confidence": self.confidence,
            "topk": [{"label": name, "probability": p} for name, p in self.topk],
        }


class Predictor:
    """Wraps a trained model for inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        classes: Sequence[str] | None = None,
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.classes = list(classes) if classes else None
        self.transform = build_transform(image_size, is_train=False)

    def _label(self, index: int) -> str:
        if self.classes and 0 <= index < len(self.classes):
            return self.classes[index]
        return str(index)

    @torch.no_grad()
    def predict_batch(
        self, paths: Sequence[Path | str], topk: int = 3
    ) -> list[Prediction]:
        """Predict a batch of image paths in a single forward pass."""
        if not paths:
            return []

        tensors = torch.stack(
            [self.transform(_load_rgb(path)) for path in paths]
        ).to(self.device)

        probs = torch.softmax(self.model(tensors), dim=1).cpu()
        k = min(topk, probs.size(1))
        top_probs, top_idx = probs.topk(k, dim=1)

        results: list[Prediction] = []
        for row, (path, p_row, i_row) in enumerate(zip(paths, top_probs, top_idx)):
            del row
            best_index = int(i_row[0].item())
            results.append(
                Prediction(
                    path=str(path),
                    label_index=best_index,
                    label=self._label(best_index),
                    confidence=float(p_row[0].item()),
                    topk=[
                        (self._label(int(i.item())), float(p.item()))
                        for p, i in zip(p_row, i_row)
                    ],
                )
            )
        return results

    def predict(self, path: Path | str, topk: int = 3) -> Prediction:
        """Predict a single image."""
        return self.predict_batch([path], topk=topk)[0]

    def predict_directory(
        self,
        directory: Path | str,
        *,
        topk: int = 3,
        batch_size: int = 32,
        recursive: bool = True,
    ) -> list[Prediction]:
        """Predict every image under ``directory``."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"directory not found: {directory}")

        pattern = directory.rglob("*") if recursive else directory.glob("*")
        paths = sorted(
            p for p in pattern
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"no images found under {directory}")

        results: list[Prediction] = []
        for start in range(0, len(paths), batch_size):
            results.extend(self.predict_batch(paths[start : start + batch_size], topk))
        return results


def load_predictor(
    checkpoint: Path | str,
    *,
    device: str | None = None,
    model_name: str | None = None,
    num_classes: int | None = None,
    image_size: int | None = None,
) -> Predictor:
    """Rebuild a :class:`Predictor` from a checkpoint.

    Architecture, class names and image size are read from the checkpoint's
    embedded config; the keyword arguments only override them, which matters
    for checkpoints produced elsewhere.
    """
    payload = load_checkpoint(checkpoint, map_location="cpu")

    if "model_state" not in payload:
        raise ValueError(
            f"{checkpoint} is not a food_recognition checkpoint "
            "(missing 'model_state')"
        )

    config = payload.get("config") or {}
    resolved_name = model_name or config.get("model_name")
    resolved_classes = payload.get("classes")
    resolved_num = (
        num_classes
        or config.get("num_classes")
        or (len(resolved_classes) if resolved_classes else None)
    )
    resolved_size = image_size or config.get("image_size") or 224
    # dropout>0 wraps the head in Sequential(Dropout, Linear), which shifts the
    # state_dict keys from "fc.weight" to "fc.1.weight". Rebuilding without it
    # makes every checkpoint trained with dropout unloadable.
    resolved_dropout = float(config.get("dropout") or 0.0)

    if not resolved_name:
        raise ValueError(
            "cannot determine model_name from checkpoint; pass model_name= explicitly"
        )
    if not resolved_num:
        raise ValueError(
            "cannot determine num_classes from checkpoint; pass num_classes= explicitly"
        )

    # Skip downloading pretrained weights: they are immediately overwritten.
    model, _ = initialize_model(
        resolved_name,
        int(resolved_num),
        use_pretrained=False,
        dropout=resolved_dropout,
    )
    model.load_state_dict(payload["model_state"])

    return Predictor(
        model,
        classes=resolved_classes,
        image_size=int(resolved_size),
        device=device,
    )
