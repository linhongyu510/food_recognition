"""Classification metrics computed from scratch (no scikit-learn dependency).

The README previously claimed precision / recall / F1 / confusion-matrix
support that did not exist anywhere in the codebase. This module provides it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

__all__ = ["ClassificationReport", "confusion_matrix", "compute_metrics", "topk_accuracy"]


def confusion_matrix(
    targets: torch.Tensor, preds: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return a ``[num_classes, num_classes]`` matrix indexed ``[true, pred]``."""
    if targets.shape != preds.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} != preds shape {tuple(preds.shape)}"
        )
    t = targets.detach().flatten().to(torch.int64)
    p = preds.detach().flatten().to(torch.int64)

    if t.numel() and (t.min() < 0 or t.max() >= num_classes):
        raise ValueError(f"target labels outside [0, {num_classes - 1}]")
    if p.numel() and (p.min() < 0 or p.max() >= num_classes):
        raise ValueError(f"predicted labels outside [0, {num_classes - 1}]")

    indices = t * num_classes + p
    matrix = torch.zeros(num_classes * num_classes, dtype=torch.int64)
    matrix.scatter_add_(0, indices, torch.ones_like(indices))
    return matrix.reshape(num_classes, num_classes)


@dataclass
class ClassificationReport:
    """Aggregate and per-class classification metrics.

    ``macro_*`` averages treat every class equally; ``weighted_*`` weights by
    support. For single-label classification, micro-averaged precision, recall
    and F1 all equal accuracy, so only accuracy is reported for that case.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    support: list[int]
    matrix: list[list[int]]
    class_names: list[str] | None = None
    loss: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "per_class": [
                {
                    "name": (
                        self.class_names[i] if self.class_names else str(i)
                    ),
                    "precision": self.per_class_precision[i],
                    "recall": self.per_class_recall[i],
                    "f1": self.per_class_f1[i],
                    "support": self.support[i],
                }
                for i in range(len(self.support))
            ],
            "confusion_matrix": self.matrix,
        }
        if self.loss is not None:
            data["loss"] = self.loss
        if self.extra:
            data.update(self.extra)
        return data

    def format_table(self, digits: int = 4) -> str:
        """Render a scikit-learn-style text report."""
        names = self.class_names or [str(i) for i in range(len(self.support))]
        width = max([len(n) for n in names] + [12])
        lines = [
            f"{'class'.ljust(width)}  {'precision':>9}  {'recall':>9}  "
            f"{'f1':>9}  {'support':>7}"
        ]
        for i, name in enumerate(names):
            lines.append(
                f"{name.ljust(width)}  "
                f"{self.per_class_precision[i]:>9.{digits}f}  "
                f"{self.per_class_recall[i]:>9.{digits}f}  "
                f"{self.per_class_f1[i]:>9.{digits}f}  "
                f"{self.support[i]:>7d}"
            )
        total = sum(self.support)
        lines.append("")
        lines.append(
            f"{'accuracy'.ljust(width)}  {'':>9}  {'':>9}  "
            f"{self.accuracy:>9.{digits}f}  {total:>7d}"
        )
        lines.append(
            f"{'macro avg'.ljust(width)}  "
            f"{self.macro_precision:>9.{digits}f}  "
            f"{self.macro_recall:>9.{digits}f}  "
            f"{self.macro_f1:>9.{digits}f}  {total:>7d}"
        )
        lines.append(
            f"{'weighted avg'.ljust(width)}  "
            f"{self.weighted_precision:>9.{digits}f}  "
            f"{self.weighted_recall:>9.{digits}f}  "
            f"{self.weighted_f1:>9.{digits}f}  {total:>7d}"
        )
        return "\n".join(lines)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Element-wise division that yields 0 where the denominator is 0.

    Matches the ``zero_division=0`` convention: a class never predicted has
    precision 0 rather than NaN.
    """
    out = torch.zeros_like(num, dtype=torch.float64)
    mask = den > 0
    out[mask] = num[mask].to(torch.float64) / den[mask].to(torch.float64)
    return out


def compute_metrics(
    targets: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
    *,
    class_names: Sequence[str] | None = None,
    loss: float | None = None,
) -> ClassificationReport:
    """Compute accuracy, per-class and averaged precision/recall/F1."""
    matrix = confusion_matrix(targets, preds, num_classes)

    tp = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    total = int(matrix.sum().item())

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    accuracy = float(tp.sum().item() / total) if total else 0.0

    # Macro average over classes that actually appear in the targets, so
    # absent classes do not silently drag the average toward zero.
    present = support > 0
    if present.any():
        macro_precision = float(precision[present].mean().item())
        macro_recall = float(recall[present].mean().item())
        macro_f1 = float(f1[present].mean().item())
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    weights = support.to(torch.float64)
    weight_sum = float(weights.sum().item())
    if weight_sum > 0:
        weighted_precision = float((precision * weights).sum().item() / weight_sum)
        weighted_recall = float((recall * weights).sum().item() / weight_sum)
        weighted_f1 = float((f1 * weights).sum().item() / weight_sum)
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return ClassificationReport(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        per_class_precision=[float(v) for v in precision.tolist()],
        per_class_recall=[float(v) for v in recall.tolist()],
        per_class_f1=[float(v) for v in f1.tolist()],
        support=[int(v) for v in support.tolist()],
        matrix=[[int(v) for v in row] for row in matrix.tolist()],
        class_names=list(class_names) if class_names else None,
        loss=loss,
    )


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy. ``k`` is clamped to the number of classes."""
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got shape {tuple(logits.shape)}")
    if logits.size(0) == 0:
        return 0.0
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    correct = (topk == targets.view(-1, 1)).any(dim=1)
    return float(correct.to(torch.float64).mean().item())
