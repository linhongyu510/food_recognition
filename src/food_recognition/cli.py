"""Console entry points: train, eval, predict.

Installed as ``food-recognition-train`` / ``-eval`` / ``-predict``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import TrainingConfig, load_training_config
from .metrics import compute_metrics
from .models import available_models
from .utils import configure_logging, write_json

__all__ = ["train_main", "eval_main", "predict_main"]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="food-recognition-train",
        description="Train a food classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, help="YAML config path. Omit to use built-in defaults."
    )
    parser.add_argument("--model-name", help=f"One of: {', '.join(available_models())}")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--unlabeled-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", help='"auto", "cpu", "cuda", "mps"')
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--scheduler", choices=["none", "cosine", "step", "plateau"]
    )
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--warmup-epochs", type=int)

    pre = parser.add_mutually_exclusive_group()
    pre.add_argument(
        "--pretrained", dest="use_pretrained", action="store_true", default=None
    )
    pre.add_argument("--no-pretrained", dest="use_pretrained", action="store_false")

    parser.add_argument("--linear-probe", action="store_true", default=None)
    parser.add_argument(
        "--no-amp", dest="use_amp", action="store_false", default=None,
        help="Disable mixed precision even on CUDA.",
    )
    parser.add_argument(
        "--semi-supervised", dest="semi_supervised_enabled",
        action="store_true", default=None,
        help="Enable pseudo-labelling (requires --unlabeled-dir).",
    )
    parser.add_argument(
        "--no-early-stopping", dest="early_stopping_enabled",
        action="store_false", default=None,
    )
    parser.add_argument("--quiet", action="store_true", help="Only log warnings.")
    return parser


_SIMPLE_OVERRIDES = (
    "model_name", "num_classes", "train_dir", "val_dir", "unlabeled_dir",
    "output_dir", "epochs", "batch_size", "learning_rate", "weight_decay",
    "image_size", "num_workers", "device", "seed", "scheduler",
    "label_smoothing", "dropout", "grad_clip_norm", "warmup_epochs",
    "use_pretrained", "linear_probe", "use_amp",
)


def _apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply non-``None`` CLI values on top of the config."""
    for name in _SIMPLE_OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)

    if getattr(args, "semi_supervised_enabled", None) is not None:
        cfg.semi_supervised.enabled = args.semi_supervised_enabled
    if getattr(args, "early_stopping_enabled", None) is not None:
        cfg.early_stopping.enabled = args.early_stopping_enabled

    cfg.__post_init__()  # re-coerce any string paths passed in
    return cfg


def train_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``food-recognition-train``."""
    args = _build_train_parser().parse_args(argv)
    configure_logging(logging.WARNING if args.quiet else logging.INFO)

    try:
        cfg = load_training_config(args.config) if args.config else TrainingConfig()
        cfg = _apply_overrides(cfg, args)
        cfg.validate()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("configuration error: %s", exc)
        return 2

    from .training import train_model  # imported late so --help stays fast

    try:
        summary = train_model(cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("training failed: %s", exc)
        return 1

    print()
    if summary.best_checkpoint:
        print(f"best val accuracy : {summary.best_accuracy:.4f} "
              f"(epoch {summary.best_epoch})")
        print(f"best checkpoint   : {summary.best_checkpoint}")
        print(f"metrics           : {cfg.metrics_path()}")
        print(f"history           : {cfg.history_path()}")
        if summary.final_report is not None:
            print()
            print(summary.final_report.format_table())
    else:
        print("no checkpoint saved (no validation set configured?)")
        print(f"history: {cfg.history_path()}")
    return 0


# ----------------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------------
def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="food-recognition-eval",
        description="Evaluate a checkpoint on a labelled directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Labelled directory (<root>/<class>/<image>).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", type=Path, help="Also write metrics as JSON.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def eval_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``food-recognition-eval``."""
    args = _build_eval_parser().parse_args(argv)
    configure_logging(logging.WARNING if args.quiet else logging.INFO)

    import torch
    from torch.utils.data import DataLoader

    from .data import LabeledImageDataset, build_transform
    from .predict import load_predictor

    try:
        predictor = load_predictor(args.checkpoint, device=args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("failed to load checkpoint: %s", exc)
        return 1

    try:
        dataset = LabeledImageDataset(
            args.data_dir,
            transform=build_transform(predictor.image_size, is_train=False),
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error("failed to load data: %s", exc)
        return 1

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = len(predictor.classes) if predictor.classes else len(dataset.classes)
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(predictor.device, non_blocking=True)
            logits = predictor.model(images)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(labels)

    report = compute_metrics(
        torch.cat(targets),
        torch.cat(preds),
        num_classes,
        class_names=predictor.classes or dataset.classes,
    )

    print(report.format_table())
    if args.json_out:
        write_json(args.json_out, report.to_dict())
        print(f"\nmetrics written to {args.json_out}")
    return 0


# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------
def _build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="food-recognition-predict",
        description="Classify an image or a directory of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def predict_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``food-recognition-predict``."""
    args = _build_predict_parser().parse_args(argv)
    configure_logging(logging.WARNING if args.quiet else logging.INFO)

    from .predict import load_predictor

    if not args.input.exists():
        logger.error("input not found: %s", args.input)
        return 1

    try:
        predictor = load_predictor(args.checkpoint, device=args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("failed to load checkpoint: %s", exc)
        return 1

    try:
        if args.input.is_dir():
            results = predictor.predict_directory(
                args.input, topk=args.topk, batch_size=args.batch_size
            )
        else:
            results = [predictor.predict(args.input, topk=args.topk)]
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("prediction failed: %s", exc)
        return 1

    for item in results:
        detail = ", ".join(f"{name} {prob:.3f}" for name, prob in item.topk)
        print(f"{item.path}\n  -> {item.label} ({item.confidence:.4f})  [{detail}]")

    if args.json_out:
        write_json(args.json_out, [item.to_dict() for item in results])
        print(f"\npredictions written to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(train_main())
