#!/usr/bin/env python
"""命令行训练入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from food_recognition import TrainingConfig, train_model
from food_recognition.config import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food Recognition 训练脚本")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="训练配置文件路径（YAML），默认使用 configs/default.yaml。",
    )
    parser.add_argument("--model-name", type=str, help="覆盖配置中的 model_name。")
    parser.add_argument("--epochs", type=int, help="覆盖配置中的 epochs。")
    parser.add_argument("--batch-size", type=int, help="覆盖配置中的 batch_size。")
    parser.add_argument("--lr", type=float, help="覆盖配置中的 learning_rate。")
    parser.add_argument("--device", type=str, help='覆盖配置中的 device，取值如 "cuda"、"cpu"。')
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="覆盖配置，使模型使用预训练权重。",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="覆盖配置，使模型不使用预训练权重。",
    )
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help="仅训练分类头（线性探测）。",
    )
    return parser.parse_args()


def load_config(path: Path) -> TrainingConfig:
    if path.exists():
        cfg = load_training_config(path)
        base = path.parent
    else:
        cfg = TrainingConfig()
        base = Path.cwd()

    cfg.train_dir = (base / cfg.train_dir).resolve()
    cfg.val_dir = (base / cfg.val_dir).resolve()
    if cfg.unlabeled_dir is not None:
        cfg.unlabeled_dir = (base / cfg.unlabeled_dir).resolve()
    cfg.checkpoint_dir = (base / cfg.checkpoint_dir).resolve()
    return cfg


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.model_name:
        cfg.model_name = args.model_name
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.device:
        cfg.device = args.device
    if args.use_pretrained:
        cfg.use_pretrained = True
    if args.no_pretrained:
        cfg.use_pretrained = False
    if args.linear_probe:
        cfg.linear_probe = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    summary = train_model(cfg)

    print("\n训练完成。")
    if summary.best_checkpoint:
        print(f"最佳验证准确率: {summary.best_accuracy:.4f}")
        print(f"最佳模型已保存至: {summary.best_checkpoint}")
    else:
        print("未保存模型，请检查配置或训练过程。")


if __name__ == "__main__":
    main()
