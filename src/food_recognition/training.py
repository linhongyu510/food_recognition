"""训练流程封装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data import (
    DataLoaders,
    PseudoLabeledDataset,
    build_transform,
    create_dataloaders,
    generate_pseudo_labels,
)
from .models import initialize_model
from .utils import ensure_dir, resolve_device, seed_everything


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    pseudo_samples: int = 0


@dataclass
class TrainingSummary:
    metrics: List[EpochMetrics] = field(default_factory=list)
    best_checkpoint: Optional[Path] = None
    best_accuracy: float = 0.0


class Trainer:
    """训练器封装，负责协调数据、模型与训练流程。"""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = resolve_device(cfg.device)

        self.model, _ = initialize_model(
            cfg.model_name,
            cfg.num_classes,
            linear_probe=cfg.linear_probe,
            use_pretrained=cfg.use_pretrained,
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        ensure_dir(cfg.checkpoint_dir)

        self.dataloaders: DataLoaders = create_dataloaders(cfg)
        self.train_transform = build_transform(cfg.image_size, is_train=True)
        self.pseudo_loader: Optional[DataLoader] = None

        self.summary = TrainingSummary()
        self.best_acc = 0.0

    def train(self) -> TrainingSummary:
        """执行完整训练流程。"""
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self._train_one_epoch(self.dataloaders.train)

            val_loss = None
            val_acc = None

            if (
                self.dataloaders.val is not None
                and (epoch % self.cfg.val_every_n_epochs == 0 or epoch == self.cfg.epochs)
            ):
                val_loss, val_acc = self._evaluate(self.dataloaders.val)
                if val_acc is not None and val_acc > self.best_acc:
                    self._save_checkpoint(epoch, val_acc)

                if (
                    self.cfg.semi_supervised.enabled
                    and val_acc is not None
                    and val_acc >= self.cfg.semi_supervised.activation_threshold
                    and self.dataloaders.unlabeled is not None
                    and epoch % self.cfg.semi_supervised.refresh_interval == 0
                ):
                    self._refresh_pseudo_loader()

            pseudo_samples = 0
            if self.pseudo_loader is not None:
                pseudo_samples = len(self.pseudo_loader.dataset)
                pseudo_loss, pseudo_acc = self._train_one_epoch(self.pseudo_loader)
                train_loss = (train_loss + pseudo_loss) / 2
                train_acc = (train_acc + pseudo_acc) / 2

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                pseudo_samples=pseudo_samples,
            )
            self.summary.metrics.append(metrics)

            self._log_epoch(metrics)

        self.summary.best_accuracy = self.best_acc
        self.summary.best_checkpoint = (
            self.cfg.checkpoint_path() if self.best_acc > 0 else None
        )
        return self.summary

    def _train_one_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        return avg_loss, avg_acc

    def _refresh_pseudo_loader(self) -> None:
        assert self.dataloaders.unlabeled is not None
        samples = generate_pseudo_labels(
            self.model,
            self.dataloaders.unlabeled,
            self.device,
            self.cfg.semi_supervised.confidence_threshold,
        )
        if not samples:
            self.pseudo_loader = None
            return

        pseudo_dataset = PseudoLabeledDataset(samples, transform=self.train_transform)
        self.pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        self.best_acc = val_acc
        checkpoint_path = self.cfg.checkpoint_path()
        torch.save(self.model.state_dict(), checkpoint_path)
        self.summary.best_checkpoint = checkpoint_path
        self.summary.best_accuracy = val_acc

    def _log_epoch(self, metrics: EpochMetrics) -> None:
        log = (
            f"[Epoch {metrics.epoch:03d}] "
            f"train_loss={metrics.train_loss:.4f} "
            f"train_acc={metrics.train_acc:.4f}"
        )
        if metrics.val_loss is not None and metrics.val_acc is not None:
            log += (
                f" | val_loss={metrics.val_loss:.4f} "
                f"val_acc={metrics.val_acc:.4f}"
            )
        if metrics.pseudo_samples:
            log += f" | pseudo_samples={metrics.pseudo_samples}"
        print(log)


def train_model(cfg: TrainingConfig) -> TrainingSummary:
    """便捷函数，直接根据配置执行训练并返回结果。"""
    trainer = Trainer(cfg)
    return trainer.train()

