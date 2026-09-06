"""Training loop with validation, early stopping, LR scheduling and self-training."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data import (
    DataBundle,
    PseudoLabeledDataset,
    build_transform,
    create_dataloaders,
    generate_pseudo_labels,
)
from .metrics import ClassificationReport, compute_metrics
from .models import count_parameters, initialize_model
from .utils import (
    EarlyStopper,
    ensure_dir,
    format_duration,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_json,
)

__all__ = ["EpochRecord", "TrainingSummary", "Trainer", "train_model", "evaluate"]

logger = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    """Metrics for one epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    learning_rate: float
    duration_sec: float
    val_loss: float | None = None
    val_acc: float | None = None
    val_macro_f1: float | None = None
    pseudo_samples: int = 0
    is_best: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainingSummary:
    """Result of a training run."""

    history: list[EpochRecord] = field(default_factory=list)
    best_checkpoint: Path | None = None
    best_accuracy: float = 0.0
    best_epoch: int = 0
    final_report: ClassificationReport | None = None
    total_duration_sec: float = 0.0
    stopped_early: bool = False
    classes: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "total_duration_sec": self.total_duration_sec,
            "stopped_early": self.stopped_early,
            "classes": self.classes,
            "history": [record.to_dict() for record in self.history],
            "final_report": self.final_report.to_dict() if self.final_report else None,
        }


def _build_scheduler(
    cfg: TrainingConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create the LR scheduler named by ``cfg.scheduler``."""
    remaining = max(cfg.epochs - cfg.warmup_epochs, 1)

    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
        )
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.scheduler_gamma, patience=2
        )
    raise ValueError(f"unknown scheduler: {cfg.scheduler}")


class Trainer:
    """Coordinates data, model and the optimisation loop."""

    def __init__(self, cfg: TrainingConfig, bundle: DataBundle | None = None) -> None:
        cfg.validate()
        self.cfg = cfg

        seed_everything(cfg.seed, deterministic=cfg.deterministic)
        self.device = resolve_device(cfg.device)

        self.bundle = bundle if bundle is not None else create_dataloaders(cfg)
        self.classes = self.bundle.classes

        self.model, native_size = initialize_model(
            cfg.model_name,
            cfg.num_classes,
            linear_probe=cfg.linear_probe,
            use_pretrained=cfg.use_pretrained,
            dropout=cfg.dropout,
        )
        # Backbones like efficientnet_b3/b4 were designed for 300/380px. Training
        # them at the 224 default silently throws away most of what the extra
        # capacity is for, and the accuracy looks disappointing for no visible
        # reason. Warn rather than override: 224 is a legitimate choice when the
        # run has to fit a compute budget, but it should be deliberate.
        #
        # Gated on native_size > 224 so this only fires for backbones that really
        # do declare a higher resolution. simple_cnn reports 224 as a nominal
        # default and is routinely trained at 32px, which is not a mistake.
        if native_size > 224 and cfg.image_size < native_size * 0.9:
            logger.warning(
                "image_size=%d is well below the native %dpx for %s; "
                "expect to lose accuracy that the larger backbone would "
                "otherwise provide (set image_size=%d to use it fully)",
                cfg.image_size,
                native_size,
                cfg.model_name,
                native_size,
            )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "no trainable parameters; linear_probe may have frozen everything"
            )
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = _build_scheduler(cfg, self.optimizer)

        # AMP only helps on CUDA; enabling it on CPU/MPS slows things or errors.
        self.amp_enabled = bool(cfg.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.early_stopper = EarlyStopper(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode="max",
        )
        # Tracked separately from the early stopper, which applies min_delta.
        # See the checkpointing branch in fit() for why these must not share.
        self._best_val: float | None = None

        self.train_transform = build_transform(
            cfg.image_size, is_train=True, use_autoaugment=cfg.use_autoaugment
        )
        self.pseudo_loader: DataLoader | None = None
        self.summary = TrainingSummary(classes=self.classes)

        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.checkpoint_dir)

        logger.info("device=%s", self.device)
        logger.info(
            "model=%s trainable_params=%s",
            cfg.model_name,
            f"{count_parameters(self.model):,}",
        )
        logger.info("train samples=%d", self.bundle.num_train_samples)
        if self.bundle.val is not None:
            logger.info("val samples=%d", len(self.bundle.val.dataset))
        if self.amp_enabled:
            logger.info("mixed precision (AMP) enabled")

    # ------------------------------------------------------------------
    def train(self) -> TrainingSummary:
        """Run the full training loop and return a summary."""
        run_start = time.perf_counter()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start = time.perf_counter()
            self._apply_warmup(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_loss, train_acc = self._train_one_epoch(self.bundle.train)

            pseudo_samples = 0
            if self.pseudo_loader is not None:
                pseudo_samples = len(self.pseudo_loader.dataset)
                # Weight the two phases by sample count instead of a plain
                # mean, so the reported epoch metric is a true average.
                p_loss, p_acc = self._train_one_epoch(self.pseudo_loader)
                n_lab = self.bundle.num_train_samples
                total = n_lab + pseudo_samples
                train_loss = (train_loss * n_lab + p_loss * pseudo_samples) / total
                train_acc = (train_acc * n_lab + p_acc * pseudo_samples) / total

            val_loss: float | None = None
            val_acc: float | None = None
            val_f1: float | None = None
            is_best = False

            should_validate = self.bundle.val is not None and (
                epoch % self.cfg.val_every_n_epochs == 0 or epoch == self.cfg.epochs
            )

            if should_validate:
                report = self.evaluate(self.bundle.val)
                val_loss = report.loss
                val_acc = report.accuracy
                val_f1 = report.macro_f1

                # Checkpoint on any strict improvement, and keep min_delta for
                # patience only. Sharing one threshold between the two means a
                # genuinely better model gets thrown away: on the Food-101 B4
                # run, epoch 27 beat the saved best by 0.000238 against a
                # min_delta of 0.0005, so best.pt kept the weaker epoch-24
                # weights and metrics.json disagreed with the history it was
                # written beside.
                improved = self._best_val is None or val_acc > self._best_val
                # Still consulted, so early-stopping behaviour is unchanged: it
                # is the thing min_delta was added for.
                self.early_stopper.update(val_acc, epoch)

                is_best = improved
                if is_best:
                    self._best_val = val_acc
                    self.summary.best_accuracy = val_acc
                    self.summary.best_epoch = epoch
                    self.summary.final_report = report
                    self.summary.best_checkpoint = save_checkpoint(
                        self.cfg.checkpoint_path(),
                        self.model,
                        epoch=epoch,
                        metrics=report.to_dict(),
                        config=self.cfg.to_dict(),
                        classes=self.classes,
                    )

                self._maybe_refresh_pseudo_labels(epoch, val_acc)

            self._step_scheduler(epoch, val_acc)

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                learning_rate=current_lr,
                duration_sec=time.perf_counter() - epoch_start,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_f1,
                pseudo_samples=pseudo_samples,
                is_best=is_best,
            )
            self.summary.history.append(record)
            self._log_epoch(record)

            if self.cfg.save_last:
                save_checkpoint(
                    self.cfg.last_checkpoint_path(),
                    self.model,
                    epoch=epoch,
                    metrics=record.to_dict(),
                    config=self.cfg.to_dict(),
                    classes=self.classes,
                )

            if self.cfg.early_stopping.enabled and self.early_stopper.should_stop:
                logger.info(
                    "early stopping at epoch %d (no improvement for %d evaluations; "
                    "best=%.4f @ epoch %d)",
                    epoch,
                    self.early_stopper.patience,
                    self.early_stopper.best or 0.0,
                    self.early_stopper.best_epoch,
                )
                self.summary.stopped_early = True
                break

        self.summary.total_duration_sec = time.perf_counter() - run_start
        self._persist()
        return self.summary

    # ------------------------------------------------------------------
    def _apply_warmup(self, epoch: int) -> None:
        """Linearly ramp LR over the first ``warmup_epochs`` epochs."""
        if self.cfg.warmup_epochs <= 0 or epoch > self.cfg.warmup_epochs:
            return
        scale = epoch / (self.cfg.warmup_epochs + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.learning_rate * scale

    def _step_scheduler(self, epoch: int, val_acc: float | None) -> None:
        if self.scheduler is None or epoch <= self.cfg.warmup_epochs:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_acc is not None:
                self.scheduler.step(val_acc)
        else:
            self.scheduler.step()

    def _train_one_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self.model(images)
                # inception_v3 returns a namedtuple in train mode
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)

            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                self.optimizer.step()

            batch = images.size(0)
            running_loss += loss.item() * batch
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            seen += batch

        if seen == 0:
            raise RuntimeError("training loader yielded no samples")
        return running_loss / seen, correct / seen

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> ClassificationReport:
        """Evaluate on ``loader`` (defaults to the validation loader)."""
        loader = loader if loader is not None else self.bundle.val
        if loader is None:
            raise ValueError("no evaluation loader available")

        self.model.eval()
        total_loss = 0.0
        seen = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())

        if seen == 0:
            raise RuntimeError("evaluation loader yielded no samples")

        return compute_metrics(
            torch.cat(all_targets),
            torch.cat(all_preds),
            self.cfg.num_classes,
            class_names=self.classes,
            loss=total_loss / seen,
        )

    # ------------------------------------------------------------------
    def _maybe_refresh_pseudo_labels(self, epoch: int, val_acc: float) -> None:
        """Regenerate pseudo-labels when the configured gates are satisfied."""
        semi = self.cfg.semi_supervised
        if not semi.enabled or self.bundle.unlabeled is None:
            return
        if val_acc < semi.activation_threshold:
            return
        if epoch % semi.refresh_interval != 0:
            return

        cap = int(self.bundle.num_train_samples * semi.max_ratio)
        samples = generate_pseudo_labels(
            self.model,
            self.bundle.unlabeled,
            self.device,
            semi.confidence_threshold,
            max_samples=cap,
        )

        if not samples:
            logger.info(
                "epoch %d: no unlabelled sample passed confidence >= %.2f",
                epoch,
                semi.confidence_threshold,
            )
            self.pseudo_loader = None
            return

        loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.cfg.num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        self.pseudo_loader = DataLoader(
            PseudoLabeledDataset(samples, transform=self.train_transform),
            shuffle=True,
            **loader_kwargs,
        )
        logger.info("epoch %d: accepted %d pseudo-labelled samples", epoch, len(samples))

    def _log_epoch(self, record: EpochRecord) -> None:
        parts = [
            f"epoch {record.epoch:3d}/{self.cfg.epochs}",
            f"loss {record.train_loss:.4f}",
            f"acc {record.train_acc:.4f}",
        ]
        if record.val_acc is not None:
            parts.append(f"val_loss {record.val_loss:.4f}")
            parts.append(f"val_acc {record.val_acc:.4f}")
            parts.append(f"val_f1 {record.val_macro_f1:.4f}")
        if record.pseudo_samples:
            parts.append(f"pseudo {record.pseudo_samples}")
        parts.append(f"lr {record.learning_rate:.2e}")
        parts.append(format_duration(record.duration_sec))
        if record.is_best:
            parts.append("<- best")
        logger.info(" | ".join(parts))

    def _persist(self) -> None:
        """Write history and the best report to the run directory."""
        write_json(
            self.cfg.history_path(),
            [record.to_dict() for record in self.summary.history],
        )
        if self.summary.final_report is not None:
            write_json(self.cfg.metrics_path(), self.summary.final_report.to_dict())


def train_model(cfg: TrainingConfig) -> TrainingSummary:
    """Convenience wrapper: build a :class:`Trainer` and run it."""
    return Trainer(cfg).train()


def evaluate(cfg: TrainingConfig, checkpoint: Path | str) -> ClassificationReport:
    """Load ``checkpoint`` and evaluate it against ``cfg``'s validation set."""
    from .utils import load_checkpoint

    trainer = Trainer(cfg)
    payload = load_checkpoint(checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(payload["model_state"])
    return trainer.evaluate()
