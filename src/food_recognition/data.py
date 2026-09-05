"""Datasets, transforms and dataloader construction.

Handles the two directory layouts that appear in this project:

* **Labelled** - class sub-directories, e.g. Food-11's ``00/``..``10/``.
  Numeric directory names are sorted *numerically*, not lexicographically, so
  ``10/`` maps to index 10 rather than landing between ``01/`` and ``02/``.
* **Unlabelled** - a flat directory of images with no class sub-directories.
  ``torchvision.datasets.ImageFolder`` raises on this layout, which is why the
  semi-supervised path in the original code could not run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import TrainingConfig

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_EXTENSIONS",
    "build_transform",
    "LabeledImageDataset",
    "UnlabeledImageDataset",
    "PseudoLabeledDataset",
    "DataBundle",
    "create_dataloaders",
    "generate_pseudo_labels",
]

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
def build_transform(
    image_size: int,
    *,
    is_train: bool,
    normalize: bool = True,
    use_autoaugment: bool = True,
) -> transforms.Compose:
    """Build the train or eval transform pipeline.

    Eval uses resize-shorter-side-then-center-crop (ratio 1.14, the standard
    256/224) which preserves aspect ratio, instead of the squashing
    ``Resize((H, W))`` used by some of the legacy scripts.
    """
    ops: list[Callable] = []

    if is_train:
        ops.append(transforms.RandomResizedCrop(image_size))
        ops.append(transforms.RandomHorizontalFlip())
        if use_autoaugment:
            ops.append(transforms.AutoAugment())
    else:
        ops.append(transforms.Resize(int(round(image_size * 1.14))))
        ops.append(transforms.CenterCrop(image_size))

    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(ops)


def _sorted_class_dirs(root: Path) -> list[Path]:
    """Return class sub-directories, numeric names sorted numerically."""
    dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        return []
    if all(d.name.isdigit() for d in dirs):
        return sorted(dirs, key=lambda d: int(d.name))
    return sorted(dirs, key=lambda d: d.name)


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_rgb(path: Path | str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
class LabeledImageDataset(Dataset):
    """Images in class sub-directories.

    Attributes:
        classes: class directory names, ordered by assigned index.
        samples: ``(path, label)`` pairs.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.return_path = return_path

        if not self.root.exists():
            raise FileNotFoundError(f"data directory does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"expected a directory: {self.root}")

        class_dirs = _sorted_class_dirs(self.root)
        if not class_dirs:
            raise ValueError(
                f"no class sub-directories found under {self.root}. "
                "Labelled data must be organised as <root>/<class>/<image>."
            )

        self.classes: list[str] = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        samples: list[tuple[Path, int]] = []
        for index, directory in enumerate(class_dirs):
            files = _list_images(directory)
            if not files:
                logger.warning("class directory is empty: %s", directory)
            samples.extend((path, index) for path in files)

        if not samples:
            raise ValueError(f"no images found under {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = _load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(path)
        return image, label


class UnlabeledImageDataset(Dataset):
    """A flat directory of images, or class sub-directories, with no labels.

    Always yields ``(image, path)`` so predictions can be traced back to files.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        transform: Callable | None = None,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"data directory does not exist: {self.root}")

        if recursive:
            paths = sorted(
                p
                for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            paths = _list_images(self.root)

        if not paths:
            raise ValueError(f"no images found under {self.root}")
        self.samples: list[Path] = paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        image = _load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


class PseudoLabeledDataset(Dataset):
    """``(path, label)`` pairs produced by :func:`generate_pseudo_labels`."""

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        transform: Callable | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = _load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------------
# Dataloaders
# ----------------------------------------------------------------------------
@dataclass
class DataBundle:
    """Dataloaders plus the resolved class names."""

    train: DataLoader
    val: DataLoader | None = None
    unlabeled: DataLoader | None = None
    classes: list[str] | None = None

    @property
    def num_train_samples(self) -> int:
        return len(self.train.dataset)


def create_dataloaders(cfg: TrainingConfig) -> DataBundle:
    """Build train/val/unlabeled loaders from ``cfg``.

    Raises:
        ValueError: if the discovered class count disagrees with
            ``cfg.num_classes`` -- catching a silent shape mismatch that would
            otherwise surface as a confusing loss error mid-training.
    """
    train_transform = build_transform(
        cfg.image_size, is_train=True, use_autoaugment=cfg.use_autoaugment
    )
    eval_transform = build_transform(cfg.image_size, is_train=False)

    train_dataset = LabeledImageDataset(cfg.train_dir, transform=train_transform)

    if len(train_dataset.classes) != cfg.num_classes:
        raise ValueError(
            f"found {len(train_dataset.classes)} class directories under "
            f"{cfg.train_dir} ({train_dataset.classes}) but num_classes="
            f"{cfg.num_classes}. Fix num_classes or the data layout."
        )

    # persistent_workers requires num_workers > 0
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **loader_kwargs
    )

    val_loader: DataLoader | None = None
    if cfg.val_dir is not None:
        if cfg.val_dir.exists():
            val_dataset = LabeledImageDataset(cfg.val_dir, transform=eval_transform)
            if val_dataset.classes != train_dataset.classes:
                logger.warning(
                    "validation classes %s differ from training classes %s",
                    val_dataset.classes,
                    train_dataset.classes,
                )
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        else:
            logger.warning("val_dir does not exist, skipping validation: %s", cfg.val_dir)

    unlabeled_loader: DataLoader | None = None
    if cfg.unlabeled_dir is not None:
        if cfg.unlabeled_dir.exists():
            unlabeled_dataset = UnlabeledImageDataset(
                cfg.unlabeled_dir, transform=eval_transform
            )
            unlabeled_loader = DataLoader(
                unlabeled_dataset, shuffle=False, **loader_kwargs
            )
        elif cfg.semi_supervised.enabled:
            raise FileNotFoundError(
                f"semi_supervised.enabled=true but unlabeled_dir does not exist: "
                f"{cfg.unlabeled_dir}"
            )

    classes = cfg.class_names or train_dataset.classes
    return DataBundle(
        train=train_loader,
        val=val_loader,
        unlabeled=unlabeled_loader,
        classes=classes,
    )


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_threshold: float,
    max_samples: int | None = None,
) -> list[tuple[str, int]]:
    """Predict labels for unlabelled data, keeping only confident predictions.

    When ``max_samples`` is set, the highest-confidence predictions are kept.
    """
    model.eval()
    scored: list[tuple[float, str, int]] = []

    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1)
        confs, preds = probs.max(dim=1)
        for conf, pred, path in zip(confs.tolist(), preds.tolist(), paths):
            if conf >= confidence_threshold:
                scored.append((conf, path, int(pred)))

    if max_samples is not None and len(scored) > max_samples:
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = scored[:max_samples]

    return [(path, label) for _, path, label in scored]
