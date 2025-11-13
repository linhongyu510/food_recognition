"""数据集与数据加载相关的工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import TrainingConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    image_size: int,
    *,
    is_train: bool,
    normalize: bool = True,
    use_autoaugment: bool = True,
) -> transforms.Compose:
    """根据用途构建 transform。"""
    t: List[Callable] = []

    if is_train:
        t.append(transforms.RandomResizedCrop(image_size))
        t.append(transforms.RandomHorizontalFlip())
        if use_autoaugment:
            t.append(transforms.AutoAugment())
    else:
        t.append(transforms.Resize(int(image_size * 1.14)))
        t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())

    if normalize:
        t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(t)


class FoodImageDataset(Dataset):
    """基于 ImageFolder 的数据集封装，支持返回文件路径。"""

    def __init__(
        self,
        root: Path,
        *,
        transform: Optional[Callable] = None,
        is_labeled: bool = True,
        return_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.is_labeled = is_labeled
        self.return_path = return_path
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        # ImageFolder 需要目录结构遵循 class -> images
        self.dataset = datasets.ImageFolder(str(self.root), transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        path, _ = self.dataset.samples[index]

        if self.is_labeled:
            if self.return_path:
                return image, label, path
            return image, label

        if self.return_path:
            return image, path
        return image


@dataclass
class DataLoaders:
    """统一保存训练/验证以及无标签数据的 DataLoader。"""

    train: DataLoader
    val: Optional[DataLoader] = None
    unlabeled: Optional[DataLoader] = None


def create_dataloaders(cfg: TrainingConfig) -> DataLoaders:
    """根据配置构建 DataLoader。"""
    train_transform = build_transform(cfg.image_size, is_train=True)
    eval_transform = build_transform(cfg.image_size, is_train=False)

    train_dataset = FoodImageDataset(
        cfg.train_dir, transform=train_transform, is_labeled=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader: Optional[DataLoader] = None
    if cfg.val_dir and cfg.val_dir.exists():
        val_dataset = FoodImageDataset(
            cfg.val_dir, transform=eval_transform, is_labeled=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    unlabeled_loader: Optional[DataLoader] = None
    if cfg.unlabeled_dir and cfg.unlabeled_dir.exists():
        unlabeled_dataset = FoodImageDataset(
            cfg.unlabeled_dir,
            transform=eval_transform,
            is_labeled=False,
            return_path=True,
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return DataLoaders(train=train_loader, val=val_loader, unlabeled=unlabeled_loader)


class PseudoLabeledDataset(Dataset):
    """通过模型预测得到伪标签的数据集。"""

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label


def generate_pseudo_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_threshold: float,
) -> List[Tuple[str, int]]:
    """使用当前模型为无标签数据生成伪标签，返回 (path, label) 列表。"""
    model.eval()
    accepted: List[Tuple[str, int]] = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in loader:
            images, paths = batch
            images = images.to(device)
            logits = model(images)
            probs = softmax(logits)
            confs, preds = probs.max(dim=1)

            for conf, pred, path in zip(confs.cpu().tolist(), preds.cpu().tolist(), paths):
                if conf >= confidence_threshold:
                    accepted.append((path, int(pred)))

    return accepted

