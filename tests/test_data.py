"""Tests for datasets and dataloaders, including the previously broken paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from food_recognition.config import TrainingConfig
from food_recognition.data import (
    LabeledImageDataset,
    UnlabeledImageDataset,
    build_transform,
    create_dataloaders,
    generate_pseudo_labels,
)


def test_labeled_dataset_reads_class_dirs(sample_dataset: Path):
    dataset = LabeledImageDataset(sample_dataset / "training/labeled")

    assert dataset.classes == ["00", "01", "02"]
    assert len(dataset) == 18
    labels = sorted({label for _, label in dataset.samples})
    assert labels == [0, 1, 2]


def test_labeled_dataset_applies_transform(sample_dataset: Path):
    transform = build_transform(32, is_train=False)
    dataset = LabeledImageDataset(
        sample_dataset / "training/labeled", transform=transform
    )
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_labeled_dataset_can_return_paths(sample_dataset: Path):
    dataset = LabeledImageDataset(
        sample_dataset / "training/labeled",
        transform=build_transform(32, is_train=False),
        return_path=True,
    )
    _, _, path = dataset[0]
    assert Path(path).exists()


def test_numeric_class_dirs_sort_numerically(nonpadded_dataset: Path):
    """Regression test: dirs 0..10 must map 10 -> index 10, not index 2.

    torchvision's ImageFolder sorts lexicographically, so '10' lands between
    '1' and '2'. That silently shifts every label above 1.
    """
    dataset = LabeledImageDataset(nonpadded_dataset)

    assert dataset.classes == [str(i) for i in range(11)]
    assert dataset.class_to_idx["10"] == 10
    assert dataset.class_to_idx["2"] == 2


def test_unlabeled_dataset_handles_flat_directory(sample_dataset: Path):
    """Regression test: a flat dir must work.

    ImageFolder raises FileNotFoundError('Couldn't find any class folder')
    here, which is what prevented the semi-supervised path from ever running.
    """
    dataset = UnlabeledImageDataset(
        sample_dataset / "training/unlabeled",
        transform=build_transform(32, is_train=False),
    )

    assert len(dataset) == 4
    image, path = dataset[0]
    assert image.shape == (3, 32, 32)
    assert Path(path).exists()


def test_missing_directory_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LabeledImageDataset(tmp_path / "does_not_exist")


def test_directory_without_class_subdirs_raises(tmp_path: Path):
    empty = tmp_path / "flat"
    empty.mkdir()
    (empty / "a.jpg").write_bytes(b"not-an-image")

    with pytest.raises(ValueError, match="no class sub-directories"):
        LabeledImageDataset(empty)


def test_eval_transform_preserves_aspect_ratio():
    transform = build_transform(64, is_train=False)
    names = [type(op).__name__ for op in transform.transforms]
    assert "Resize" in names
    assert "CenterCrop" in names
    assert "Normalize" in names


def test_train_transform_includes_augmentation():
    with_aug = build_transform(64, is_train=True, use_autoaugment=True)
    without = build_transform(64, is_train=True, use_autoaugment=False)

    assert "AutoAugment" in [type(t).__name__ for t in with_aug.transforms]
    assert "AutoAugment" not in [type(t).__name__ for t in without.transforms]


def test_create_dataloaders_builds_all_three_loaders(sample_dataset: Path):
    cfg = TrainingConfig(
        num_classes=3,
        train_dir=sample_dataset / "training/labeled",
        val_dir=sample_dataset / "validation",
        unlabeled_dir=sample_dataset / "training/unlabeled",
        image_size=32,
        batch_size=4,
        num_workers=0,
    )
    bundle = create_dataloaders(cfg)

    assert bundle.num_train_samples == 18
    assert bundle.val is not None and len(bundle.val.dataset) == 9
    assert bundle.unlabeled is not None and len(bundle.unlabeled.dataset) == 4
    assert bundle.classes == ["00", "01", "02"]


def test_class_count_mismatch_is_caught_early(sample_dataset: Path):
    """A wrong num_classes must fail loudly before training starts."""
    cfg = TrainingConfig(
        num_classes=11,  # data only has 3
        train_dir=sample_dataset / "training/labeled",
        val_dir=None,
        image_size=32,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="num_classes"):
        create_dataloaders(cfg)


def test_generate_pseudo_labels_filters_by_confidence(sample_dataset: Path):
    cfg = TrainingConfig(
        num_classes=3,
        train_dir=sample_dataset / "training/labeled",
        val_dir=None,
        unlabeled_dir=sample_dataset / "training/unlabeled",
        image_size=32,
        batch_size=4,
        num_workers=0,
    )
    bundle = create_dataloaders(cfg)

    class ConfidentModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.size(0), 3)
            out[:, 1] = 20.0  # softmax -> ~1.0 on class 1
            return out

    accepted = generate_pseudo_labels(
        ConfidentModel(), bundle.unlabeled, torch.device("cpu"), 0.9
    )
    assert len(accepted) == 4
    assert all(label == 1 for _, label in accepted)

    class UncertainModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 3)  # uniform -> 0.333

    rejected = generate_pseudo_labels(
        UncertainModel(), bundle.unlabeled, torch.device("cpu"), 0.9
    )
    assert rejected == []


def test_generate_pseudo_labels_respects_max_samples(sample_dataset: Path):
    cfg = TrainingConfig(
        num_classes=3,
        train_dir=sample_dataset / "training/labeled",
        val_dir=None,
        unlabeled_dir=sample_dataset / "training/unlabeled",
        image_size=32,
        batch_size=4,
        num_workers=0,
    )
    bundle = create_dataloaders(cfg)

    class ConfidentModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.size(0), 3)
            out[:, 0] = 20.0
            return out

    accepted = generate_pseudo_labels(
        ConfidentModel(), bundle.unlabeled, torch.device("cpu"), 0.5, max_samples=2
    )
    assert len(accepted) == 2
