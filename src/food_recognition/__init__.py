"""food_recognition - food image classification with PyTorch.

Quick start::

    from food_recognition import TrainingConfig, train_model

    cfg = TrainingConfig(model_name="resnet18", num_classes=11, epochs=5)
    summary = train_model(cfg)
    print(summary.best_accuracy)

Command line::

    food-recognition-train --config configs/food11_resnet18.yaml
    food-recognition-eval  --checkpoint runs/exp/checkpoints/best.pt
    food-recognition-predict --checkpoint runs/exp/checkpoints/best.pt --input img.jpg
"""

from __future__ import annotations

__version__ = "0.7.0"

from .config import (
    EarlyStoppingConfig,
    SemiSupervisedConfig,
    TrainingConfig,
    dump_training_config,
    load_training_config,
)
from .data import (
    DataBundle,
    LabeledImageDataset,
    PseudoLabeledDataset,
    UnlabeledImageDataset,
    build_transform,
    create_dataloaders,
)
from .gradcam import GradCAM, GradCAMResult, overlay_heatmap, resolve_target_layer
from .metrics import ClassificationReport, compute_metrics, confusion_matrix
from .models import CBAM, BackboneWithCBAM, SimpleConvNet, available_models, initialize_model
from .predict import Prediction, Predictor, load_predictor
from .training import EpochRecord, Trainer, TrainingSummary, train_model
from .utils import EarlyStopper, resolve_device, seed_everything

__all__ = [
    "__version__",
    # config
    "TrainingConfig",
    "SemiSupervisedConfig",
    "EarlyStoppingConfig",
    "load_training_config",
    "dump_training_config",
    # data
    "LabeledImageDataset",
    "UnlabeledImageDataset",
    "PseudoLabeledDataset",
    "DataBundle",
    "build_transform",
    "create_dataloaders",
    # models
    "CBAM",
    "BackboneWithCBAM",
    "SimpleConvNet",
    "initialize_model",
    "available_models",
    # metrics
    "ClassificationReport",
    "compute_metrics",
    "confusion_matrix",
    # explainability
    "GradCAM",
    "GradCAMResult",
    "overlay_heatmap",
    "resolve_target_layer",
    # training
    "Trainer",
    "TrainingSummary",
    "EpochRecord",
    "train_model",
    # inference
    "Predictor",
    "Prediction",
    "load_predictor",
    # utils
    "seed_everything",
    "resolve_device",
    "EarlyStopper",
]
