"""Model definitions and factory.

Includes the CBAM attention module carried over from the original experiment
scripts, wired onto a torchvision backbone. Unlike the legacy scripts, the
CBAM channel count is read from the backbone instead of being hard-coded, so
every EfficientNet / ResNet variant works.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "SimpleConvNet",
    "BackboneWithCBAM",
    "initialize_model",
    "available_models",
    "count_parameters",
]


# ----------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
# ----------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel attention using both average- and max-pooled descriptors.

    The original repo's implementation used average pooling only. The paper
    specifies a shared MLP applied to *both* pooled descriptors which are then
    summed; that is what is implemented here.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention over channel-wise mean/max maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class SimpleConvNet(nn.Module):
    """Small from-scratch CNN kept as a no-pretrained-weights baseline.

    Uses adaptive pooling so it accepts any input resolution; the legacy
    version hard-coded ``512 * 7 * 7`` and broke on non-224 inputs.
    """

    def __init__(self, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class BackboneWithCBAM(nn.Module):
    """torchvision backbone -> CBAM -> global pool -> linear head.

    ``feature_channels`` is inferred with a dry run, so this works for
    EfficientNet-B0..B7 and every ResNet without per-variant constants.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cbam = CBAM(feature_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.cbam(x)
        return self.head(self.pool(x))


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def _weights_for(builder_name: str, use_pretrained: bool):
    """Resolve the modern ``weights=`` enum, avoiding the removed ``pretrained=``.

    torchvision deprecated ``pretrained=`` in 0.13 and removed it in 0.15;
    the legacy code passed it directly and crashes on modern torchvision.
    """
    if not use_pretrained:
        return None
    try:
        return tvm.get_model_weights(builder_name).DEFAULT
    except Exception:  # pragma: no cover - very old torchvision
        return None


def _infer_channels(module: nn.Module, image_size: int = 224) -> int:
    """Run one dummy forward pass to discover the feature channel count."""
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            out = module(torch.zeros(1, 3, image_size, image_size))
    finally:
        module.train(was_training)
    if out.ndim != 4:
        raise RuntimeError(
            f"expected 4D feature map from backbone, got shape {tuple(out.shape)}"
        )
    return int(out.shape[1])


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    """Swap the final linear layer of a torchvision classifier for ``num_classes``."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = _head(in_features, num_classes, dropout)
        return model

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        model.classifier = _head(classifier.in_features, num_classes, dropout)
        return model

    if isinstance(classifier, nn.Sequential):
        for idx in reversed(range(len(classifier))):
            if isinstance(classifier[idx], nn.Linear):
                classifier[idx] = _head(
                    classifier[idx].in_features, num_classes, dropout
                )
                return model
            if isinstance(classifier[idx], nn.Conv2d):  # squeezenet
                conv = classifier[idx]
                classifier[idx] = nn.Conv2d(
                    conv.in_channels, num_classes, kernel_size=1
                )
                if hasattr(model, "num_classes"):
                    model.num_classes = num_classes
                return model

    raise RuntimeError(f"cannot locate classifier head on {type(model).__name__}")


def _head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def _freeze_except_head(model: nn.Module) -> None:
    """Freeze everything, then unfreeze the classifier head (linear probing)."""
    for param in model.parameters():
        param.requires_grad = False

    head: nn.Module | None = None
    if hasattr(model, "head"):
        head = model.head
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier

    if head is None:
        raise RuntimeError("linear_probe=True but no head found to unfreeze")
    for param in head.parameters():
        param.requires_grad = True


# name -> (torchvision builder, default input size)
_TORCHVISION_MODELS: dict[str, tuple[str, int]] = {
    "resnet18": ("resnet18", 224),
    "resnet34": ("resnet34", 224),
    "resnet50": ("resnet50", 224),
    "resnet101": ("resnet101", 224),
    "alexnet": ("alexnet", 224),
    "vgg11_bn": ("vgg11_bn", 224),
    "vgg16_bn": ("vgg16_bn", 224),
    "squeezenet": ("squeezenet1_0", 224),
    "densenet121": ("densenet121", 224),
    "googlenet": ("googlenet", 224),
    "efficientnet_b0": ("efficientnet_b0", 224),
    "efficientnet_b1": ("efficientnet_b1", 240),
    "efficientnet_b2": ("efficientnet_b2", 260),
    "efficientnet_b3": ("efficientnet_b3", 300),
    "efficientnet_b4": ("efficientnet_b4", 380),
    "mobilenet_v3_large": ("mobilenet_v3_large", 224),
    "convnext_tiny": ("convnext_tiny", 224),
}

# CBAM variants: <backbone>_cbam
_CBAM_MODELS = {
    f"{name}_cbam": (builder, size)
    for name, (builder, size) in _TORCHVISION_MODELS.items()
    if name.startswith(("resnet", "efficientnet", "densenet", "convnext"))
}


def available_models() -> list[str]:
    """Sorted list of every accepted ``model_name``."""
    return sorted({"simple_cnn", *_TORCHVISION_MODELS, *_CBAM_MODELS})


def _feature_extractor(builder_name: str, weights) -> nn.Module:
    """Return the convolutional trunk of a torchvision model (no classifier)."""
    model = getattr(tvm, builder_name)(weights=weights)
    if hasattr(model, "features"):  # efficientnet / densenet / vgg / convnext
        return model.features
    # resnet family: drop avgpool + fc
    children = list(model.children())
    return nn.Sequential(*children[:-2])


def initialize_model(
    model_name: str,
    num_classes: int,
    *,
    linear_probe: bool = False,
    use_pretrained: bool = True,
    dropout: float = 0.0,
) -> tuple[nn.Module, int]:
    """Build a model by name.

    Returns ``(model, recommended_input_size)``.

    Raises:
        ValueError: if ``model_name`` is not in :func:`available_models`.
    """
    key = model_name.strip().lower()

    if key in {"simple_cnn", "mymodel"}:
        model = SimpleConvNet(num_classes, dropout=dropout)
        if linear_probe:
            _freeze_except_head(model)
        return model, 224

    if key in _CBAM_MODELS:
        builder_name, input_size = _CBAM_MODELS[key]
        weights = _weights_for(builder_name, use_pretrained)
        trunk = _feature_extractor(builder_name, weights)
        channels = _infer_channels(trunk, input_size)
        model = BackboneWithCBAM(trunk, channels, num_classes, dropout=dropout)
        if linear_probe:
            _freeze_except_head(model)
        return model, input_size

    if key in _TORCHVISION_MODELS:
        builder_name, input_size = _TORCHVISION_MODELS[key]
        weights = _weights_for(builder_name, use_pretrained)
        kwargs = {}
        if builder_name == "googlenet" and weights is None:
            kwargs["init_weights"] = True
        model = getattr(tvm, builder_name)(weights=weights, **kwargs)
        if linear_probe:
            model = _replace_classifier(model, num_classes, dropout)
            _freeze_except_head(model)
        else:
            model = _replace_classifier(model, num_classes, dropout)
        return model, input_size

    raise ValueError(
        f"unsupported model_name {model_name!r}. "
        f"Available: {', '.join(available_models())}"
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
