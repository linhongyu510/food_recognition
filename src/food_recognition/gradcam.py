"""Grad-CAM: gradient-weighted class activation mapping.

Implements Selvaraju et al., ICCV 2017. Ported from the original
``experiments/legacy/model3_1.py`` with three changes:

* ``register_full_backward_hook`` instead of ``register_backward_hook``, which
  PyTorch documents as "deprecated in favor of register_full_backward_hook and
  the behavior of this function will change in future versions".
* Bilinear upsampling via ``torch.nn.functional.interpolate`` instead of
  ``cv2.resize``, dropping the OpenCV dependency entirely.
* The target layer is resolved automatically per architecture instead of being
  passed in by hand, and hooks are always removed via ``try/finally``.

The colormap is computed with numpy, so matplotlib is only needed if you want
to render figures yourself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD, _load_rgb, build_transform
from .models import BackboneWithCBAM, SimpleConvNet

__all__ = ["GradCAMResult", "GradCAM", "resolve_target_layer", "overlay_heatmap"]

logger = logging.getLogger(__name__)


@dataclass
class GradCAMResult:
    """A Grad-CAM heatmap and the prediction it explains."""

    heatmap: np.ndarray
    """Normalised to [0, 1], shape ``(H, W)`` matching the model input size."""

    class_index: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "heatmap_shape": list(self.heatmap.shape),
        }


def resolve_target_layer(model: nn.Module) -> nn.Module:
    """Pick the last convolutional stage to attribute against.

    Grad-CAM needs a layer whose activations retain spatial structure. The
    deepest conv stage gives the best trade-off between semantics and
    resolution.

    Raises:
        RuntimeError: if no 4D-output layer can be located.
    """
    if isinstance(model, BackboneWithCBAM):
        # Attribute after attention, so the map reflects what CBAM emphasised.
        return model.cbam
    if isinstance(model, SimpleConvNet):
        return model.features

    # torchvision ResNet family
    if hasattr(model, "layer4"):
        return model.layer4
    # efficientnet / densenet / vgg / convnext
    if hasattr(model, "features"):
        return model.features
    # googlenet
    if hasattr(model, "inception5b"):
        return model.inception5b

    # Fall back to the last Conv2d in the graph.
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if conv_layers:
        return conv_layers[-1]

    raise RuntimeError(
        f"cannot resolve a Grad-CAM target layer for {type(model).__name__}; "
        "pass target_layer= explicitly"
    )


class GradCAM:
    """Produce Grad-CAM heatmaps for a trained classifier.

    Example::

        cam = GradCAM(predictor.model, classes=predictor.classes)
        result = cam.generate(image_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_layer: nn.Module | None = None,
        classes: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer = target_layer or resolve_target_layer(model)
        self.classes = list(classes) if classes else None

    def _label(self, index: int) -> str:
        if self.classes and 0 <= index < len(self.classes):
            return self.classes[index]
        return str(index)

    def generate(
        self,
        image: torch.Tensor,
        *,
        class_index: int | None = None,
    ) -> GradCAMResult:
        """Compute a heatmap for ``image``.

        Args:
            image: a ``(3, H, W)`` or ``(1, 3, H, W)`` normalised tensor.
            class_index: class to explain. Defaults to the predicted class.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.size(0) != 1:
            raise ValueError(
                f"expected a single image of shape (3, H, W) or (1, 3, H, W), "
                f"got {tuple(image.shape)}"
            )

        image = image.to(self.device)
        # Without this, PyTorch warns that the full backward hook fires only
        # w.r.t. module outputs because no input requires grad. Grad-CAM only
        # needs the output gradients, but requiring grad on the input keeps the
        # backward graph complete and silences a warning users cannot act on.
        image = image.detach().requires_grad_(True)

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        handles = [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

        was_training = self.model.training
        self.model.eval()

        try:
            # Gradients are required here, so no torch.no_grad().
            logits = self.model(image)
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]

            probabilities = torch.softmax(logits, dim=1)
            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())
            elif not 0 <= class_index < logits.size(1):
                raise ValueError(
                    f"class_index {class_index} outside [0, {logits.size(1) - 1}]"
                )

            confidence = float(probabilities[0, class_index].item())

            self.model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        if not activations or not gradients:
            raise RuntimeError(
                f"hooks on {type(self.target_layer).__name__} captured nothing; "
                "the target layer may not be part of the forward pass"
            )

        activation = activations[0].detach()[0]
        gradient = gradients[0].detach()[0]

        if activation.ndim != 3:
            raise RuntimeError(
                f"Grad-CAM needs a 4D feature map, but the target layer output "
                f"{activation.ndim + 1}D. Choose a convolutional layer."
            )

        # Channel weights = spatially averaged gradients, then weighted sum.
        weights = gradient.mean(dim=(1, 2))
        cam = torch.einsum("c,chw->hw", weights, activation)
        cam = F.relu(cam)  # keep only evidence *for* the class

        cam = F.interpolate(
            cam[None, None],
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        # Normalise to [0, 1]; a flat map means no localised evidence.
        cam_min = float(cam.min().item())
        cam_max = float(cam.max().item())
        if cam_max - cam_min < 1e-12:
            logger.warning(
                "Grad-CAM produced a uniform map for class %d; "
                "the model may not be using localised features here", class_index
            )
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return GradCAMResult(
            heatmap=cam.cpu().numpy(),
            class_index=class_index,
            class_name=self._label(class_index),
            confidence=confidence,
        )

    def generate_from_path(
        self,
        path: Path | str,
        *,
        image_size: int = 224,
        class_index: int | None = None,
    ) -> tuple[GradCAMResult, Image.Image]:
        """Load an image from disk and return its heatmap plus the resized RGB."""
        transform = build_transform(image_size, is_train=False)
        pil_image = _load_rgb(path)
        tensor = transform(pil_image)
        result = self.generate(tensor, class_index=class_index)

        # Match the eval transform's resize+crop so the overlay lines up.
        display = _resize_and_center_crop(pil_image, image_size)
        return result, display


def _resize_and_center_crop(image: Image.Image, size: int) -> Image.Image:
    """Mirror build_transform's eval geometry (resize shorter side, then crop)."""
    target_short = int(round(size * 1.14))
    width, height = image.size
    scale = target_short / min(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.BILINEAR,
    )

    new_width, new_height = resized.size
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map [0, 1] to a jet-like RGB array without importing matplotlib."""
    v = np.clip(values, 0.0, 1.0)
    four = 4.0 * v
    red = np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0)
    green = np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0)
    blue = np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.5,
) -> Image.Image:
    """Blend a heatmap over an image.

    Args:
        image: RGB base image.
        heatmap: values in [0, 1]; resized to the image if shapes differ.
        alpha: heatmap opacity, 0 (invisible) to 1 (opaque).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if heatmap.shape != (image.height, image.width):
        heatmap = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image.width, image.height), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

    coloured = (_jet_colormap(heatmap) * 255).astype(np.uint8)
    base = np.array(image.convert("RGB"), dtype=np.float32)
    blended = (1 - alpha) * base + alpha * coloured.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def denormalize(tensor: torch.Tensor) -> Image.Image:
    """Invert ImageNet normalisation and return a viewable PIL image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    array = (restored.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)
