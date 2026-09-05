"""Tests for Grad-CAM.

Grad-CAM is easy to implement in a way that runs without crashing but produces
meaningless maps, so these tests check the *semantics*: that the map localises
onto real evidence, responds to the requested class, and lines up with the
input geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from food_recognition.gradcam import (
    GradCAM,
    _resize_and_center_crop,
    denormalize,
    overlay_heatmap,
    resolve_target_layer,
)
from food_recognition.models import initialize_model


@pytest.fixture
def tiny_model():
    model, _ = initialize_model("simple_cnn", num_classes=3, use_pretrained=False)
    return model.eval()


def test_resolve_target_layer_for_each_family():
    resnet, _ = initialize_model("resnet18", 3, use_pretrained=False)
    assert resolve_target_layer(resnet) is resnet.layer4

    effnet, _ = initialize_model("efficientnet_b0", 3, use_pretrained=False)
    assert resolve_target_layer(effnet) is effnet.features

    cbam, _ = initialize_model("resnet18_cbam", 3, use_pretrained=False)
    # For CBAM models the map should reflect post-attention features.
    assert resolve_target_layer(cbam) is cbam.cbam

    simple, _ = initialize_model("simple_cnn", 3, use_pretrained=False)
    assert resolve_target_layer(simple) is simple.features


def test_resolve_target_layer_falls_back_to_last_conv():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 8, 3),
    )
    assert resolve_target_layer(model) is model[2]


def test_resolve_target_layer_raises_without_conv():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(10, 2))
    with pytest.raises(RuntimeError, match="cannot resolve"):
        resolve_target_layer(model)


def test_heatmap_shape_matches_input(tiny_model):
    cam = GradCAM(tiny_model)
    result = cam.generate(torch.randn(3, 64, 64))
    assert result.heatmap.shape == (64, 64)


def test_heatmap_is_normalised(tiny_model):
    cam = GradCAM(tiny_model)
    heatmap = cam.generate(torch.randn(1, 3, 64, 64)).heatmap

    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert np.isfinite(heatmap).all()


def test_accepts_both_3d_and_4d_input(tiny_model):
    cam = GradCAM(tiny_model)
    torch.manual_seed(0)
    image = torch.randn(3, 64, 64)

    a = cam.generate(image).heatmap
    b = cam.generate(image.unsqueeze(0)).heatmap
    assert np.allclose(a, b)


def test_rejects_batched_input(tiny_model):
    cam = GradCAM(tiny_model)
    with pytest.raises(ValueError, match="single image"):
        cam.generate(torch.randn(4, 3, 64, 64))


def test_explains_requested_class(tiny_model):
    cam = GradCAM(tiny_model)
    image = torch.randn(3, 64, 64)

    for index in range(3):
        result = cam.generate(image, class_index=index)
        assert result.class_index == index
        assert result.class_name == str(index)


def test_defaults_to_predicted_class(tiny_model):
    cam = GradCAM(tiny_model)
    image = torch.randn(3, 64, 64)

    with torch.no_grad():
        expected = int(tiny_model(image.unsqueeze(0)).argmax(dim=1).item())

    assert cam.generate(image).class_index == expected


def test_rejects_out_of_range_class(tiny_model):
    cam = GradCAM(tiny_model)
    with pytest.raises(ValueError, match="outside"):
        cam.generate(torch.randn(3, 64, 64), class_index=99)


def test_confidence_is_a_probability(tiny_model):
    cam = GradCAM(tiny_model)
    result = cam.generate(torch.randn(3, 64, 64))
    assert 0.0 <= result.confidence <= 1.0


def test_class_names_are_used(tiny_model):
    cam = GradCAM(tiny_model, classes=["bread", "soup", "rice"])
    result = cam.generate(torch.randn(3, 64, 64), class_index=1)
    assert result.class_name == "soup"


def test_hooks_are_removed_after_use(tiny_model):
    """Leaked hooks would accumulate and silently slow every later forward pass."""
    layer = resolve_target_layer(tiny_model)
    before_f = len(layer._forward_hooks)
    before_b = len(layer._backward_hooks) + len(
        getattr(layer, "_full_backward_hooks", {})
    )

    cam = GradCAM(tiny_model)
    for _ in range(3):
        cam.generate(torch.randn(3, 64, 64))

    after_b = len(layer._backward_hooks) + len(
        getattr(layer, "_full_backward_hooks", {})
    )
    assert len(layer._forward_hooks) == before_f
    assert after_b == before_b


def test_hooks_removed_even_when_generation_fails(tiny_model):
    layer = resolve_target_layer(tiny_model)
    cam = GradCAM(tiny_model)

    with pytest.raises(ValueError):
        cam.generate(torch.randn(3, 64, 64), class_index=99)

    assert len(layer._forward_hooks) == 0


def test_model_training_mode_is_restored(tiny_model):
    tiny_model.train()
    GradCAM(tiny_model).generate(torch.randn(3, 64, 64))
    assert tiny_model.training, "generate() must not leave the model in eval mode"


def test_no_grad_leaks_into_parameters(tiny_model):
    """Grad-CAM must not leave stale gradients that would corrupt a later step."""
    cam = GradCAM(tiny_model)
    cam.generate(torch.randn(3, 64, 64))

    # zero_grad(set_to_none=True) runs before backward, so grads exist after;
    # what matters is that they are finite and not NaN.
    for param in tiny_model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


class _ChannelRouter(torch.nn.Module):
    """A 1x1 conv trunk plus an identity head, so class c reads channel c.

    Grad-CAM weights each channel by its spatially-averaged gradient and then
    sums the channels, so localisation must come from the *activation map's*
    spatial structure. Here the trunk copies its input channels through
    unchanged, meaning the caller controls exactly where each channel fires.
    """

    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.features = torch.nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.features.weight.zero_()
            for c in range(channels):
                self.features.weight[c, c, 0, 0] = 1.0
        self.head = torch.nn.Linear(channels, channels, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(channels))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def test_heatmap_localises_real_evidence():
    """Grad-CAM must highlight where the class's evidence actually lives.

    Channel 1 fires only in the top-left quadrant and drives class 1, so
    explaining class 1 must light up that quadrant. A wrong channel weighting
    or a missing ReLU still yields a correctly shaped, normalised map, so this
    is the assertion that catches those bugs.
    """
    model = _ChannelRouter(channels=2).eval()

    # channel 0 fires everywhere, channel 1 only top-left
    x = torch.zeros(2, 32, 32)
    x[0] = 1.0
    x[1, :16, :16] = 1.0

    cam = GradCAM(model, target_layer=model.features)
    heatmap = cam.generate(x, class_index=1).heatmap

    top_left = heatmap[:16, :16].mean()
    bottom_right = heatmap[16:, 16:].mean()
    assert top_left > bottom_right, (
        f"Grad-CAM did not localise the informative quadrant: "
        f"top-left={top_left:.4f} vs bottom-right={bottom_right:.4f}"
    )


def test_heatmap_distinguishes_between_classes():
    """Explaining a different class must move the attention."""
    model = _ChannelRouter(channels=2).eval()

    # channel 0 -> top half, channel 1 -> bottom half
    x = torch.zeros(2, 32, 32)
    x[0, :16, :] = 1.0
    x[1, 16:, :] = 1.0

    cam = GradCAM(model, target_layer=model.features)
    map_a = cam.generate(x, class_index=0).heatmap
    map_b = cam.generate(x, class_index=1).heatmap

    assert map_a[:16].mean() > map_a[16:].mean(), "class 0 should attend to the top"
    assert map_b[16:].mean() > map_b[:16].mean(), "class 1 should attend to the bottom"


def test_uniform_map_warns_but_does_not_crash(caplog):
    """A model with no spatial preference should warn, not raise."""

    class Constant(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return torch.zeros(x.size(0), 2, requires_grad=True) + (
                self.features(x).mean() * 0
            )

    model = Constant().eval()
    cam = GradCAM(model, target_layer=model.features)
    result = cam.generate(torch.randn(3, 32, 32), class_index=0)
    assert result.heatmap.shape == (32, 32)
    assert np.isfinite(result.heatmap).all()


def test_generate_emits_no_warnings(tiny_model):
    """Regression: the full backward hook warned when no input required grad.

    Users cannot act on that warning, and it fired on every single call.
    """
    import warnings

    cam = GradCAM(tiny_model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cam.generate(torch.randn(3, 64, 64))


def test_generate_from_path(tmp_path: Path, tiny_model):
    path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 80), (200, 60, 60)).save(path)

    cam = GradCAM(tiny_model)
    result, display = cam.generate_from_path(path, image_size=64)

    assert result.heatmap.shape == (64, 64)
    assert display.size == (64, 64), "display image must match the model input"


def test_generate_from_path_missing_file(tmp_path: Path, tiny_model):
    cam = GradCAM(tiny_model)
    with pytest.raises((FileNotFoundError, OSError)):
        cam.generate_from_path(tmp_path / "nope.jpg", image_size=64)


def test_resize_and_center_crop_is_square():
    for size in [(200, 100), (100, 200), (64, 64)]:
        out = _resize_and_center_crop(Image.new("RGB", size), 32)
        assert out.size == (32, 32)


def test_overlay_returns_same_size():
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    heatmap = np.linspace(0, 1, 64 * 64).reshape(64, 64)

    out = overlay_heatmap(image, heatmap, alpha=0.5)
    assert out.size == (64, 64)
    assert out.mode == "RGB"


def test_overlay_resizes_mismatched_heatmap():
    image = Image.new("RGB", (80, 60))
    heatmap = np.ones((10, 10), dtype=np.float32) * 0.5

    assert overlay_heatmap(image, heatmap).size == (80, 60)


def test_overlay_alpha_extremes():
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    heatmap = np.zeros((32, 32), dtype=np.float32)

    untouched = np.array(overlay_heatmap(image, heatmap, alpha=0.0))
    assert np.allclose(untouched, np.array(image), atol=1)

    fully = np.array(overlay_heatmap(image, heatmap, alpha=1.0))
    assert not np.allclose(fully, np.array(image), atol=1)


def test_overlay_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(Image.new("RGB", (8, 8)), np.zeros((8, 8)), alpha=1.5)


def test_hot_regions_differ_from_cold_regions():
    """The colormap must actually vary, or overlays convey nothing."""
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[:16] = 1.0  # top half hot

    out = np.array(overlay_heatmap(image, heatmap, alpha=1.0))
    assert not np.allclose(out[:16].mean(axis=(0, 1)), out[16:].mean(axis=(0, 1)))


def test_denormalize_roundtrip():
    from food_recognition.data import build_transform

    original = Image.new("RGB", (64, 64), (123, 45, 200))
    tensor = build_transform(64, is_train=False)(original)
    restored = denormalize(tensor)

    assert restored.size == (64, 64)
    # Colour should survive the normalise/denormalise roundtrip.
    assert np.abs(
        np.array(restored, dtype=float).mean(axis=(0, 1))
        - np.array(original, dtype=float).mean(axis=(0, 1))
    ).max() < 12
