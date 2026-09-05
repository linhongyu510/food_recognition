"""Tests for model construction, CBAM and the model registry."""

from __future__ import annotations

import pytest
import torch

from food_recognition.models import (
    CBAM,
    BackboneWithCBAM,
    ChannelAttention,
    SimpleConvNet,
    SpatialAttention,
    available_models,
    count_parameters,
    initialize_model,
)


def test_channel_attention_shape_and_range():
    module = ChannelAttention(32)
    weights = module(torch.randn(2, 32, 8, 8))

    # one scalar gate per channel, in (0, 1)
    assert weights.shape == (2, 32, 1, 1)
    assert bool((weights > 0).all()) and bool((weights < 1).all())


def test_spatial_attention_shape_and_range():
    module = SpatialAttention(kernel_size=7)
    weights = module(torch.randn(2, 16, 8, 8))

    # one gate per spatial position, shared across channels
    assert weights.shape == (2, 1, 8, 8)
    assert bool((weights > 0).all()) and bool((weights < 1).all())


def test_spatial_attention_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        SpatialAttention(kernel_size=8)


def test_cbam_preserves_shape():
    x = torch.randn(2, 64, 7, 7)
    out = CBAM(64)(x)
    assert out.shape == x.shape


def test_cbam_actually_modulates_input():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 7, 7)
    out = CBAM(64)(x)
    # attention gates are in (0,1), so output must differ from input
    assert not torch.allclose(out, x)


def test_cbam_handles_small_channel_count():
    """reduction=16 on 8 channels must not collapse to a 0-width layer."""
    out = CBAM(8, reduction=16)(torch.randn(1, 8, 4, 4))
    assert out.shape == (1, 8, 4, 4)


def test_cbam_is_differentiable():
    x = torch.randn(1, 16, 4, 4, requires_grad=True)
    CBAM(16)(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_simple_convnet_forward():
    model = SimpleConvNet(num_classes=5)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 5)


def test_simple_convnet_accepts_variable_input_size():
    """Adaptive pooling: the legacy hard-coded 512*7*7 broke on other sizes."""
    model = SimpleConvNet(num_classes=4)
    for size in (32, 64, 96):
        assert model(torch.randn(1, 3, size, size)).shape == (1, 4)


@pytest.mark.parametrize("name", ["simple_cnn", "resnet18", "efficientnet_b0"])
def test_initialize_model_without_pretrained(name):
    model, input_size = initialize_model(name, num_classes=7, use_pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, input_size, input_size))
    assert out.shape == (1, 7)


@pytest.mark.parametrize("name", ["resnet18_cbam", "efficientnet_b0_cbam"])
def test_cbam_variants_infer_channels(name):
    """CBAM channel count must come from the backbone, not a constant."""
    model, input_size = initialize_model(name, num_classes=11, use_pretrained=False)
    assert isinstance(model, BackboneWithCBAM)

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, input_size, input_size))
    assert out.shape == (1, 11)


def test_efficientnet_b4_cbam_uses_b4_channel_count():
    """B4's final stage is 1792 channels, not B0's 1280."""
    model, _ = initialize_model(
        "efficientnet_b4_cbam", num_classes=101, use_pretrained=False
    )
    assert model.cbam.channel_attention.mlp[0].in_channels == 1792


def test_efficientnet_b0_cbam_uses_b0_channel_count():
    model, _ = initialize_model(
        "efficientnet_b0_cbam", num_classes=11, use_pretrained=False
    )
    assert model.cbam.channel_attention.mlp[0].in_channels == 1280


def test_unknown_model_name_raises_with_suggestions():
    with pytest.raises(ValueError, match="unsupported model_name"):
        initialize_model("not_a_real_model", num_classes=3, use_pretrained=False)


def test_registry_contains_expected_entries():
    names = available_models()
    for expected in [
        "simple_cnn",
        "resnet18",
        "resnet50_cbam",
        "efficientnet_b0_cbam",
        "efficientnet_b4_cbam",
    ]:
        assert expected in names
    assert names == sorted(set(names))


def test_linear_probe_freezes_backbone_only():
    model, _ = initialize_model(
        "resnet18", num_classes=5, use_pretrained=False, linear_probe=True
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]

    assert trainable, "linear probe must leave the head trainable"
    assert all(name.startswith("fc") for name in trainable)
    assert count_parameters(model, trainable_only=True) < count_parameters(
        model, trainable_only=False
    )


def test_linear_probe_on_cbam_model_trains_head():
    model, _ = initialize_model(
        "resnet18_cbam", num_classes=5, use_pretrained=False, linear_probe=True
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(name.startswith("head") for name in trainable)


def test_dropout_is_inserted_when_requested():
    model, _ = initialize_model(
        "resnet18", num_classes=5, use_pretrained=False, dropout=0.5
    )
    assert isinstance(model.fc, torch.nn.Sequential)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.fc)


def test_head_replacement_sets_output_dimension():
    model, _ = initialize_model("resnet18", num_classes=42, use_pretrained=False)
    assert model.fc.out_features == 42
