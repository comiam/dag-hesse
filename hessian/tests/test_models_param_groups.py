"""SegmentedResNet parameter-group partition + ParamGroupedModel conformance.

Validates that the real ResNet backbones expose `get_param_groups` as a clean
partition of trainable parameters (one group per measurement segment, in segment
order) and that they plug into `ParamBlockEstimator.curvature` as `ParamGroupedModel`s.
ResNet-50 is built without pretrained weights to keep the test offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.models import (  # noqa: E402
    SegmentedMLP,
    SegmentedPlainResNet18,
    SegmentedResNet18,
    SegmentedResNet34,
    SegmentedResNet50,
    SegmentedWideResNet18,
)
from hessian.param_space import ParamBlockEstimator  # noqa: E402

_EXPECTED_GROUPS = ["stem", "layer1", "layer2", "layer3", "layer4", "head"]


def _assert_partitions(model: SegmentedResNet18) -> None:
    groups = model.get_param_groups()
    assert list(groups.keys()) == _EXPECTED_GROUPS, f"groups: {list(groups.keys())}"

    grouped = [p for ps in groups.values() for p in ps]
    grouped_ids = {id(p) for p in grouped}
    all_ids = {id(p) for p in model.parameters()}
    assert len(grouped_ids) == len(grouped), "a parameter appears in two groups"
    assert grouped_ids == all_ids, "groups must partition the model parameters exactly"


def test_param_groups_partition_resnet18() -> None:
    _assert_partitions(SegmentedResNet18(num_classes=10))


def test_param_groups_partition_resnet50() -> None:
    model = SegmentedResNet50(num_classes=10, pretrained=False)
    _assert_partitions(model)
    # head was resized to num_classes; the inherited segments produce matching logits
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10), "ResNet-50 shape mismatch"


def test_resnet18_is_param_grouped_model_for_curvature() -> None:
    torch.manual_seed(0)
    model = SegmentedResNet18(num_classes=4)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 4, (2,))

    total = sum(p.numel() for p in model.parameters())
    cur = ParamBlockEstimator(model, loss_fn, n_probes=1).curvature(x, y)

    assert not cur.flat_grad.requires_grad, "flat_grad must be detached"
    assert cur.flat_grad.numel() == total, "flat_grad must span all parameters"

    hv = cur.hvp(torch.randn(total))
    assert hv.shape == (total,), "hvp must return a full-parameter vector"
    assert torch.isfinite(hv).all(), "hvp must be finite"

    coupling = cur.coupling()
    expected_pairs = [
        (v, w)
        for i, v in enumerate(_EXPECTED_GROUPS)
        for w in _EXPECTED_GROUPS[i + 1 :]
    ]
    assert list(coupling.keys()) == expected_pairs, "coupling pairs/order wrong"


def test_param_groups_partition_plain_resnet18() -> None:
    _assert_partitions(SegmentedPlainResNet18(num_classes=10))


def test_param_groups_partition_resnet34() -> None:
    _assert_partitions(SegmentedResNet34(num_classes=10))


def test_wide_resnet18_partition_shape_and_scaling() -> None:
    counts = {}
    for width in (0.5, 1.0, 2.0):
        model = SegmentedWideResNet18(num_classes=10, width_mult=width)
        _assert_partitions(model)
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10), f"shape at {width}x"
        counts[width] = sum(p.numel() for p in model.parameters())
    assert counts[0.5] < counts[1.0] < counts[2.0], "width must scale capacity"
    tv_count = sum(p.numel() for p in SegmentedResNet18(num_classes=10).parameters())
    assert counts[1.0] == tv_count, "width 1.0 must reproduce standard ResNet-18 widths"


def test_segmented_mlp_param_groups() -> None:
    """SegmentedMLP exposes one group per Linear layer, partitioning its parameters."""
    model = SegmentedMLP(in_dim=16, hidden=16, depth=2, num_classes=4)
    groups = model.get_param_groups()
    assert list(groups.keys()) == ["fc0", "fc1", "head"], f"groups: {list(groups)}"

    grouped = [p for ps in groups.values() for p in ps]
    grouped_ids = {id(p) for p in grouped}
    assert len(grouped_ids) == len(grouped), "a parameter appears in two groups"
    assert grouped_ids == {id(p) for p in model.parameters()}, "groups must partition"
    assert model(torch.randn(3, 16)).shape == (3, 4), "forward maps (B, in) -> (B, cls)"


if __name__ == "__main__":
    test_param_groups_partition_resnet18()
    print("param_groups_partition_resnet18: OK")

    test_param_groups_partition_resnet50()
    print("param_groups_partition_resnet50: OK")

    test_resnet18_is_param_grouped_model_for_curvature()
    print("resnet18_is_param_grouped_model_for_curvature: OK")

    test_param_groups_partition_plain_resnet18()
    print("param_groups_partition_plain_resnet18: OK")

    test_param_groups_partition_resnet34()
    print("param_groups_partition_resnet34: OK")

    test_wide_resnet18_partition_shape_and_scaling()
    print("wide_resnet18_partition_shape_and_scaling: OK")

    test_segmented_mlp_param_groups()
    print("segmented_mlp_param_groups: OK")

    print("test_models_param_groups: all checks passed")
