"""Exp7 runner wiring: second-order step hooks, the Newton-CG oracle, and metrics.

White-box, dataset-free checks that the COUPLE-FAC / Newton-CG step hooks compose the
provider, the parameter-space estimator, the overlay, and the safeguard into a finite
in-place update on a real ResNet-18, that the truncated-CG oracle solves a known
damped system, and that the backbone dispatch and speed-to-target metrics behave. The
mathematical correctness of the overlay and the safeguard is covered by their own tests;
here we verify the integration seam.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.config import Exp7Config  # noqa: E402
from experiments.models import SegmentedResNet18, SegmentedResNet50  # noqa: E402
from experiments.runner_exp7 import (  # noqa: E402
    _aggregate_method,
    _build_model,
    _epochs_to_target,
    _group_major_params,
    _newton_cg,
    build_couplefac_hook,
    build_newton_cg_hook,
)
from experiments.training import StepCtx  # noqa: E402
from hessian import ParamGroupedModel  # noqa: E402

_DEVICE = torch.device("cpu")


def _tiny_cfg() -> Exp7Config:
    """An Exp7 config with the cheapest curvature knobs (for fast CPU tests)."""
    return Exp7Config(
        coupling_n_probes=1,
        overlay_rank=1,
        overlay_n_power_iter=1,
        measure_period=1,
        damping=1e-1,
    )


def _resnet_batch(seed: int = 0) -> tuple[SegmentedResNet18, Tensor, Tensor]:
    """A real ResNet-18 (`ParamGroupedModel`) and a CIFAR-shaped classification minibatch.

    This is the Exp7 headline regime - a deep convolutional backbone in float32 - so
    the step hooks are exercised against the architecture and dtype the real run uses.
    Factored Tikhonov damping keeps the K-FAC factor inversion stable here, where the
    raw-factor eigendecomposition of exact damping fails to converge in float32.
    """
    torch.manual_seed(seed)
    model = SegmentedResNet18(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    return model, x, y


def _flat(model: ParamGroupedModel) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in _group_major_params(model)])


def _assert_hook_updates(
    model: ParamGroupedModel, ctx: StepCtx, hook: Callable[[StepCtx], None]
) -> None:
    before = _flat(model).clone()
    hook(ctx)
    after = _flat(model)
    assert torch.isfinite(after).all(), "update produced non-finite parameters"
    assert not torch.allclose(before, after), "hook did not change any parameter"


def test_newton_cg_solves_damped_system() -> None:
    torch.manual_seed(0)
    n = 12
    root = torch.randn(n, n)
    a = root @ root.t() + torch.eye(n)  # SPD
    damping = 0.1
    grad = torch.randn(n)

    direction = _newton_cg(
        grad, lambda v: a @ v, damping=damping, max_iter=n + 10, tol=1e-12
    )
    expected = torch.linalg.solve(a + damping * torch.eye(n), -grad)
    assert torch.allclose(
        direction, expected, atol=1e-4
    ), f"Newton-CG residual {(direction - expected).norm().item():.2e}"


def test_couplefac_hook_kfac_updates() -> None:
    model, x, y = _resnet_batch()
    hook = build_couplefac_hook(
        model, _tiny_cfg(), tau=float("inf"), seed=0, device=_DEVICE
    )
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)


def test_couplefac_hook_overlay_updates() -> None:
    model, x, y = _resnet_batch()
    cfg = _tiny_cfg()
    hook = build_couplefac_hook(model, cfg, tau=cfg.tau, seed=0, device=_DEVICE)
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)


def test_newton_cg_hook_updates() -> None:
    model, x, y = _resnet_batch()
    hook = build_newton_cg_hook(model, _tiny_cfg(), device=_DEVICE)
    ctx = StepCtx(model=model, x=x, y=y, step=0, device=_DEVICE)
    _assert_hook_updates(model, ctx, hook)


def test_group_major_params_is_group_major() -> None:
    model = SegmentedResNet18(num_classes=10)
    expected = [p for params in model.get_param_groups().values() for p in params]
    flat = _group_major_params(model)
    assert len(flat) == len(expected) and all(
        a is b for a, b in zip(flat, expected, strict=True)
    ), "params must follow get_param_groups (group-major) order"
    assert sum(p.numel() for p in flat) == sum(p.numel() for p in model.parameters())


def test_build_model_dispatch() -> None:
    m50 = _build_model(Exp7Config(model="resnet50", num_classes=10, pretrained=False))
    assert isinstance(
        m50, SegmentedResNet50
    ), "resnet50 key must build SegmentedResNet50"

    m18 = _build_model(Exp7Config(model="resnet18", num_classes=10))
    assert isinstance(m18, SegmentedResNet18) and not isinstance(m18, SegmentedResNet50)

    raised = False
    try:
        _build_model(Exp7Config(model="vit"))
    except ValueError:
        raised = True
    assert raised, "unknown backbone must raise ValueError"


def test_epochs_to_target_and_aggregate() -> None:
    assert (
        _epochs_to_target([0.1, 0.5, 0.9, 0.95], 0.9) == 3
    ), "first crossing at epoch 3"
    assert _epochs_to_target([0.1, 0.2], 0.9) is None, "never reaching target -> None"

    per_seed = {
        "seed_1": {"final_acc": 0.8, "epochs_to_target": 3},
        "seed_2": {"final_acc": 0.9, "epochs_to_target": None},
    }
    agg = _aggregate_method(per_seed, target=0.9)
    assert abs(agg["final_acc_mean"] - 0.85) < 1e-9, "final-acc mean"
    assert agg["n_reached_target"] == 1, "one seed reached target"
    assert agg["epochs_to_target_mean"] == 3.0, "mean over reached seeds only"
    assert agg["n_seeds"] == 2


if __name__ == "__main__":
    test_newton_cg_solves_damped_system()
    print("newton_cg_solves_damped_system: OK")

    test_couplefac_hook_kfac_updates()
    print("couplefac_hook_kfac_updates: OK")

    test_couplefac_hook_overlay_updates()
    print("couplefac_hook_overlay_updates: OK")

    test_newton_cg_hook_updates()
    print("newton_cg_hook_updates: OK")

    test_group_major_params_is_group_major()
    print("group_major_params_is_group_major: OK")

    test_build_model_dispatch()
    print("build_model_dispatch: OK")

    test_epochs_to_target_and_aggregate()
    print("epochs_to_target_and_aggregate: OK")

    print("test_exp7: all checks passed")
