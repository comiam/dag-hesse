"""Exp8 runner wiring: backbone dispatch, checkpoint resolution, and Phi measurement.

White-box, mostly dataset-free checks that the Stage-A diagnostic runner composes the
parameter-space estimator, the backbone segmentation, and the checkpoint schedule into a
JSON-safe Phi report on a real ResNet-18, and that the named-checkpoint resolution and
seed aggregation behave. The numerical fidelity of Phi, the diagonal norms, and the
coupling field is covered exactly in `test_param_space`; here we verify the integration
seam.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.config import Exp8Config  # noqa: E402
from experiments.models import (  # noqa: E402
    SegmentedPlainResNet18,
    SegmentedResNet18,
    SegmentedResNet50,
)
from experiments.runner_exp8 import (  # noqa: E402
    _aggregate,
    _build_model,
    _epoch_to_label,
    _measure_phi,
    _resolve_checkpoint_epochs,
)

_DEVICE = torch.device("cpu")


def _resnet_batch(
    seed: int = 0,
) -> tuple[SegmentedResNet18, torch.Tensor, torch.Tensor]:
    """A real ResNet-18 (`ParamGroupedModel`) and a CIFAR-shaped classification minibatch."""
    torch.manual_seed(seed)
    model = SegmentedResNet18(num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    return model, x, y


def test_build_model_dispatch() -> None:
    cfg = Exp8Config()

    m18 = _build_model(cfg, "resnet18")
    assert isinstance(m18, SegmentedResNet18) and not isinstance(
        m18, (SegmentedPlainResNet18, SegmentedResNet50)
    ), "resnet18 key must build a plain SegmentedResNet18"

    m_plain = _build_model(cfg, "plain_resnet18")
    assert isinstance(
        m_plain, SegmentedPlainResNet18
    ), "plain_resnet18 key must build SegmentedPlainResNet18"

    m50 = _build_model(Exp8Config.a2_pretrained(), "resnet50")
    assert isinstance(
        m50, SegmentedResNet50
    ), "resnet50 key must build SegmentedResNet50"

    raised = False
    try:
        _build_model(cfg, "vit")
    except ValueError:
        raised = True
    assert raised, "unknown arch must raise ValueError"


def test_resolve_checkpoint_epochs() -> None:
    assert _resolve_checkpoint_epochs(30, ["init", "mid", "final"]) == [
        0,
        15,
        30,
    ], "init/mid/final must map to 0, total//2, total"
    assert _resolve_checkpoint_epochs(0, ["init"]) == [
        0
    ], "epochs=0 (a2) collapses to the single init checkpoint"


def test_epoch_to_label() -> None:
    assert _epoch_to_label(0, 30) == "init"
    assert _epoch_to_label(15, 30) == "mid"
    assert _epoch_to_label(30, 30) == "final"
    # a2: with total=0 the init snapshot (epoch 0) is still labelled "init"
    assert _epoch_to_label(0, 0) == "init"


def test_measure_phi_structure() -> None:
    model, x, y = _resnet_batch()
    report = _measure_phi(model, x, y, nn.CrossEntropyLoss(), n_probes=1)

    assert set(report) == {"phi", "diag_frob", "coupling"}, "report keys"
    assert isinstance(report["phi"], float) and not math.isinf(
        report["phi"]
    ), "phi must be a finite float (or NaN)"

    names = list(model.get_param_groups().keys())
    assert list(report["diag_frob"].keys()) == names, "diag_frob must be group-major"
    assert all(isinstance(v, float) for v in report["diag_frob"].values())

    expected_pairs = {f"{v}->{w}" for i, v in enumerate(names) for w in names[i + 1 :]}
    assert (
        set(report["coupling"]) == expected_pairs
    ), "coupling must cover all v<w pairs"
    assert all(isinstance(v, float) for v in report["coupling"].values())


def test_aggregate() -> None:
    raw = {
        "resnet18": {
            "init": {
                "seed_1": {
                    "phi": 0.4,
                    "diag_frob": {"stem": 1.0, "head": 2.0},
                    "coupling": {"stem->head": 0.3},
                    "test_acc": 0.10,
                },
                "seed_2": {
                    "phi": 0.6,
                    "diag_frob": {"stem": 3.0, "head": 4.0},
                    "coupling": {"stem->head": 0.5},
                    "test_acc": 0.20,
                },
            }
        }
    }
    agg = _aggregate(raw)["resnet18"]["init"]

    assert abs(agg["phi_mean"] - 0.5) < 1e-9 and abs(agg["phi_std"] - 0.1) < 1e-9
    assert abs(agg["diag_frob"]["stem"]["mean"] - 2.0) < 1e-9
    assert abs(agg["diag_frob"]["head"]["mean"] - 3.0) < 1e-9
    assert abs(agg["coupling"]["stem->head"]["mean"] - 0.4) < 1e-9
    assert abs(agg["test_acc_mean"] - 0.15) < 1e-9
    assert agg["n_seeds"] == 2


if __name__ == "__main__":
    test_build_model_dispatch()
    print("build_model_dispatch: OK")

    test_resolve_checkpoint_epochs()
    print("resolve_checkpoint_epochs: OK")

    test_epoch_to_label()
    print("epoch_to_label: OK")

    test_measure_phi_structure()
    print("measure_phi_structure: OK")

    test_aggregate()
    print("aggregate: OK")

    print("test_exp8: all checks passed")
