"""Refactor-safety test for the activation-space StochasticHessianEstimator.

Guards the DRY extraction of the Hutchinson / power-iteration kernels
(`hessian/_estimators.py`): after the refactor the public estimates must still
match the exact block Hessian (`ExactBlockHessian`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# dag_hesse is the root experiment package; add it to sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.models import PlainMLP  # noqa: E402
from hessian import ExactBlockHessian, StochasticHessianEstimator  # noqa: E402


def _make_model(device: torch.device) -> PlainMLP:
    model = PlainMLP(input_dim=32, width=8, depth=3, activation="gelu").to(device)
    model.eval()
    return model


def _make_data(*, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    return x, y


# ------------------------------------------------------------------
# 1. Cross-block Frobenius norm vs exact
# ------------------------------------------------------------------


def test_frob_norm_vs_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    model = _make_model(device)
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v, w = names[0], names[1]

    exact = ExactBlockHessian(model, loss_fn).compute_block(x, y, v, w)
    r_exact = exact.frobenius_norm

    est = StochasticHessianEstimator(model, loss_fn, n_probes=2000)
    r_approx = est.estimate_block_frob_norm(x, y, v, w)

    rel = abs(r_exact - r_approx) / (r_exact + 1e-12)
    assert rel < 0.1, (
        f"||H_{v},{w}||_F mismatch: exact={r_exact:.4f} "
        f"approx={r_approx:.4f} rel={rel:.3f}"
    )
    print(f"  ||H_{v},{w}||_F exact={r_exact:.4f} approx={r_approx:.4f} - OK")


# ------------------------------------------------------------------
# 2. Frobenius + stable rank (diagonal block) vs exact
# ------------------------------------------------------------------


def test_frob_and_stable_rank_vs_exact() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    model = _make_model(device)
    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v = names[0]

    exact = ExactBlockHessian(model, loss_fn).compute_block(x, y, v, v)
    r_exact = exact.frobenius_norm
    d_exact = exact.stable_rank

    est = StochasticHessianEstimator(model, loss_fn, n_probes=2000, n_power_iter=100)
    r_approx, d_approx = est.estimate_frob_and_stable_rank(x, y, v, v)

    rel_r = abs(r_exact - r_approx) / (r_exact + 1e-12)
    rel_d = abs(d_exact - d_approx) / (d_exact + 1e-12)
    assert rel_r < 0.1, f"frob mismatch exact={r_exact:.4f} approx={r_approx:.4f}"
    assert (
        rel_d < 0.15
    ), f"stable-rank mismatch exact={d_exact:.4f} approx={d_approx:.4f} rel={rel_d:.3f}"
    print(
        f"  D({v},{v}) exact={d_exact:.4f} approx={d_approx:.4f} "
        f"(||H||_F exact={r_exact:.4f} approx={r_approx:.4f}) - OK"
    )


# ------------------------------------------------------------------

if __name__ == "__main__":
    print("test_frob_norm_vs_exact...")
    test_frob_norm_vs_exact()

    print("test_frob_and_stable_rank_vs_exact...")
    test_frob_and_stable_rank_vs_exact()

    print("\nAll tests passed.")
