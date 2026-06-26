"""
Validates the COUPLE-FAC overlay (Delta-M4) on random quadratics, where the exact
Hessian, the block-diagonal (K-FAC) optimum, and the full-Newton optimum are all known
in closed form. This is a fully independent reference for the overlay math.

Checked:
  - no-harm  : m(d_accepted) <= m(d0) for every random quadratic (the safeguard
               guarantee), across PD and indefinite curvature;
  - benefit  : with real inter-block coupling and PD curvature, the overlay strictly
               decreases the local model and never beats the exact Newton optimum;
  - fallback : the safeguard rejects a model-increasing or out-of-trust-region step;
  - select   : coupling-thresholded pair selection is correct and capped;
  - step     : `.step` applies theta <- theta + lr*d (d == compute_direction) to the
               held parameters in block layout, and requires a parameter list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# dag_hesse is the root experiment package; add it to sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from secondorder import (  # noqa: E402
    CoupleFacOptimizer,
    CoupleFacOverlay,
    TrustRegionSafeguard,
)

_DTYPE = torch.float64


def _spd(n: int, gen: torch.Generator) -> torch.Tensor:
    """A symmetric positive-definite n x n block."""
    a = torch.randn(n, n, dtype=_DTYPE, generator=gen)
    return a @ a.t() / n + torch.eye(n, dtype=_DTYPE)


def _build_quadratic(
    sizes: list[int],
    coupling_scale: float,
    gen: torch.Generator,
    *,
    make_pd: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
    """Random quadratic m(d) = g^T d + 1/2 d^T H d with a known block structure.

    Diagonal blocks are SPD (so K-FAC is always well-defined); off-diagonal blocks are
    scaled by `coupling_scale`. When `make_pd`, H is shifted to be PD overall.
    """
    p = sum(sizes)
    h = torch.zeros(p, p, dtype=_DTYPE)
    ranges: dict[str, slice] = {}
    offset = 0
    for i, s in enumerate(sizes):
        sl = slice(offset, offset + s)
        ranges[str(i)] = sl
        h[sl, sl] = _spd(s, gen)
        offset += s

    names = list(ranges)
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            si, sj = ranges[names[i]], ranges[names[j]]
            block = coupling_scale * torch.randn(
                sizes[i], sizes[j], dtype=_DTYPE, generator=gen
            )
            h[si, sj] = block
            h[sj, si] = block.t()

    h = 0.5 * (h + h.t())
    if make_pd:
        lo = torch.linalg.eigvalsh(h)[0].item()
        if lo <= 1e-3:
            h = h + (1e-3 - lo) * torch.eye(p, dtype=_DTYPE)
    g = torch.randn(p, dtype=_DTYPE, generator=gen)
    return g, h, ranges


class _ExactKFAC:
    """Block-diagonal Newton provider: d0 = -blkdiag(H)^{-1} g (the K-FAC oracle)."""

    def __init__(self, h: torch.Tensor, ranges: dict[str, slice]) -> None:
        self._h = h
        self._ranges = ranges

    def base_direction(self, grad: torch.Tensor) -> torch.Tensor:
        d0 = torch.zeros_like(grad)
        for sl in self._ranges.values():
            d0[sl] = torch.linalg.solve(self._h[sl, sl], -grad[sl])
        return d0

    def block_ranges(self) -> dict[str, slice]:
        return self._ranges


def _hvp_of(h: torch.Tensor):
    return lambda u: h @ u


def _model(g: torch.Tensor, h: torch.Tensor, d: torch.Tensor) -> float:
    return (g.dot(d) + 0.5 * d.dot(h @ d)).item()


def _coupling(
    h: torch.Tensor, ranges: dict[str, slice]
) -> dict[tuple[str, str], float]:
    names = list(ranges)
    out: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            si, sj = ranges[names[i]], ranges[names[j]]
            f_vw = h[si, sj].norm().item()
            f_vv = h[si, si].norm().item()
            f_ww = h[sj, sj].norm().item()
            out[(names[i], names[j])] = f_vw / (f_vv * f_ww) ** 0.5
    return out


def test_no_harm_random_quadratics() -> None:
    """The safeguard guarantees m(d_accepted) <= m(d0) for every quadratic."""
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        g, h, ranges = _build_quadratic(
            [4, 5, 3], coupling_scale=0.3, gen=gen, make_pd=(seed % 2 == 0)
        )
        opt = CoupleFacOptimizer(provider=_ExactKFAC(h, ranges), tau=0.0)
        d0 = opt.provider.base_direction(g)
        d = opt.compute_direction(g, _hvp_of(h), _coupling(h, ranges), generator=gen)
        m0, md = _model(g, h, d0), _model(g, h, d)
        assert md <= m0 + 1e-7, f"no-harm violated (seed={seed}): m(d)={md} m(d0)={m0}"


def test_overlay_strict_benefit() -> None:
    """With coupling and PD curvature, the overlay strictly decreases m, m* <= m < m0."""
    for seed in range(10):
        gen = torch.Generator().manual_seed(100 + seed)
        g, h, ranges = _build_quadratic([5, 5], coupling_scale=0.7, gen=gen)
        opt = CoupleFacOptimizer(
            provider=_ExactKFAC(h, ranges),
            overlay=CoupleFacOverlay(rank=2, n_power_iter=12),
            tau=0.0,
        )
        d0 = opt.provider.base_direction(g)
        d = opt.compute_direction(g, _hvp_of(h), _coupling(h, ranges), generator=gen)
        d_newton = torch.linalg.solve(h, -g)
        m0, md, m_star = (
            _model(g, h, d0),
            _model(g, h, d),
            _model(g, h, d_newton),
        )
        assert md < m0 - 1e-6, f"overlay gave no benefit (seed={seed}): m={md} m0={m0}"
        assert (
            md >= m_star - 1e-7
        ), f"beat the Newton optimum (seed={seed}): {md} < {m_star}"


def test_safeguard_rejects_harmful_and_far_steps() -> None:
    """The safeguard falls back to d0 on a model-increasing or out-of-region step."""
    gen = torch.Generator().manual_seed(7)
    g, h, _ = _build_quadratic([4, 4], coupling_scale=0.5, gen=gen)
    hvp = _hvp_of(h)
    sg = TrustRegionSafeguard()
    d0 = torch.zeros_like(g)

    harmful = 10.0 * torch.randn(g.numel(), dtype=_DTYPE, generator=gen)
    assert torch.equal(sg.accept(g, d0, harmful, hvp, radius=float("inf")), d0)

    far = torch.ones(g.numel(), dtype=_DTYPE)
    assert torch.equal(sg.accept(g, d0, far, hvp, radius=0.1), d0)


def test_select_threshold_and_cap() -> None:
    """select keeps pairs above tau, strongest first, capped at max_pairs."""
    overlay = CoupleFacOverlay(max_pairs=2)
    coupling = {("0", "1"): 0.9, ("0", "2"): 0.05, ("1", "2"): 0.5}
    assert overlay.select(coupling, tau=0.1) == [("0", "1"), ("1", "2")]


def test_step_applies_accepted_direction() -> None:
    """`.step` applies theta <- theta + lr*d in block layout, with d == compute_direction."""
    gen = torch.Generator().manual_seed(11)
    g, h, ranges = _build_quadratic([4, 5], coupling_scale=0.7, gen=gen)
    lr = 0.5
    params = [
        torch.nn.Parameter(torch.randn(4, dtype=_DTYPE, generator=gen)),
        torch.nn.Parameter(torch.randn(5, dtype=_DTYPE, generator=gen)),
    ]
    opt = CoupleFacOptimizer(
        provider=_ExactKFAC(h, ranges),
        overlay=CoupleFacOverlay(rank=2, n_power_iter=12),
        params=params,
        lr=lr,
        tau=0.0,
    )

    # the overlay draws its subspace from `generator`; seed both calls identically so the
    # reference direction matches the one `.step` applies. compute_direction mutates nothing.
    d_expected = opt.compute_direction(
        g, _hvp_of(h), _coupling(h, ranges), generator=torch.Generator().manual_seed(3)
    )
    before = [p.detach().clone() for p in params]

    d_step = opt.step(
        g, _hvp_of(h), _coupling(h, ranges), generator=torch.Generator().manual_seed(3)
    )

    assert torch.allclose(d_step, d_expected), "step direction != compute_direction"
    offset = 0
    for p, p0 in zip(params, before, strict=True):
        n = p.numel()
        moved = p.detach() - p0
        assert torch.allclose(
            moved, lr * d_expected[offset : offset + n].view_as(p), atol=1e-12
        ), "parameter not updated by lr * d in block layout"
        offset += n


def test_step_requires_params() -> None:
    """`.step` without a held parameter list is a usage error, not a silent no-op."""
    gen = torch.Generator().manual_seed(5)
    g, h, ranges = _build_quadratic([3, 3], coupling_scale=0.3, gen=gen)
    opt = CoupleFacOptimizer(provider=_ExactKFAC(h, ranges), tau=0.0)
    raised = False
    try:
        opt.step(g, _hvp_of(h), _coupling(h, ranges))
    except RuntimeError:
        raised = True
    assert raised, "step must require params to be set"


if __name__ == "__main__":
    test_no_harm_random_quadratics()
    test_overlay_strict_benefit()
    test_safeguard_rejects_harmful_and_far_steps()
    test_select_threshold_and_cap()
    test_step_applies_accepted_direction()
    test_step_requires_params()
    print("secondorder: all checks passed")
