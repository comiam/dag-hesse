"""
COUPLE-FAC overlay: coupling-gated rank-r correction over K-FAC (Delta-M4, steps 1-2).

Given the K-FAC direction d0 = -(blkdiag H_vv + lambda I)^{-1} g, the overlay:

  1. Selects coupled block pairs  S = {(v, w) : C(v, w) > tau}.
  2. For each (v, w) in S, builds a small subspace
        U_vw = span{ d0|_v, d0|_w }  (+)  top-r singular directions of H_{theta_v,theta_w}
     and solves the *reduced* Newton system there,
        c* = -(U^T H U + mu I)^{-1} U^T (g + H d0),     Delta_vw = U c*,
     lifting back   d1 = d0 + sum_{(v,w) in S} Delta_vw.

Why this captures exactly the discarded curvature: at the K-FAC point the block-v
residual is purely off-diagonal,
        (g + H d0)|_v = sum_{u != v} H_{vu} d0|_u,
i.e. the inter-layer curvature a block-diagonal preconditioner ignores. The reduced
Newton step over a subspace containing the cross-block singular directions provably
decreases the local model whenever this residual has mass on U_vw (PD case); the
`TrustRegionSafeguard` backstops the indefinite / trust-region cases.

Everything is matrix-free: the cross-block action is read off the full HVP by
embed/slice,   H_{vw} z_w = (H * embed_w(z_w))|_v.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from hessian.types import LayerID

from .types import HessianOracle, Pair

_RANK_REVEAL_TOL = 1e-8


def _embed(values: Tensor, sl: slice, dim: int) -> Tensor:
    """Lift a block-local vector into the full P-dim space (zeros off the block)."""
    col = torch.zeros(dim, device=values.device, dtype=values.dtype)
    col[sl] = values
    return col


def _apply_cols(op: Callable[[Tensor], Tensor], cols: Tensor) -> Tensor:
    """Apply a matrix-free operator column-wise to `cols` (shape d_in x k)."""
    return torch.stack([op(cols[:, j]) for j in range(cols.shape[1])], dim=1)


def _orthonormal(basis: Tensor) -> Tensor | None:
    """Orthonormalize columns (QR), dropping (near-)dependent ones; None if empty."""
    basis = basis[:, basis.norm(dim=0) > _RANK_REVEAL_TOL]
    if basis.shape[1] == 0:
        return None
    q, r = torch.linalg.qr(basis)
    q = q[:, r.diagonal().abs() > _RANK_REVEAL_TOL]
    return q if q.shape[1] > 0 else None


@dataclass(frozen=True)
class CoupleFacOverlay:
    """Coupling-gated rank-r reduced-Newton correction over a K-FAC direction."""

    rank: int = 2
    n_power_iter: int = 8
    damping: float = 1e-6
    max_pairs: int | None = None

    # -- selection (Delta-M4 step 1) -----------------------------------
    def select(self, coupling: dict[Pair, float], tau: float) -> list[Pair]:
        """S = {(v, w) : C(v, w) > tau}, strongest coupling first, capped at max_pairs."""
        flagged = sorted(
            (pair for pair, c in coupling.items() if c > tau),
            key=lambda pair: coupling[pair],
            reverse=True,
        )
        return flagged if self.max_pairs is None else flagged[: self.max_pairs]

    # -- correction (Delta-M4 step 2) ----------------------------------
    def correct(
        self,
        grad: Tensor,
        d0: Tensor,
        hvp: HessianOracle,
        block_ranges: dict[LayerID, slice],
        selected: list[Pair],
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """d1 = d0 + sum_{(v,w) in S} U_vw c*_vw  (independent reduced Newton per pair)."""
        if not selected:
            return d0
        residual = grad + hvp(
            d0
        )  # g + H d0; the block-v slice is the discarded coupling
        d1 = d0.clone()
        for v, w in selected:
            basis = self._pair_subspace(
                d0, hvp, block_ranges[v], block_ranges[w], generator
            )
            if basis is not None:
                d1 = d1 + self._reduced_newton(basis, residual, hvp)
        return d1

    # -- internals -----------------------------------------------------
    def _pair_subspace(
        self,
        d0: Tensor,
        hvp: HessianOracle,
        sv: slice,
        sw: slice,
        generator: torch.Generator | None,
    ) -> Tensor | None:
        dim = d0.numel()
        cols = [_embed(d0[sv], sv, dim), _embed(d0[sw], sw, dim)]
        left, right = self._cross_singular(d0, hvp, sv, sw, generator)
        for i in range(right.shape[1]):
            cols.append(_embed(left[:, i], sv, dim))
            cols.append(_embed(right[:, i], sw, dim))
        return _orthonormal(torch.stack(cols, dim=1))

    def _cross_singular(
        self,
        ref: Tensor,
        hvp: HessianOracle,
        sv: slice,
        sw: slice,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """Top-r singular subspaces of H_{vw} via matrix-free subspace iteration.

        Returns (left in R^{d_v x r}, right in R^{d_w x r}) - the dominant left/right
        singular directions of z_w |-> (H embed_w(z_w))|_v.
        """
        dim = ref.numel()
        d_v, d_w = ref[sv].numel(), ref[sw].numel()
        r = min(self.rank, d_v, d_w)
        if r <= 0:
            return ref.new_zeros(d_v, 0), ref.new_zeros(d_w, 0)

        def matvec(z_w: Tensor) -> Tensor:  # H_{vw}: R^{d_w} -> R^{d_v}
            return hvp(_embed(z_w, sw, dim))[sv]

        def rmatvec(z_v: Tensor) -> Tensor:  # H_{vw}^T: R^{d_v} -> R^{d_w}
            return hvp(_embed(z_v, sv, dim))[sw]

        right = torch.randn(
            d_w, r, device=ref.device, dtype=ref.dtype, generator=generator
        )
        right, _ = torch.linalg.qr(right)
        for _ in range(self.n_power_iter):
            left, _ = torch.linalg.qr(_apply_cols(matvec, right))
            right, _ = torch.linalg.qr(_apply_cols(rmatvec, left))
        left, _ = torch.linalg.qr(_apply_cols(matvec, right))
        return left, right

    def _reduced_newton(
        self, basis: Tensor, residual: Tensor, hvp: HessianOracle
    ) -> Tensor:
        """Delta = U c*,  c* = -(U^T H U + mu I)^{-1} U^T (g + H d0)."""
        h_red = basis.t() @ _apply_cols(hvp, basis)
        h_red = 0.5 * (h_red + h_red.t())
        k = basis.shape[1]
        mu = self.damping * h_red.diagonal().abs().mean().clamp(min=1.0)
        eye = torch.eye(k, device=basis.device, dtype=basis.dtype)
        c = torch.linalg.solve(h_red + mu * eye, -(basis.t() @ residual))
        return basis @ c
