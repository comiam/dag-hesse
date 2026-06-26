"""
CoupleFacOptimizer: composition of a K-FAC provider, the COUPLE-FAC overlay, and the
trust-region safeguard (Delta-M4).

DIP: the optimizer depends on the `KFACProvider` abstraction and an exact
`HessianOracle`, never on a concrete K-FAC library. Per step, the accepted update
direction is

  d0 = provider.base_direction(g)                          # K-FAC / EKFAC Newton step
  S  = overlay.select(coupling, tau)                       # coupled block pairs
  d1 = overlay.correct(g, d0, hvp, provider.block_ranges(), S)
  d  = safeguard.accept(g, d0, d1, hvp, radius)            # no-harm guarantee

The model-free direction algebra (`compute_direction`) is validated against exact
random quadratics. The `.step` binding then applies that direction in place to a held
parameter list - the group-major `block_ranges` layout - given the per-batch gradient,
exact Hessian-vector product, and parameter-space coupling the caller supplies (the
Stage-B runner builds these from the model with
`hessian.param_space.ParamBlockEstimator` and refreshes the provider's curvature on the
same minibatch). The concrete K-FAC backend lives behind the `KFACProvider` seam
(`secondorder.kfac_adapter.CurvlinopsKFACProvider`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import Parameter

from hessian.types import LayerID

from .overlay import CoupleFacOverlay
from .safeguard import TrustRegionSafeguard
from .types import HessianOracle, Pair


class KFACProvider(Protocol):
    """A block-diagonal second-order backend (K-FAC / EKFAC).

    Supplies the base preconditioned direction and the parameter-group block layout
    the overlay needs. It owns no training logic (SRP) and is the only seam the
    optimizer binds to a concrete library (DIP).
    """

    def base_direction(self, grad: Tensor) -> Tensor:
        """d0 = -(blkdiag H_vv + lambda I)^{-1} grad  (flat, full-parameter)."""
        ...

    def block_ranges(self) -> dict[LayerID, slice]:
        """Contiguous slice of the flat parameter vector owned by each block."""
        ...


@dataclass
class CoupleFacOptimizer:
    """Coupling-gated, safeguarded overlay over a block-diagonal K-FAC backend."""

    provider: KFACProvider
    overlay: CoupleFacOverlay = field(default_factory=CoupleFacOverlay)
    safeguard: TrustRegionSafeguard = field(default_factory=TrustRegionSafeguard)
    tau: float = 0.1
    radius: float = float("inf")
    params: list[Parameter] | None = None
    lr: float = 1.0

    def compute_direction(
        self,
        grad: Tensor,
        hvp: HessianOracle,
        coupling: dict[Pair, float],
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Accepted update direction (no-harm: m(d) <= m(d0) on every call)."""
        d0 = self.provider.base_direction(grad)
        selected = self.overlay.select(coupling, self.tau)
        d1 = self.overlay.correct(
            grad,
            d0,
            hvp,
            self.provider.block_ranges(),
            selected,
            generator=generator,
        )
        return self.safeguard.accept(grad, d0, d1, hvp, self.radius)

    def step(
        self,
        grad: Tensor,
        hvp: HessianOracle,
        coupling: dict[Pair, float],
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Apply one accepted COUPLE-FAC update in place; return the direction taken.

        ``grad``, ``hvp`` and ``coupling`` are the per-batch parameter-space gradient,
        exact Hessian-vector product, and block coupling - the caller builds them on the
        minibatch used to refresh the provider's curvature (e.g. via
        `hessian.param_space.ParamBlockEstimator`). The accepted direction ``d`` (no-harm:
        m(d) <= m(d0)) updates the held parameters as ``theta <- theta + lr * d``,
        unflattened over the group-major `block_ranges` layout.
        """
        params = self.params
        if params is None:
            raise RuntimeError(
                "CoupleFacOptimizer.step needs `params` (the group-major parameter list "
                "matching the provider's block layout); it was not set"
            )
        d = self.compute_direction(grad, hvp, coupling, generator=generator)
        self._apply(d, params)
        return d

    def _apply(self, d: Tensor, params: list[Parameter]) -> None:
        """theta <- theta + lr * d, unflattened row-major over ``params``."""
        with torch.no_grad():
            offset = 0
            for p in params:
                n = p.numel()
                p.add_(self.lr * d[offset : offset + n].view_as(p))
                offset += n
