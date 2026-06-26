"""
Trust-region safeguard for the COUPLE-FAC overlay (Delta-M4, step 3).

The overlay proposes a corrected direction d1 = d0 + sum_vw Delta_vw on top of the
block-diagonal (K-FAC) direction d0. The safeguard accepts d1 only if it does not
worsen the *exact* local quadratic model and stays inside the trust region; otherwise
it falls back to d0. This yields the no-harm guarantee:

  m(d) = g^T d + 1/2 d^T H d,
  accept d1  iff  m(d1) <= m(d0)  and  ||d1|| <= radius,  else  d0,
  =>  m(d_accepted) <= m(d0)  every step      (Proposition: no-harm).

H enters only through the exact HessianOracle (one HVP per evaluated candidate).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .types import HessianOracle


@dataclass(frozen=True)
class TrustRegionSafeguard:
    """Accepts a correction iff it does not increase the exact local model."""

    tol: float = 0.0  # acceptance slack: accept when m(d1) <= m(d0) + tol

    def accept(
        self,
        grad: Tensor,
        d0: Tensor,
        d1: Tensor,
        hvp: HessianOracle,
        radius: float,
    ) -> Tensor:
        """Returns d1 if it is safe (no model increase, within radius), else d0."""
        if d1.norm().item() > radius:
            return d0
        m0 = self._model_value(grad, d0, hvp)
        m1 = self._model_value(grad, d1, hvp)
        return d1 if m1 <= m0 + self.tol else d0

    @staticmethod
    def _model_value(grad: Tensor, d: Tensor, hvp: HessianOracle) -> float:
        """m(d) = g^T d + 1/2 d^T H d (exact, via one HVP)."""
        return (grad.dot(d) + 0.5 * d.dot(hvp(d))).item()
