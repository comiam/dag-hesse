"""
Second-order optimization: the COUPLE-FAC overlay over K-FAC (Delta-M4).

Where `hessian/` *measures* inter-layer curvature, this package *acts* on it: a
coupling-gated, trust-region-safeguarded rank-r correction applied on top of a
block-diagonal (K-FAC / EKFAC) Newton direction. The correction recovers the
off-diagonal curvature a block-diagonal preconditioner discards, while the safeguard
guarantees the exact local quadratic model never gets worse (no-harm).

Public API:
  TrustRegionSafeguard - accept/reject a correction by the exact local model.
  CoupleFacOverlay     - select coupled blocks + rank-r reduced-Newton correction.
  CoupleFacOptimizer   - composition (K-FAC provider + overlay + safeguard).
  KFACProvider         - abstraction a K-FAC / EKFAC backend must satisfy (DIP).
  HessianOracle, Pair  - the matrix-free HVP closure and a coupled block pair.

The core above is library-independent (DIP). The concrete curvlinops-backed provider
lives in the opt-in submodule ``secondorder.kfac_adapter`` (``CurvlinopsKFACProvider``)
so that importing the overlay never pulls in a K-FAC library.
"""

from __future__ import annotations

from .optimizer import CoupleFacOptimizer, KFACProvider
from .overlay import CoupleFacOverlay
from .safeguard import TrustRegionSafeguard
from .types import HessianOracle, Pair

__all__ = [
    "CoupleFacOptimizer",
    "CoupleFacOverlay",
    "HessianOracle",
    "KFACProvider",
    "Pair",
    "TrustRegionSafeguard",
]
