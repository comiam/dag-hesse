"""General-purpose utilities: seeding, serialization helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from hessian import PhiReport


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def phi_report_to_dict(report: PhiReport) -> dict[str, Any]:
    """JSON-safe view of a `PhiReport`: scalar Phi, diagonal norms, coupling field.

    Coupling keys are flattened from ``(v, w)`` tuples to ``"v->w"`` strings so the bundle
    serializes directly. Shared by every Phi runner (exp8, exp9).
    """
    return {
        "phi": report.phi,
        "diag_frob": dict(report.diag_frob),
        "coupling": {f"{v}->{w}": c for (v, w), c in report.coupling.items()},
    }
