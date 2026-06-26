"""
CurvlinopsKFACProvider: the concrete K-FAC backend behind the `KFACProvider` seam.

This is the single place in `secondorder/` that binds a real K-FAC library
(curvlinops, Dangel et al.). It supplies the block-diagonal Newton / natural-gradient
direction the overlay then corrects:

  d0 = -( blkdiag_l (F_l + lambda I) )^{-1} g    over K-FAC-able layers,
  d0 = -g                                        for every other parameter,

where each per-layer block F_l is the Kronecker-factored curvature F_l = G_l (x) A_l
(output factor G_l = E[delta delta^T], input factor A_l = E[a a^T]) that curvlinops
estimates by hooking the forward / backward pass of every supported layer. Because the
factors are collected by *local* per-layer hooks - not by tracing the module graph -
the provider is architecture-agnostic: skip connections, BatchNorm inside the forward,
and custom ``forward`` methods are transparent, so the *whole* network is evaluated
(graph-tracing K-FAC, by contrast, is limited to plain sequential stacks).

Supported layers are ``Linear`` and ``Conv2d``. A parameter outside them (e.g. a
BatchNorm / LayerNorm affine weight) carries no Kronecker curvature and receives the
identity step ``-g`` - the well-defined limit of a damped Newton step with no curvature.
The split is per *parameter*, so a group mixing a convolution with its normalisation is
handled correctly.

The per-layer block solve uses curvlinops' *factored* Tikhonov damping: each Kronecker
factor is damped and Cholesky-inverted on its own,

  d0|_l = -(G_l + lambda I)^{-1} grad_W_l (A_l + lambda I)^{-1},

the canonical K-FAC update (Martens & Grosse, 2015, Section 6.3). This is the damped
block-diagonal natural-gradient step the overlay corrects. Factoring the damping avoids
the eigendecomposition of the *raw* factors G_l, A_l that exact (joint-eigenbasis)
damping needs - an ``eigh`` that fails to converge in float32 on the rank-deficient
input factors of a deep convolutional network (A_l is C_in k^2 wide but has rank at most
batch * spatial) - and curvlinops retries a factor's Cholesky in float64 should it be
numerically singular. For a single linear layer K-FAC *is* the exact GGN (G_l (x) A_l),
so as lambda -> 0 the step converges to the exact GGN Newton step -(GGN)^{-1} g (the
adapter test asserts this lambda -> 0 limit); the factored and joint dampings differ only
in how the finite-lambda Tikhonov term is distributed. With a bias or a multi-class
cross-entropy, K-FAC is - as always - an approximation, which is precisely what the
coupling-gated overlay and the no-harm safeguard exist to handle.

The flat parameter layout - and therefore ``block_ranges`` - is the group-major,
parameter-major concatenation of ``model.get_param_groups()``, identical to the layout
the parameter-space estimator (`hessian.param_space`) and the overlay's HVP operate on,
so the base direction, the coupling, and the correction all compose. The K-FAC-able
parameters are handed to curvlinops in that same order, so the operator's vector layout
matches the gradient's supported sub-vector.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn
from curvlinops import KFACInverseLinearOperator, KFACLinearOperator
from torch import Tensor
from torch.nn import Parameter

from hessian.param_space import ParamGroupedModel
from hessian.types import LayerID

# curvlinops K-FAC factorises these layer types; any other parameter (norm affine
# weights, etc.) has no Kronecker curvature and falls back to the identity step.
_SUPPORTED: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d)


def _build_ranges(groups: dict[LayerID, list[Parameter]]) -> dict[LayerID, slice]:
    """Group-major slices of the flat parameter vector (matches the estimator layout)."""
    ranges: dict[LayerID, slice] = {}
    offset = 0
    for name, params in groups.items():
        size = sum(p.numel() for p in params)
        ranges[name] = slice(offset, offset + size)
        offset += size
    return ranges


class CurvlinopsKFACProvider:
    """K-FAC provider backed by curvlinops (structurally implements `KFACProvider`).

    Stateful by design: K-FAC curvature changes every step, so ``update(x, y)``
    rebuilds the Kronecker-factored operator on a batch and ``base_direction(g)``
    returns the damped block-diagonal Newton step for that batch. The model is used
    as-is - the caller owns its mode (e.g. BatchNorm in eval during factor collection)
    and its gradients (``base_direction`` never touches the autograd graph).
    """

    def __init__(
        self,
        model: ParamGroupedModel,
        loss_fn: nn.Module,
        *,
        damping: float = 1e-2,
        fisher_type: str = "type-2",
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._damping = damping
        self._fisher_type = fisher_type

        self._groups = model.get_param_groups()
        self._ranges = _build_ranges(self._groups)

        # Per-parameter split (in flat-layout order): the Linear / Conv2d parameters go
        # to curvlinops; the rest take the identity step. ``model`` is an nn.Module (it
        # is handed to curvlinops as one), so walking its modules to classify is sound.
        supported_ids = {
            id(p)
            for module in cast(nn.Module, model).modules()
            if isinstance(module, _SUPPORTED)
            for p in module.parameters(recurse=False)
        }
        total = sum(p.numel() for p in self._iter_params())
        sup_mask = torch.zeros(total, dtype=torch.bool)
        kfac_params: list[Parameter] = []
        offset = 0
        for p in self._iter_params():
            n = p.numel()
            if id(p) in supported_ids:
                kfac_params.append(p)
                sup_mask[offset : offset + n] = True
            offset += n
        self._kfac_params = kfac_params
        self._sup_mask = sup_mask

        self._inverse: KFACInverseLinearOperator | None = None
        self._ready = False

    # -- KFACProvider --------------------------------------------------
    def base_direction(self, grad: Tensor) -> Tensor:
        """d0 = -(blkdiag_l (F_l + lambda I))^{-1} grad on K-FAC-able layers, -grad else."""
        if not self._ready:
            raise RuntimeError("call update(x, y) before base_direction(grad)")
        grad = grad.detach()
        d0 = -grad  # identity (gradient-descent) step ...
        if self._inverse is not None:
            if self._sup_mask.device != grad.device:
                self._sup_mask = self._sup_mask.to(grad.device)
            mask = self._sup_mask
            d0[mask] = -(self._inverse @ grad[mask])  # ... K-FAC where it applies
        return d0

    def block_ranges(self) -> dict[LayerID, slice]:
        """Contiguous slice of the flat parameter vector owned by each group."""
        return self._ranges

    # -- curvature refresh ---------------------------------------------
    def update(self, x: Tensor, y: Tensor) -> CurvlinopsKFACProvider:
        """Rebuild the Kronecker-factored inverse on batch (x, y) (one K-FAC step)."""
        if self._kfac_params:
            operator = KFACLinearOperator(
                self._model,
                self._loss_fn,
                self._kfac_params,
                [(x, y)],
                fisher_type=self._fisher_type,
                check_deterministic=False,
            )
            # Factored Tikhonov damping (Cholesky on each damped factor, with a
            # float64 retry on a singular factor); the exact joint-eigenbasis damping
            # would instead run ``eigh`` on the raw factors, which fails to converge in
            # float32 on a deep conv net's rank-deficient input factors.
            self._inverse = KFACInverseLinearOperator(operator, damping=self._damping)
        self._ready = True
        return self

    # -- internals -----------------------------------------------------
    def _iter_params(self) -> Iterator[Parameter]:
        for params in self._groups.values():
            yield from params
