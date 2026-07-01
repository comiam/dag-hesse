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

import copy
import logging
from typing import cast

import torch
import torch.nn as nn
from curvlinops import (
    EKFACLinearOperator,
    KFACInverseLinearOperator,
    KFACLinearOperator,
)
from torch import Tensor
from torch.nn import Parameter

from hessian.param_space import ParamGroupedModel
from hessian.types import LayerID

logger = logging.getLogger(__name__)

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


def _classify_supported(
    model: ParamGroupedModel, groups: dict[LayerID, list[Parameter]]
) -> tuple[list[Parameter], Tensor]:
    """Split the group-major flat layout into curvature-able params and a support mask.

    The ``Linear`` / ``Conv2d`` parameters - the ones carrying Kronecker curvature - are
    returned in group-major order (the layout curvlinops receives), and ``sup_mask`` flags
    their positions in the flat parameter vector so ``base_direction`` routes the curvature
    step to them and the identity step ``-g`` to every other parameter (norm affines, etc.).
    ``model`` is walked as an ``nn.Module`` to classify the modules owning each parameter.
    """
    supported_ids = {
        id(p)
        for module in cast(nn.Module, model).modules()
        if isinstance(module, _SUPPORTED)
        for p in module.parameters(recurse=False)
    }
    total = sum(p.numel() for params in groups.values() for p in params)
    sup_mask = torch.zeros(total, dtype=torch.bool)
    kfac_params: list[Parameter] = []
    offset = 0
    for params in groups.values():
        for p in params:
            n = p.numel()
            if id(p) in supported_ids:
                kfac_params.append(p)
                sup_mask[offset : offset + n] = True
            offset += n
    return kfac_params, sup_mask


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
        # to curvlinops; the rest take the identity step.
        self._kfac_params, self._sup_mask = _classify_supported(model, self._groups)

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
            try:
                d0[mask] = -(self._inverse @ grad[mask])  # ... K-FAC where it applies
            except torch.linalg.LinAlgError:
                # A degenerate Kronecker factor (non-positive-definite Cholesky) on this
                # minibatch: take the identity step -grad on the K-FAC-able params rather
                # than aborting the run (the no-harm safeguard, when active, backstops it).
                logger.warning(
                    "K-FAC factor inversion failed (non-positive-definite factor); "
                    "taking the gradient step on this minibatch."
                )
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


class CurvlinopsEKFACProvider:
    """EKFAC provider backed by curvlinops (structurally implements `KFACProvider`).

    EKFAC (eigenvalue-corrected K-FAC; George et al., 2018) keeps the K-FAC Kronecker
    eigenbasis but rescales it by the exact diagonal variances measured in that basis - a
    strictly finer block-diagonal curvature than K-FAC, and the second block-diagonal
    baseline that demonstrates the overlay is optimizer-agnostic. It is wired exactly like
    `CurvlinopsKFACProvider` - the same group-major layout, the same Linear/Conv2d support
    split, the same identity step ``-g`` for norm parameters - with two curvlinops-imposed
    differences forced by the eigenvalue correction:

      * EKFAC supports only *exact* (eigenbasis) damping, so the inverse is built with
        ``use_exact_damping=True`` instead of the factored Tikhonov damping K-FAC uses;
      * that eigendecomposition of the raw Kronecker factors fails to converge (LAPACK
        ``syevd`` on an ill-conditioned matrix) on the rank-deficient input factors of a
        convolutional network in float32 - the same ill-conditioning that pushed K-FAC to
        factored damping. EKFAC cannot sidestep the eigenbasis, so the factors are
        collected and the eigendecomposition is run in float64 on a private double-
        precision copy of the model, using the ``reduce`` Kronecker approximation (full-
        rank C x C factors, which converge where the ``expand`` factors - C k^2 wide,
        rank <= batch * spatial - do not). The resulting double direction is cast back to
        the parameter dtype.

    The double copy is built once and kept in eval mode; its weights and BatchNorm buffers
    are synced from the live model on each refresh, so it tracks training without disturbing
    the float32 model the estimator and optimizer share. The eigenbasis is refreshed only
    every ``refresh_period`` updates (the eigendecomposition is the dominant cost), and a
    non-convergent refresh keeps the previous eigenbasis rather than aborting the run.
    """

    def __init__(
        self,
        model: ParamGroupedModel,
        loss_fn: nn.Module,
        *,
        damping: float = 1e-2,
        fisher_type: str = "type-2",
        refresh_period: int = 1,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._damping = damping
        self._fisher_type = fisher_type
        self._refresh_period = max(1, refresh_period)
        self._calls = 0

        self._groups = model.get_param_groups()
        self._ranges = _build_ranges(self._groups)
        _, self._sup_mask = _classify_supported(model, self._groups)

        # Private float64 copy, kept in eval mode: EKFAC's eigenbasis is built here so the
        # eigendecomposition runs in double precision on frozen-BatchNorm factors (the
        # float32 / train-mode eigh diverges on conv factors). Its supported params, in the
        # same group-major order, feed curvlinops.
        self._double_model = copy.deepcopy(cast(nn.Module, model)).double().eval()
        double_groups = cast(ParamGroupedModel, self._double_model).get_param_groups()
        self._kfac_params, _ = _classify_supported(
            cast(ParamGroupedModel, self._double_model), double_groups
        )
        self._dim_supported = sum(p.numel() for p in self._kfac_params)
        self._device = (
            self._kfac_params[0].device if self._kfac_params else torch.device("cpu")
        )

        self._inverse: KFACInverseLinearOperator | None = None
        self._ready = False

    # -- KFACProvider --------------------------------------------------
    def base_direction(self, grad: Tensor) -> Tensor:
        """d0 = -(EKFAC + lambda I)^{-1} grad on supported layers, -grad everywhere else."""
        if not self._ready:
            raise RuntimeError("call update(x, y) before base_direction(grad)")
        grad = grad.detach()
        d0 = -grad  # identity (gradient-descent) step ...
        if self._inverse is not None:
            if self._sup_mask.device != grad.device:
                self._sup_mask = self._sup_mask.to(grad.device)
            mask = self._sup_mask
            # Solve in float64 (the eigenbasis dtype), then return in the param dtype.
            sol = self._inverse @ grad[mask].double()
            d0[mask] = -sol.to(grad.dtype)  # ... EKFAC where it applies
        return d0

    def block_ranges(self) -> dict[LayerID, slice]:
        """Contiguous slice of the flat parameter vector owned by each group."""
        return self._ranges

    # -- curvature refresh ---------------------------------------------
    def update(self, x: Tensor, y: Tensor) -> CurvlinopsEKFACProvider:
        """Refresh the eigenvalue-corrected inverse every ``refresh_period`` updates."""
        if self._kfac_params and self._calls % self._refresh_period == 0:
            self._refresh(x, y)
        self._calls += 1
        self._ready = True
        return self

    def _refresh(self, x: Tensor, y: Tensor) -> None:
        """Rebuild the float64 eigenbasis on the current weights; keep the last on failure."""
        live = cast(nn.Module, self._model)
        self._double_model.load_state_dict(live.state_dict())  # sync weights + buffers
        operator = EKFACLinearOperator(
            self._double_model,
            self._loss_fn,
            self._kfac_params,
            [(x.double(), y)],
            fisher_type=self._fisher_type,
            kfac_approx="reduce",
            check_deterministic=False,
        )
        inverse = KFACInverseLinearOperator(
            operator, damping=self._damping, use_exact_damping=True
        )
        try:
            # Force (and cache) the eigendecomposition now so a non-convergent factor
            # surfaces here, not mid-step; the cached result then serves base_direction.
            probe = torch.zeros(
                self._dim_supported, dtype=torch.float64, device=self._device
            )
            _ = inverse @ probe
        except torch.linalg.LinAlgError:
            logger.warning(
                "EKFAC eigendecomposition did not converge on this minibatch; keeping "
                "the previous eigenbasis (gradient step until the next successful build)."
            )
            return
        self._inverse = inverse
