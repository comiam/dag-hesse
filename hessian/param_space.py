"""
Parameter-space inter-layer Hessian blocks H_{theta_v, theta_w} and the
discarded-coupling fraction Phi.

Where the activation-space estimator (`stochastic.StochasticHessianEstimator`)
measures curvature between *activations* f_v, f_w, this module measures curvature
between *parameter groups* theta_v, theta_w - the quantity a block-diagonal
preconditioner (K-FAC / EKFAC) actually sees, and the off-diagonal mass it discards.

Parameter Hessian block (exact action, one double-backward):
  (H u)_{theta_v} = grad_{theta_v} < grad_{theta_w} L , u_{theta_w} >
                  = H_{theta_v, theta_w} u_{theta_w}.

Block Frobenius norm via Hutchinson (no full matrix):
  ||H_{theta_v, theta_w}||_F^2 = E_z[ || H_{theta_v, theta_w} z ||^2 ],  z ~ Rademacher(d_w).

Discarded-coupling fraction (the fraction of total curvature mass any
block-diagonal preconditioner discards):
  Phi = ( sum_{v != w} ||H_{theta_v, theta_w}||_F^2 ) / ||H||_F^2
      = 1 - ( sum_v ||H_{theta_v, theta_v}||_F^2 ) / ||H||_F^2   in [0, 1).

Estimator identity: ||H||_F^2 from a full-parameter Rademacher probe; each diagonal
||H_{theta_v,theta_v}||_F^2 from a probe supported only on group v - all sharing one
backward graph g = grad_theta L (create_graph=True), so the second backward
differentiates that single graph.

The same single-graph machinery backs `curvature(x, y)`: the per-minibatch bundle
(detached group-major gradient, exact full-parameter HVP, and a deferred inter-block
coupling) that a parameter-space second-order step (COUPLE-FAC) consumes.

Complexity: O((L+1) m) HVPs for Phi (L groups + full), O(P) memory.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import Parameter

from ._estimators import HVPClosure, hutchinson_frob_sq
from .metrics import coupling_from_norms
from .types import LayerID

_EPS = 1e-12


class ParamGroupedModel(Protocol):
    """A model whose trainable parameters are partitioned into named groups.

    Groups are the parameter-space analogue of the activation measurement points
    in `SegmentedModel`. The model must be callable (x -> logits) so the loss graph
    can be built directly in parameter space.
    """

    def __call__(self, x: Tensor) -> Tensor: ...
    def get_param_groups(self) -> dict[LayerID, list[Parameter]]: ...


@dataclass(frozen=True)
class ParamCurvature:
    """Per-minibatch curvature bundle for one parameter-space second-order step.

    The three members share the single backward graph g = grad_theta L
    (create_graph=True) the estimator builds, so they are mutually consistent and cheap
    to obtain together; holding this object keeps that graph alive (discard it after the
    step to free it).

      - ``flat_grad``: the *detached* parameter gradient, concatenated in
        `get_param_groups` (group-major) order - the layout a K-FAC provider's
        `block_ranges` and the optimizer's parameter list must match;
      - ``hvp``: the exact full-parameter Hessian-vector product z |-> H z;
      - ``coupling``: a *deferred* call returning the inter-block coupling C(v, w) for
        every pair v before w in group order. It costs one set of Hutchinson probes per
        block, so it is kept behind a call for the runner to amortize over T steps.
    """

    flat_grad: Tensor
    hvp: HVPClosure
    coupling: Callable[[], dict[tuple[LayerID, LayerID], float]]


@dataclass
class _ParamGraph:
    """A single backward graph g = grad_theta L, grouped by parameter block.

    Holds everything the HVP closures need so the graph is built once and reused
    across all probes / blocks (the second backward differentiates this graph).
    """

    names: list[LayerID]
    params: dict[LayerID, list[Parameter]]
    grads: dict[LayerID, list[Tensor]]
    dims: dict[LayerID, int]
    device: torch.device

    @property
    def all_params(self) -> list[Parameter]:
        return [p for name in self.names for p in self.params[name]]

    @property
    def all_grads(self) -> list[Tensor]:
        return [g for name in self.names for g in self.grads[name]]

    @property
    def total_dim(self) -> int:
        return sum(self.dims[name] for name in self.names)


class ParamBlockEstimator:
    """Hutchinson estimation of parameter-space blocks H_{theta_v, theta_w} and Phi."""

    def __init__(
        self,
        model: ParamGroupedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        n_probes: int = 30,
        n_power_iter: int = 20,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._n_probes = n_probes
        self._n_power_iter = n_power_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_block_frob_norm(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """||H_{theta_v, theta_w}||_F via Hutchinson on the cross-block HVP."""
        graph = self._grad_graph(x, y)
        return self._block_frob_sq(graph, layer_v, layer_w) ** 0.5

    def estimate_full_frob_sq(self, x: Tensor, y: Tensor) -> float:
        """||H||_F^2 (full parameter Hessian) via Hutchinson."""
        return self._full_frob_sq(self._grad_graph(x, y))

    def estimate_diag_frob_sq(self, x: Tensor, y: Tensor, layer_v: LayerID) -> float:
        """||H_{theta_v, theta_v}||_F^2 (a single diagonal block)."""
        return self._block_frob_sq(self._grad_graph(x, y), layer_v, layer_v)

    def estimate_phi(self, x: Tensor, y: Tensor) -> float:
        """Phi = 1 - sum_v ||H_{vv}||_F^2 / ||H||_F^2  in [0, 1)  (Delta-M1).

        Built on a single backward graph: one full-parameter probe set for ||H||_F^2
        and one group-localized probe set per diagonal block.
        """
        graph = self._grad_graph(x, y)
        full_sq = self._full_frob_sq(graph)
        if full_sq < _EPS:
            return math.nan
        diag_sq = sum(self._block_frob_sq(graph, name, name) for name in graph.names)
        return 1.0 - diag_sq / full_sq

    def estimate_coupling(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """C(v,w) = ||H_{vw}||_F / sqrt(||H_{vv}||_F ||H_{ww}||_F) in parameter space.

        The three blocks share one backward graph. As in activation space, C > 1 is
        possible when the off-diagonal block is non-PSD; it is left unclamped.
        """
        graph = self._grad_graph(x, y)
        r_vw = self._block_frob_sq(graph, layer_v, layer_w) ** 0.5
        r_vv = self._block_frob_sq(graph, layer_v, layer_v) ** 0.5
        r_ww = self._block_frob_sq(graph, layer_w, layer_w) ** 0.5
        return coupling_from_norms(r_vw, r_vv, r_ww)

    def curvature(self, x: Tensor, y: Tensor) -> ParamCurvature:
        """Per-minibatch curvature bundle feeding one COUPLE-FAC step (Delta-M3).

        Builds the single backward graph g = grad_theta L (create_graph=True) once and
        exposes, all sharing it: the detached group-major gradient ``flat_grad``; the
        exact full-parameter HVP ``hvp`` (z |-> H z); and ``coupling`` - a deferred call
        returning the inter-block coupling C(v, w), kept lazy because it costs one set of
        Hutchinson probes per block and the caller amortizes it over T steps.
        """
        graph = self._grad_graph(x, y)
        flat_grad = torch.cat([g.reshape(-1) for g in graph.all_grads]).detach()
        hvp = self._cross_block_hvp(graph.all_params, graph.all_grads)
        return ParamCurvature(
            flat_grad=flat_grad,
            hvp=hvp,
            coupling=lambda: self._coupling_dict(graph),
        )

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _grad_graph(self, x: Tensor, y: Tensor) -> _ParamGraph:
        """Builds g = grad_theta L (create_graph=True), grouped by parameter block."""
        groups = self._model.get_param_groups()
        names = list(groups.keys())
        params = {name: list(groups[name]) for name in names}

        flat_params = [p for name in names for p in params[name]]
        loss = self._loss_fn(self._model(x), y)
        flat_grads = torch.autograd.grad(loss, flat_params, create_graph=True)

        grads: dict[LayerID, list[Tensor]] = {}
        dims: dict[LayerID, int] = {}
        offset = 0
        for name in names:
            k = len(params[name])
            grads[name] = list(flat_grads[offset : offset + k])
            dims[name] = sum(p.numel() for p in params[name])
            offset += k

        return _ParamGraph(
            names=names,
            params=params,
            grads=grads,
            dims=dims,
            device=flat_params[0].device,
        )

    # ------------------------------------------------------------------
    # Block estimators
    # ------------------------------------------------------------------

    def _block_frob_sq(self, graph: _ParamGraph, v: LayerID, w: LayerID) -> float:
        """||H_{theta_v, theta_w}||_F^2 via Hutchinson on the cross-block HVP."""
        hvp = self._cross_block_hvp(graph.params[v], graph.grads[w])
        return hutchinson_frob_sq(hvp, graph.dims[w], self._n_probes, graph.device)

    def _full_frob_sq(self, graph: _ParamGraph) -> float:
        """||H||_F^2 via Hutchinson on the full-parameter HVP."""
        hvp = self._cross_block_hvp(graph.all_params, graph.all_grads)
        return hutchinson_frob_sq(hvp, graph.total_dim, self._n_probes, graph.device)

    def _coupling_dict(
        self, graph: _ParamGraph
    ) -> dict[tuple[LayerID, LayerID], float]:
        """All inter-block couplings C(v, w), v before w in group order, from one graph.

        Each diagonal norm is measured once and reused across the pairs that contain it -
        cheaper than `estimate_coupling`, which re-measures both diagonals per pair (and
        rebuilds the graph).
        """
        names = graph.names
        diag = {v: self._block_frob_sq(graph, v, v) ** 0.5 for v in names}
        coupling: dict[tuple[LayerID, LayerID], float] = {}
        for i, v in enumerate(names):
            for w in names[i + 1 :]:
                r_vw = self._block_frob_sq(graph, v, w) ** 0.5
                coupling[(v, w)] = coupling_from_norms(r_vw, diag[v], diag[w])
        return coupling

    @staticmethod
    def _cross_block_hvp(
        out_params: list[Parameter],
        grad_probe: list[Tensor],
    ) -> HVPClosure:
        """Builds z |-> (H z) restricted to ``out_params``  (Delta-M3).

        z is a flat Rademacher probe over group w, the block behind ``grad_probe``
        (= dL/d theta_w); the returned vector is the ``out_params`` (group v) slice
        of H z:
          grad_{out_params} < grad_probe, z > = H_{out, w} z.

        Specializations (one helper, three uses):
          - cross block:  out = theta_v, grad_probe = dL/d theta_w;
          - diagonal:     out = theta_v, grad_probe = dL/d theta_v;
          - full HVP:     out = all params, grad_probe = dL/d theta.

        ``allow_unused=True`` zero-fills structurally disconnected coordinates, which
        is exactly their (vanishing) second-order contribution.
        """
        flat_grad = torch.cat([g.reshape(-1) for g in grad_probe])

        def hvp(z: Tensor) -> Tensor:
            dot = flat_grad.dot(z)
            grads = torch.autograd.grad(
                dot, out_params, retain_graph=True, allow_unused=True
            )
            return torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(grads, out_params, strict=True)
                ]
            )

        return hvp
