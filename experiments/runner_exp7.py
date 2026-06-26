"""Exp7: COUPLE-FAC overlay optimization on a real classification task (the repair experiment).

This runner *uses* the inter-layer curvature the rest of the project *measures*: it
trains one backbone with several optimizers and compares them by how fast they reach a
target validation accuracy and by their final accuracy. The comparison set is

  - ``sgd`` / ``adam``     : first-order references (the built-in Trainer optimizer);
  - ``kfac``              : the block-diagonal K-FAC Newton step (overlay disabled);
  - ``kfac+overlay``      : COUPLE-FAC - K-FAC plus the coupling-gated, trust-region
                            safeguarded rank-r cross-block correction (the method);
  - ``newton_cg``         : the exact damped-Newton oracle (truncated CG on the exact
                            HVP), the upper bound whose gap to K-FAC the overlay closes.

The three second-order methods all run through the `Trainer` step-hook seam (Delta-M4):
each step refreshes the K-FAC provider's curvature on the minibatch, builds the exact
parameter-space bundle (gradient, HVP, inter-block coupling) with
`hessian.param_space.ParamBlockEstimator`, and applies one accepted update. ``kfac`` and
``kfac+overlay`` differ only in the selection threshold tau (tau = +inf disables the
overlay, so S = {C_{vw} > tau} is empty and d = d_KFAC); ``newton_cg`` solves
(H + lambda I) d = -g directly. The coupling C is re-measured only every T steps
(``measure_period``), amortizing its Hutchinson cost.

All methods share one fixed train/val split and the same backbone factory, so the only
varying factor is the optimizer; ``training.seeds`` controls model init and minibatch
order (>= 5 seeds per the revision plan). The result bundle is JSON-serializable:
per-seed accuracy curves plus mean/std final accuracy and mean epochs-to-target.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader

from experiments.config import Exp7Config
from experiments.data import (
    get_cifar10_image_loaders,
    get_cifar100_loaders,
    get_stanford_cars_loaders,
)
from experiments.models import SegmentedResNet18, SegmentedResNet50
from experiments.training import StepCtx, Trainer
from experiments.utils import set_seed
from hessian import ParamBlockEstimator, ParamGroupedModel
from secondorder import CoupleFacOptimizer, CoupleFacOverlay, HessianOracle, Pair
from secondorder.kfac_adapter import CurvlinopsKFACProvider

logger = logging.getLogger(__name__)

# Reference first-order Adam learning rate (the Trainer config carries the SGD lr).
_ADAM_LR = 1e-3
# Step size for the damped-Newton family: a full damped-Newton step, the natural unit
# for d = -(H + lambda I)^{-1} g; the no-harm safeguard backstops it.
_SECOND_ORDER_LR = 1.0
# Truncated-CG iteration budget for the Newton-CG oracle (HVPs per step).
_NEWTON_CG_ITERS = 20
# Fixed seed for the train/val split, so every method and seed sees the same data.
_SPLIT_SEED = 0

_SECOND_ORDER_METHODS = frozenset({"kfac", "kfac+overlay", "newton_cg"})


# ======================================================================
# Update direction helpers
# ======================================================================


def _group_major_params(model: ParamGroupedModel) -> list[Parameter]:
    """Flat parameter list in `get_param_groups` (group-major) order.

    This is the layout the K-FAC provider's `block_ranges`, the parameter-space
    estimator, and the optimizer all share, so the gradient, the HVP, and the applied
    direction compose coordinate-for-coordinate.
    """
    return [p for params in model.get_param_groups().values() for p in params]


def _apply_flat(params: list[Parameter], direction: Tensor, lr: float) -> None:
    """theta <- theta + lr * direction, unflattened group-major over ``params``."""
    with torch.no_grad():
        offset = 0
        for p in params:
            n = p.numel()
            p.add_(lr * direction[offset : offset + n].view_as(p))
            offset += n


def _newton_cg(
    grad: Tensor,
    hvp: HessianOracle,
    *,
    damping: float,
    max_iter: int,
    tol: float = 1e-6,
) -> Tensor:
    """Truncated-CG solve of (H + lambda I) d = -grad (the exact-Newton oracle step).

    Linear conjugate gradients on the damped exact Hessian, terminated early on a
    negative-curvature direction (p^T A p <= 0) - the standard Newton-CG safeguard for a
    non-convex H. If the very first direction has negative curvature, it falls back to
    steepest descent d = -grad. Each iteration costs one exact HVP.
    """
    b = -grad
    d = torch.zeros_like(grad)
    r = b.clone()
    p = r.clone()
    rs = float(r.dot(r))
    if rs <= tol * tol:
        return d
    for _ in range(max_iter):
        ap = hvp(p) + damping * p
        pap = float(p.dot(ap))
        if pap <= 0.0:
            if float(d.norm()) == 0.0:
                d = b  # negative curvature on the first step -> steepest descent
            break
        alpha = rs / pap
        d = d + alpha * p
        r = r - alpha * ap
        rs_new = float(r.dot(r))
        if rs_new <= tol * tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return d


# ======================================================================
# Step-hook factories (the COUPLE-FAC / Newton-CG seam into Trainer)
# ======================================================================


def build_couplefac_hook(
    model: ParamGroupedModel,
    cfg: Exp7Config,
    *,
    tau: float,
    seed: int,
    device: torch.device,
) -> Callable[[StepCtx], None]:
    """A Trainer step hook running one COUPLE-FAC update per minibatch.

    With ``tau = +inf`` the selection set S = {C_{vw} > tau} is empty and the step is
    the plain K-FAC direction (the ``kfac`` baseline); with a finite ``tau`` the
    coupling-gated rank-r overlay is added and the no-harm safeguard accepts it only if
    it does not worsen the exact local quadratic model. The coupling C is recomputed
    only every ``measure_period`` steps.
    """
    loss_fn = nn.CrossEntropyLoss()
    provider = CurvlinopsKFACProvider(
        model, loss_fn, damping=cfg.damping, fisher_type=cfg.fisher_type
    )
    overlay = CoupleFacOverlay(
        rank=cfg.overlay_rank, n_power_iter=cfg.overlay_n_power_iter
    )
    optimizer = CoupleFacOptimizer(
        provider=provider,
        overlay=overlay,
        tau=tau,
        radius=cfg.tr_radius,
        params=_group_major_params(model),
        lr=_SECOND_ORDER_LR,
    )
    estimator = ParamBlockEstimator(model, loss_fn, n_probes=cfg.coupling_n_probes)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    overlay_active = math.isfinite(tau)
    period = max(1, cfg.measure_period)
    last_coupling: dict[Pair, float] = {}

    def hook(ctx: StepCtx) -> None:
        nonlocal last_coupling
        provider.update(ctx.x, ctx.y)  # refresh K-FAC curvature on this minibatch
        cur = estimator.curvature(ctx.x, ctx.y)  # exact grad + HVP + deferred coupling
        if overlay_active and ctx.step % period == 0:
            last_coupling = cur.coupling()  # amortized: re-measure C every T steps
        coupling = last_coupling if overlay_active else {}
        optimizer.step(cur.flat_grad, cur.hvp, coupling, generator=generator)
        del cur  # release the create_graph backward graph held by the bundle

    return hook


def build_newton_cg_hook(
    model: ParamGroupedModel,
    cfg: Exp7Config,
    *,
    device: torch.device,
) -> Callable[[StepCtx], None]:
    """A Trainer step hook taking the exact damped-Newton (Newton-CG oracle) step."""
    loss_fn = nn.CrossEntropyLoss()
    estimator = ParamBlockEstimator(model, loss_fn, n_probes=cfg.coupling_n_probes)
    flat_params = _group_major_params(model)

    def hook(ctx: StepCtx) -> None:
        cur = estimator.curvature(ctx.x, ctx.y)
        direction = _newton_cg(
            cur.flat_grad, cur.hvp, damping=cfg.damping, max_iter=_NEWTON_CG_ITERS
        )
        _apply_flat(flat_params, direction, _SECOND_ORDER_LR)
        del cur

    return hook


# ======================================================================
# Dataset / backbone construction
# ======================================================================


def _build_loaders(cfg: Exp7Config) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) for the configured dataset (fixed split)."""
    batch_size = cfg.training.batch_size
    if cfg.dataset == "stanford_cars":
        return get_stanford_cars_loaders(
            batch_size,
            image_size=cfg.image_size,
            augment=cfg.augment,
            seed=_SPLIT_SEED,
        )
    if cfg.dataset == "cifar10":
        return get_cifar10_image_loaders(batch_size)
    if cfg.dataset == "cifar100":
        return get_cifar100_loaders(batch_size)
    if cfg.dataset == "imagenet32":
        raise NotImplementedError(
            "the imagenet32 loader (Exp7 B2 control) is not implemented yet; "
            "add it before running Exp7Config.b2_largebatch"
        )
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def _build_model(cfg: Exp7Config) -> SegmentedResNet18:
    """Backbone for the configured registry key."""
    if cfg.model == "resnet50":
        return SegmentedResNet50(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    if cfg.model == "resnet18":
        return SegmentedResNet18(num_classes=cfg.num_classes)
    raise ValueError(f"unknown model {cfg.model!r}; expected 'resnet50' or 'resnet18'")


# ======================================================================
# Metrics
# ======================================================================


def _epochs_to_target(curve: list[float], target: float) -> int | None:
    """First 1-indexed epoch whose validation accuracy reaches ``target`` (else None)."""
    for epoch, acc in enumerate(curve, start=1):
        if acc >= target:
            return epoch
    return None


def _aggregate_method(per_seed: dict[str, Any], target: float) -> dict[str, Any]:
    """Mean/std final accuracy and mean epochs-to-target over seeds."""
    finals = [payload["final_acc"] for payload in per_seed.values()]
    reached = [
        payload["epochs_to_target"]
        for payload in per_seed.values()
        if payload["epochs_to_target"] is not None
    ]
    return {
        "seeds": per_seed,
        "target_acc": target,
        "final_acc_mean": float(np.mean(finals)),
        "final_acc_std": float(np.std(finals)),
        "epochs_to_target_mean": float(np.mean(reached)) if reached else None,
        "n_reached_target": len(reached),
        "n_seeds": len(per_seed),
    }


# ======================================================================
# Runner
# ======================================================================


class Exp7Runner:
    """Runner for Exp7 (COUPLE-FAC overlay vs. first-/second-order baselines)."""

    def __init__(self, config: Exp7Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {method: {seeds: {...}, final_acc_mean, epochs_to_target_mean, ...}}."""
        train_loader, val_loader = _build_loaders(self._cfg)
        results: dict[str, Any] = {}

        for method in self._cfg.methods:
            logger.info("=== Method: %s ===", method)
            per_seed: dict[str, Any] = {}

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)
                set_seed(seed)

                model = _build_model(self._cfg).to(self._device)
                trainer = self._make_trainer(method, model, seed)
                accs, losses = self._train_with_curve(trainer, train_loader, val_loader)

                per_seed[f"seed_{seed}"] = {
                    "val_acc": accs,
                    "val_loss": losses,
                    "final_acc": accs[-1] if accs else float("nan"),
                    "epochs_to_target": _epochs_to_target(accs, self._cfg.target_acc),
                }

            results[method] = _aggregate_method(per_seed, self._cfg.target_acc)

        return results

    # ------------------------------------------------------------------
    # Per-method wiring
    # ------------------------------------------------------------------

    def _make_trainer(
        self, method: str, model: SegmentedResNet18, seed: int
    ) -> Trainer:
        """Builds the Trainer for ``method``: optimizer slot or a second-order hook."""
        cfg = self._cfg
        if method == "sgd":
            return Trainer(model, replace(cfg.training, optimizer="sgd"), self._device)
        if method == "adam":
            tcfg = replace(cfg.training, optimizer="adam", lr=_ADAM_LR)
            return Trainer(model, tcfg, self._device)
        if method in ("kfac", "kfac+overlay"):
            tau = math.inf if method == "kfac" else cfg.tau
            hook = build_couplefac_hook(
                model, cfg, tau=tau, seed=seed, device=self._device
            )
            return Trainer(model, cfg.training, self._device, step_hook=hook)
        if method == "newton_cg":
            hook = build_newton_cg_hook(model, cfg, device=self._device)
            return Trainer(model, cfg.training, self._device, step_hook=hook)
        raise ValueError(
            f"unknown method {method!r}; expected one of "
            "{'sgd', 'adam', 'kfac', 'kfac+overlay', 'newton_cg'}"
        )

    def _train_with_curve(
        self,
        trainer: Trainer,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[list[float], list[float]]:
        """Trains and records the per-epoch validation accuracy / loss curve."""
        accs: list[float] = []
        losses: list[float] = []

        def record(_epoch: int) -> None:
            evaluation = trainer.evaluate(val_loader)
            accs.append(evaluation["test_acc"])
            losses.append(evaluation["test_loss"])

        trainer.train(train_loader, checkpoint_epochs=[], on_epoch_end=record)
        return accs, losses
