"""Exp8: Stage-A Phi diagnostic - the K-FAC blind-spot map (the diagnosis experiment).

This runner *measures* the inter-layer curvature the optimizer experiment (exp7) later
*repairs*. On a real convolutional DAG it reports, with no change to training, the
discarded-coupling fraction

  Phi = 1 - sum_v ||H_{theta_v, theta_v}||_F^2 / ||H||_F^2  in [0, 1)   (Delta-M1)

- the share of total curvature mass any block-diagonal preconditioner (K-FAC / EKFAC)
throws away - together with where that mass sits: the per-block diagonal norms
||H_{vv}||_F it keeps and the coupling field C(v, w) it discards (Delta-M2). All three
come from one parameter-space backward graph via `ParamBlockEstimator.estimate_phi_report`.

Two regimes share the runner (`Exp8Config` profiles):

  - ``a1`` (from scratch, the inversion headline): `SegmentedResNet18` vs its skip-free
    control `SegmentedPlainResNet18` on CIFAR-100, Phi tracked at the init / mid / final
    checkpoints. The headline is the inversion Phi_ResNet >> Phi_Plain - skips preserve
    inter-layer coupling, so K-FAC discards more curvature on the ResNet (source (S)).
  - ``a2`` (pretrained): an ImageNet-pretrained ResNet-50 measured pre-finetune on
    Stanford Cars (``epochs = 0`` -> the single ``init`` checkpoint), where correlated
    pretrained Jacobians inflate Phi further (source (P)).

For each architecture and seed the model is (optionally) trained, restored at every
requested checkpoint, and measured on a fixed-size validation minibatch; results are
aggregated over seeds into per-(arch, checkpoint) mean/std of Phi, the diagonal norms,
the coupling field, and the validation accuracy. The bundle is JSON-serializable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.config import Exp8Config
from experiments.data import (
    get_cifar10_image_loaders,
    get_cifar100_image_loaders,
    get_stanford_cars_loaders,
)
from experiments.models import (
    SegmentedPlainResNet18,
    SegmentedResNet18,
    SegmentedResNet50,
)
from experiments.training import Trainer
from experiments.utils import set_seed
from hessian import ParamBlockEstimator, ParamGroupedModel

logger = logging.getLogger(__name__)

# Fixed seed for the train/val split, so every architecture and seed sees the same data.
_SPLIT_SEED = 0


# ======================================================================
# Dataset / backbone construction
# ======================================================================


def _build_loaders(cfg: Exp8Config) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) for the configured dataset (fixed split).

    Phi is read off the validation loader; ``a1`` also trains on the train loader, while
    ``a2`` (epochs = 0) leaves it untouched and only samples a measurement minibatch.
    """
    batch_size = cfg.training.batch_size
    if cfg.dataset == "cifar100":
        return get_cifar100_image_loaders(batch_size)
    if cfg.dataset == "cifar10":
        return get_cifar10_image_loaders(batch_size)
    if cfg.dataset == "stanford_cars":
        return get_stanford_cars_loaders(
            batch_size,
            image_size=cfg.image_size,
            augment=False,  # diagnostic only: measure on un-augmented inputs
            seed=_SPLIT_SEED,
        )
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def _build_model(cfg: Exp8Config, arch: str) -> SegmentedResNet18:
    """Backbone for the registry key ``arch`` (all share the ResNet segmentation)."""
    if arch == "resnet50":
        return SegmentedResNet50(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    if arch == "resnet18":
        return SegmentedResNet18(num_classes=cfg.num_classes)
    if arch == "plain_resnet18":
        return SegmentedPlainResNet18(num_classes=cfg.num_classes)
    raise ValueError(
        f"unknown arch {arch!r}; expected "
        "'resnet18', 'plain_resnet18', or 'resnet50'"
    )


# ======================================================================
# Checkpoint helpers
# ======================================================================


def _resolve_checkpoint_epochs(total_epochs: int, labels: list[str]) -> list[int]:
    """Maps the named checkpoints to epoch numbers (init = 0, mid, final = total)."""
    mapping = {"init": 0, "mid": total_epochs // 2, "final": total_epochs}
    return [mapping.get(label, total_epochs) for label in labels]


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"


# ======================================================================
# Phi measurement
# ======================================================================


def _measure_phi(
    model: ParamGroupedModel,
    x: Tensor,
    y: Tensor,
    loss_fn: nn.Module,
    n_probes: int,
) -> dict[str, Any]:
    """JSON-safe Phi report on one minibatch: Phi, diagonal norms, coupling field.

    Coupling keys are flattened to ``"v->w"`` strings so the bundle serializes directly.
    """
    estimator = ParamBlockEstimator(model, loss_fn, n_probes=n_probes)
    report = estimator.estimate_phi_report(x, y)
    return {
        "phi": report.phi,
        "diag_frob": dict(report.diag_frob),
        "coupling": {f"{v}->{w}": c for (v, w), c in report.coupling.items()},
    }


# ======================================================================
# Aggregation
# ======================================================================


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _aggregate(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregates over seeds: mean/std for each (arch, checkpoint).

    Per (arch, checkpoint): scalar Phi, per-block diagonal norm, per-pair coupling, and
    validation accuracy - each summarized over the seeds that measured it.
    """
    aggregated: dict[str, Any] = {}
    for arch, ckpt_map in raw.items():
        aggregated[arch] = {}
        for ckpt, seed_data in ckpt_map.items():
            payloads = list(seed_data.values())
            ref = payloads[0]
            aggregated[arch][ckpt] = {
                "phi_mean": float(np.mean([p["phi"] for p in payloads])),
                "phi_std": float(np.std([p["phi"] for p in payloads])),
                "diag_frob": {
                    block: _mean_std([p["diag_frob"][block] for p in payloads])
                    for block in ref["diag_frob"]
                },
                "coupling": {
                    pair: _mean_std([p["coupling"][pair] for p in payloads])
                    for pair in ref["coupling"]
                },
                "test_acc_mean": float(np.mean([p["test_acc"] for p in payloads])),
                "test_acc_std": float(np.std([p["test_acc"] for p in payloads])),
                "n_seeds": len(payloads),
            }
    return aggregated


# ======================================================================
# Runner
# ======================================================================


class Exp8Runner:
    """Runner for Exp8 (Stage-A Phi blind-spot map: the inversion Phi_ResNet >> Phi_Plain)."""

    def __init__(self, config: Exp8Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {arch: {checkpoint: {phi_mean/std, diag_frob, coupling, test_acc...}}}."""
        train_loader, val_loader = _build_loaders(self._cfg)
        raw: dict[str, dict[str, Any]] = {arch: {} for arch in self._cfg.archs}

        for arch in self._cfg.archs:
            logger.info("=== Arch: %s ===", arch)

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)
                set_seed(seed)

                model = _build_model(self._cfg, arch)
                trainer = Trainer(model, self._cfg.training, self._device)

                epochs = self._cfg.training.epochs
                ckpt_epochs = _resolve_checkpoint_epochs(
                    epochs, self._cfg.checkpoint_epochs
                )
                checkpoints = trainer.train(train_loader, checkpoint_epochs=ckpt_epochs)

                for ep, state in checkpoints.items():
                    trainer.restore(state)
                    model.to(self._device)
                    model.eval()

                    x_batch, y_batch = self._sample_hessian_batch(val_loader)
                    report = _measure_phi(
                        model, x_batch, y_batch, trainer.loss_fn, self._cfg.n_probes
                    )
                    report["test_acc"] = trainer.evaluate(val_loader)["test_acc"]

                    ckpt_name = _epoch_to_label(ep, epochs)
                    raw[arch].setdefault(ckpt_name, {})[f"seed_{seed}"] = report

        return _aggregate(raw)

    # ------------------------------------------------------------------
    # Measurement batch
    # ------------------------------------------------------------------

    def _sample_hessian_batch(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        bs = self._cfg.hessian_batch_size
        xs, ys = [], []
        n = 0
        for x, y in loader:
            xs.append(x)
            ys.append(y)
            n += x.size(0)
            if n >= bs:
                break
        x_cat = torch.cat(xs)[:bs].to(self._device)
        y_cat = torch.cat(ys)[:bs].to(self._device)
        return x_cat, y_cat
