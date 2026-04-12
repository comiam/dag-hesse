"""Experiment 6: ResNet-18 on CIFAR-10 - verification of the theory on a conv architecture.

Claim:
  - For piecewise-linear sigma (ReLU): H^T = 0, GN-Gap ~= 0
    -> confirmation on a real conv architecture (variant A).
  - For smooth sigma (SiLU): sigma'' != 0, GN-Gap > 0 (variant B - control).
  - In a plain conv-net (no skip), R(d) decays faster than in ResNet-18.

Protocol:
  1. For each activation (relu, silu):
     a. For each seed:
        i.   Create SegmentedResNet18(act) and SegmentedPlainResNet18(act).
        ii.  Train on CIFAR-10 (augmented). Checkpoints: init/mid/final.
        iii. At each checkpoint:
             - R(d), C(d), D(d) for all pairs via estimate_frob_and_stable_rank.
             - GN-Gap(d) via StochasticGNGapEstimator.
     b. Aggregate over seeds: mean +/- std.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.config import Exp6Config
from experiments.data import get_cifar10_image_loaders
from experiments.models import SegmentedPlainResNet18, SegmentedResNet18
from experiments.training import Trainer
from experiments.utils import set_seed
from hessian import StochasticGNGapEstimator, StochasticHessianEstimator
from hessian.metrics import coupling_from_norms
from hessian.types import GNGapResult

logger = logging.getLogger(__name__)

_ARCH_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "resnet18": SegmentedResNet18,
    "plain_resnet18": SegmentedPlainResNet18,
}


class Exp6Runner:
    """Runner for Experiment 6 (ResNet-18 conv, GN-Gap + R/C/D)."""

    def __init__(self, config: Exp6Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {activation: {arch: {ckpt: {distances: ..., gn_gap: ...}}}}."""
        all_results: dict[str, Any] = {}
        train_loader, val_loader = get_cifar10_image_loaders(
            self._cfg.training.batch_size,
        )

        for act in self._cfg.activations:
            logger.info("=== Activation: %s ===", act)
            raw: dict[str, dict[str, Any]] = {arch: {} for arch in _ARCH_FACTORIES}

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)

                for arch_name, model_cls in _ARCH_FACTORIES.items():
                    logger.info("    Architecture: %s", arch_name)
                    set_seed(seed)

                    model = model_cls(num_classes=10, activation=act)
                    trainer = Trainer(model, self._cfg.training, self._device)

                    epochs = self._cfg.training.epochs
                    ckpt_epochs = self._resolve_checkpoint_epochs(epochs)
                    checkpoints = trainer.train(
                        train_loader,
                        checkpoint_epochs=ckpt_epochs,
                    )

                    for ep, state in checkpoints.items():
                        trainer.restore(state)
                        model.to(self._device)
                        model.eval()

                        x_batch, y_batch = self._sample_hessian_batch(val_loader)

                        metrics = self._compute_hessian_metrics(
                            model,
                            x_batch,
                            y_batch,
                            trainer.loss_fn,
                        )
                        gn_gap = self._compute_gn_gap_metrics(
                            model,
                            x_batch,
                            y_batch,
                            trainer.loss_fn,
                        )
                        eval_result = trainer.evaluate(val_loader)

                        ckpt_name = _epoch_to_label(ep, epochs)
                        key = f"seed_{seed}"
                        raw[arch_name].setdefault(ckpt_name, {})[key] = {
                            "distances": metrics,
                            "gn_gap": gn_gap,
                            "test_acc": eval_result["test_acc"],
                            "test_loss": eval_result["test_loss"],
                        }

            all_results[act] = self._aggregate(raw)

        return all_results

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_epochs(self, total_epochs: int) -> list[int]:
        mapping = {"init": 0, "mid": total_epochs // 2, "final": total_epochs}
        return [
            mapping.get(label, total_epochs) for label in self._cfg.checkpoint_epochs
        ]

    def _sample_hessian_batch(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        bs = self._cfg.hessian.hessian_batch_size
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

    # ------------------------------------------------------------------
    # R / C / D metrics (per distance)
    # ------------------------------------------------------------------

    def _compute_hessian_metrics(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
    ) -> dict[int, dict[str, float]]:
        """R, C, D for all pairs, aggregated by distance."""
        layer_names = model.get_layer_names()  # type: ignore[attr-defined, operator]
        estimator = StochasticHessianEstimator(
            model,  # type: ignore[arg-type]
            loss_fn,
            n_probes=self._cfg.hessian.n_probes,  # type: ignore[arg-type]
        )
        n = len(layer_names)

        # Diagonal R(v,v) for coupling computation
        diag_r: dict[str, float] = {}
        diag_d: dict[str, float] = {}
        for name in layer_names:
            r, d = estimator.estimate_frob_and_stable_rank(x, y, name, name)
            diag_r[name] = r
            diag_d[name] = d

        by_distance: dict[int, list[dict[str, float]]] = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                dist = j - i
                v, w = layer_names[i], layer_names[j]
                if i == j:
                    by_distance[dist].append({"R": diag_r[v], "C": 1.0, "D": diag_d[v]})
                else:
                    r, d = estimator.estimate_frob_and_stable_rank(x, y, v, w)
                    c = coupling_from_norms(r, diag_r[v], diag_r[w])
                    by_distance[dist].append({"R": r, "C": c, "D": d})

        return _average_by_distance(by_distance)

    # ------------------------------------------------------------------
    # GN-Gap metrics (per distance)
    # ------------------------------------------------------------------

    def _compute_gn_gap_metrics(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, Any]:
        """GN-Gap: global + per-distance."""
        layer_names = model.get_layer_names()  # type: ignore[attr-defined, operator]
        estimator = StochasticGNGapEstimator(
            model,  # type: ignore[arg-type]
            loss_fn,
            n_probes=self._cfg.hessian.n_probes,  # type: ignore[arg-type]
        )
        n = len(layer_names)

        by_distance: dict[int, list[GNGapResult]] = defaultdict(list)
        all_results: list[GNGapResult] = []

        for i in range(n):
            for j in range(i, n):
                v, w = layer_names[i], layer_names[j]
                result = estimator.estimate_gn_gap(x, y, v, w)
                by_distance[j - i].append(result)
                all_results.append(result)

        # Global gap
        total_tensor = sum(r.frob_tensor for r in all_results)
        total_gn = sum(r.frob_gn for r in all_results)
        gap_global = total_tensor / (total_gn + 1e-12)

        # Per-distance
        distances: dict[int, dict[str, float]] = {}
        for d, results in sorted(by_distance.items()):
            d_tensor = sum(r.frob_tensor for r in results)
            d_gn = sum(r.frob_gn for r in results)
            distances[d] = {
                "gap": d_tensor / (d_gn + 1e-12),
                "frob_tensor": d_tensor,
                "frob_gn": d_gn,
            }

        return {"gap_global": gap_global, "distances": distances}

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Aggregates over seeds: mean +/- std for each (arch, ckpt)."""
        aggregated: dict[str, Any] = {}

        for arch_name, ckpt_map in raw.items():
            aggregated[arch_name] = {}
            for ckpt_name, seed_data in ckpt_map.items():
                # --- R/C/D per distance ---
                all_dists: dict[int, dict[str, list[float]]] = defaultdict(
                    lambda: defaultdict(list),
                )
                for payload in seed_data.values():
                    for dist, metrics in payload["distances"].items():
                        for metric_name, val in metrics.items():
                            all_dists[dist][metric_name].append(val)

                dist_agg: dict[int, dict[str, float]] = {}
                for dist, metric_lists in sorted(all_dists.items()):
                    d_agg: dict[str, float] = {}
                    for metric_name, vals in metric_lists.items():
                        arr = np.array(vals)
                        d_agg[f"{metric_name}_mean"] = float(arr.mean())
                        d_agg[f"{metric_name}_std"] = float(arr.std())
                    dist_agg[dist] = d_agg

                # --- GN-Gap ---
                gap_globals = [v["gn_gap"]["gap_global"] for v in seed_data.values()]
                gn_dists: dict[int, dict[str, list[float]]] = defaultdict(
                    lambda: defaultdict(list),
                )
                for payload in seed_data.values():
                    for d, metrics in payload["gn_gap"]["distances"].items():
                        for metric_name, val in metrics.items():
                            gn_dists[d][metric_name].append(val)

                gn_dist_agg: dict[int, dict[str, float]] = {}
                for d, metric_lists in sorted(gn_dists.items()):
                    d_agg_gn: dict[str, float] = {}
                    for metric_name, vals in metric_lists.items():
                        arr = np.array(vals)
                        d_agg_gn[f"{metric_name}_mean"] = float(arr.mean())
                        d_agg_gn[f"{metric_name}_std"] = float(arr.std())
                    gn_dist_agg[d] = d_agg_gn

                # --- Test accuracy ---
                test_accs = [v["test_acc"] for v in seed_data.values()]
                test_losses = [v["test_loss"] for v in seed_data.values()]

                aggregated[arch_name][ckpt_name] = {
                    "distances": dist_agg,
                    "gn_gap_global_mean": float(np.mean(gap_globals)),
                    "gn_gap_global_std": float(np.std(gap_globals)),
                    "gn_gap_distances": gn_dist_agg,
                    "test_acc_mean": float(np.mean(test_accs)),
                    "test_acc_std": float(np.std(test_accs)),
                    "test_loss_mean": float(np.mean(test_losses)),
                    "test_loss_std": float(np.std(test_losses)),
                }

        return aggregated


# ======================================================================
# Helpers
# ======================================================================


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"


def _average_by_distance(
    by_distance: dict[int, list[dict[str, float]]],
) -> dict[int, dict[str, float]]:
    """Averages R, C, D over pairs with the same distance."""
    result: dict[int, dict[str, float]] = {}
    for dist, entries in sorted(by_distance.items()):
        agg: dict[str, float] = {}
        for metric in ("R", "C", "D"):
            vals = [e[metric] for e in entries if not np.isnan(e[metric])]
            agg[metric] = float(np.mean(vals)) if vals else float("nan")
        result[dist] = agg
    return result
