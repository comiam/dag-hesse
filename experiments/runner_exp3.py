"""Experiment 3: selective relevance of GN-Gap by activation type.

Claim: for piecewise-linear activations sigma''(z)=0 a.e., so H^T=0 and GN-Gap~=0.
For smooth activations sigma''!=0 and GN-Gap is noticeable.
This is a direct consequence of the canonical decomposition H = H^GN + H^T.

Additionally: correlation of Gap vs E[sigma''(z)^2] across activation types.

Protocol:
  1. For each activation in activations:
     a. For each seed:
        i.   Create PlainMLP(depth, width, activation).
        ii.  Train on CIFAR-10. Save checkpoints (init/mid/final).
        iii. At each checkpoint:
             - GN-Gap for all pairs (v,w): global gap + per-distance gap(d).
             - E[sigma''(z)^2] over pre-activation values.
     b. Aggregate over seeds: mean +/- std.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.config import Exp3Config
from experiments.data import get_cifar10_loaders
from experiments.models import _ACTIVATION_MAP, PlainMLP
from experiments.training import Trainer
from experiments.utils import set_seed
from hessian import ExactGNDecomposition, StochasticGNGapEstimator
from hessian.types import GNGapResult

logger = logging.getLogger(__name__)


class Exp3Runner:
    """Runner for Experiment 3 (ReLU vs GELU GN-Gap)."""

    def __init__(self, config: Exp3Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {activation: {ckpt: {gap_global_mean/std, distances: ...}}}."""
        all_results: dict[str, Any] = {}
        train_loader, val_loader = get_cifar10_loaders(self._cfg.training.batch_size)

        for act in self._cfg.activations:
            logger.info("=== Activation: %s ===", act)
            raw: dict[str, Any] = {}

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)
                set_seed(seed)

                model = PlainMLP(
                    width=self._cfg.width,
                    depth=self._cfg.depth,
                    activation=act,
                    use_layernorm=self._cfg.use_layernorm,
                )
                trainer = Trainer(model, self._cfg.training, self._device)

                epochs = self._cfg.training.epochs
                ckpt_epochs = self._resolve_checkpoint_epochs(epochs)
                checkpoints = trainer.train(train_loader, checkpoint_epochs=ckpt_epochs)

                for ep, state in checkpoints.items():
                    trainer.restore(state)
                    model.to(self._device)
                    model.eval()

                    x_batch, y_batch = self._sample_hessian_batch(val_loader)
                    metrics = self._compute_gn_gap_metrics(
                        model,
                        x_batch,
                        y_batch,
                        trainer.loss_fn,
                    )
                    metrics["sigma_pp_sq"] = self._measure_sigma_pp_sq(
                        model,
                        x_batch,
                        act,
                    )
                    eval_result = trainer.evaluate(val_loader)
                    metrics["test_acc"] = eval_result["test_acc"]
                    metrics["test_loss"] = eval_result["test_loss"]

                    ckpt_name = _epoch_to_label(ep, epochs)
                    key = f"seed_{seed}"
                    raw.setdefault(ckpt_name, {})[key] = metrics

            all_results[act] = self._aggregate(raw)

        return all_results

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_epochs(self, total_epochs: int) -> list[int]:
        mapping = {
            "init": 0,
            "mid": total_epochs // 2,
            "final": total_epochs,
        }
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
    # GN-Gap metrics
    # ------------------------------------------------------------------

    def _compute_gn_gap_metrics(
        self,
        model: PlainMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, Any]:
        """GN-Gap for all pairs (v, w): global + per-distance."""
        layer_names = model.get_layer_names()

        if self._cfg.hessian.mode == "exact":
            return self._compute_exact(model, x, y, loss_fn, layer_names)
        return self._compute_stochastic(model, x, y, loss_fn, layer_names)

    def _compute_exact(
        self,
        model: PlainMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        layer_names: list[str],
    ) -> dict[str, Any]:
        decomp = ExactGNDecomposition(model, loss_fn)  # type: ignore[arg-type]
        n = len(layer_names)

        by_distance: dict[int, list[GNGapResult]] = defaultdict(list)
        all_results: list[GNGapResult] = []

        for i in range(n):
            for j in range(i, n):
                v, w = layer_names[i], layer_names[j]
                result = decomp.compute_gn_gap(x, y, v, w)
                by_distance[j - i].append(result)
                all_results.append(result)

        return self._format_metrics(by_distance, all_results)

    def _compute_stochastic(
        self,
        model: PlainMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        layer_names: list[str],
    ) -> dict[str, Any]:
        estimator = StochasticGNGapEstimator(
            model,
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

        return self._format_metrics(by_distance, all_results)

    @staticmethod
    def _format_metrics(
        by_distance: dict[int, list[GNGapResult]],
        all_results: list[GNGapResult],
    ) -> dict[str, Any]:
        """Formats global gap + per-distance gap from a list of GNGapResult."""
        # Global: sum(||H^T||_F) / (sum(||H^GN||_F) + eps)
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
    # sigma'' measurement
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_sigma_pp_sq(
        model: PlainMLP,
        x: Tensor,
        activation: str,
    ) -> float:
        """E[sigma''(z)^2] over pre-activation values of all blocks."""
        act_module = _ACTIVATION_MAP[activation]()

        # Collect pre-activation values (after LN, before activation)
        pre_acts: list[Tensor] = []
        with torch.no_grad():
            h = model.projection.ln(model.projection.linear(x.flatten(1)))
            pre_acts.append(h)
            h = model.projection.act(h)
            for block in model.blocks:
                h = block.ln(block.linear(h))  # type: ignore[operator]
                pre_acts.append(h)
                h = block.act(h)  # type: ignore[operator]

        all_z = torch.cat([pa.flatten() for pa in pre_acts])

        # sigma''(z) via double autograd
        z = all_z.detach().requires_grad_(True)
        a = act_module(z)  # type: ignore[operator]
        grad1 = torch.autograd.grad(a.sum(), z, create_graph=True)[0]  # type: ignore[operator]
        grad2 = torch.autograd.grad(grad1.sum(), z)[0]
        return float((grad2**2).mean().item())

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(raw: dict[str, Any]) -> dict[str, Any]:
        """Aggregates over seeds: mean +/- std for each checkpoint."""
        aggregated: dict[str, Any] = {}
        for ckpt_name, seed_data in raw.items():
            # seed_data: {seed_key: {"gap_global": float, "sigma_pp_sq": float, ...}}
            gap_vals = [v["gap_global"] for v in seed_data.values()]
            sigma_vals = [v["sigma_pp_sq"] for v in seed_data.values()]

            test_accs = [v.get("test_acc", float("nan")) for v in seed_data.values()]
            test_losses = [v.get("test_loss", float("nan")) for v in seed_data.values()]
            valid_accs = [a for a in test_accs if not np.isnan(a)]
            valid_losses = [lo for lo in test_losses if not np.isnan(lo)]

            ckpt_agg: dict[str, Any] = {
                "gap_global_mean": float(np.mean(gap_vals)),
                "gap_global_std": float(np.std(gap_vals)),
                "sigma_pp_sq_mean": float(np.mean(sigma_vals)),
                "sigma_pp_sq_std": float(np.std(sigma_vals)),
            }
            if valid_accs:
                ckpt_agg["test_acc_mean"] = float(np.mean(valid_accs))
                ckpt_agg["test_acc_std"] = float(np.std(valid_accs))
            if valid_losses:
                ckpt_agg["test_loss_mean"] = float(np.mean(valid_losses))
                ckpt_agg["test_loss_std"] = float(np.std(valid_losses))

            # Per-distance aggregation
            all_dists: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list),
            )
            for payload in seed_data.values():
                for d, metrics in payload["distances"].items():
                    for metric_name, val in metrics.items():
                        all_dists[d][metric_name].append(val)

            dist_agg: dict[int, Any] = {}
            for d, metric_lists in sorted(all_dists.items()):
                d_agg: dict[str, float] = {}
                for metric_name, vals in metric_lists.items():
                    arr = np.array(vals)
                    d_agg[f"{metric_name}_mean"] = float(arr.mean())
                    d_agg[f"{metric_name}_std"] = float(arr.std())
                dist_agg[d] = d_agg
            ckpt_agg["distances"] = dist_agg

            aggregated[ckpt_name] = ckpt_agg

        return aggregated


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"
