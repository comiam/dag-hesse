"""Experiment 2: Bottleneck ablation - sensitivity of R/C to a narrow layer.

Claim (thm:rank-propagation + cor:bottleneck): narrowing an intermediate layer
constrains the rank of the inter-layer Hessian and weakens the long-range
coupling C(v,w) between layers on opposite sides of the bottleneck.

Protocol:
  1. For each depth in depths:
     a. For each bottleneck_width in bottleneck_widths:
        i.  For each seed:
            - Create BottleneckMLP(base_width, bottleneck_width, depth).
            - Train on CIFAR-10/100 (set in config). Save checkpoints (init/mid/final).
            - At each checkpoint compute C_far and R_far -
              mean C(v,w) and R(v,w) over pairs that "cross" the bottleneck
              (idx_v < bottleneck_pos < idx_w).
     b. Aggregate over seeds: mean +/- std.

Result:
  {depth: {bottleneck_width: {checkpoint: {C_far_mean, C_far_std,
                                            R_far_mean, R_far_std}}}}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.config import Exp2Config
from experiments.data import get_cifar10_loaders, get_cifar100_loaders
from experiments.models import BottleneckMLP
from experiments.training import Trainer
from experiments.utils import set_seed
from hessian import ExactBlockHessian, StochasticHessianEstimator
from hessian.metrics import (
    compute_coupling,
    compute_dimension,
    compute_resonance,
    coupling_from_norms,
)

logger = logging.getLogger(__name__)


class Exp2Runner:
    """Runner for Experiment 2 (bottleneck width sweep)."""

    def __init__(self, config: Exp2Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {depth: {bneck_width: {ckpt: {metric_mean, metric_std, ...}}}}."""
        all_results: dict[str, Any] = {}
        if self._cfg.dataset == "cifar100":
            train_loader, val_loader = get_cifar100_loaders(
                self._cfg.training.batch_size
            )
        else:
            train_loader, val_loader = get_cifar10_loaders(
                self._cfg.training.batch_size
            )

        for depth in self._cfg.depths:
            if depth < 3:
                logger.warning(
                    "Depth %d < 3: cross-bottleneck pairs empty (need i < depth//2 < j). Skipping.",
                    depth,
                )
                continue
            logger.info("=== Depth %d ===", depth)
            raw: dict[int, Any] = {}

            for bneck_w in self._cfg.bottleneck_widths:
                logger.info("  Bottleneck width %d", bneck_w)
                raw[bneck_w] = {}

                for seed in self._cfg.training.seeds:
                    logger.info("    Seed %d", seed)
                    set_seed(seed)

                    model = BottleneckMLP(
                        base_width=self._cfg.base_width,
                        bottleneck_width=bneck_w,
                        depth=depth,
                        num_classes=self._cfg.num_classes,
                    )
                    trainer = Trainer(model, self._cfg.training, self._device)

                    epochs = self._cfg.training.epochs
                    ckpt_epochs = self._resolve_checkpoint_epochs(epochs)
                    checkpoints = trainer.train(
                        train_loader, checkpoint_epochs=ckpt_epochs
                    )

                    for ep, state in checkpoints.items():
                        trainer.restore(state)
                        model.to(self._device)
                        model.eval()

                        # Evaluate test loss / accuracy
                        eval_metrics = self._evaluate(
                            model, val_loader, trainer.loss_fn
                        )

                        x_batch, y_batch = self._sample_hessian_batch(val_loader)

                        metrics = self._compute_far_metrics(
                            model,
                            x_batch,
                            y_batch,
                            trainer.loss_fn,
                        )
                        metrics.update(eval_metrics)

                        ckpt_name = _epoch_to_label(ep, epochs)
                        key = f"seed_{seed}"
                        raw[bneck_w].setdefault(ckpt_name, {})[key] = metrics

            all_results[str(depth)] = self._aggregate(raw)

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
    # Evaluation (loss / accuracy)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        """Computes test loss and accuracy on the full val loader."""
        model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y in loader:
            x, y = x.to(self._device), y.to(self._device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        return {
            "test_loss": total_loss / max(n, 1),
            "test_acc": correct / max(n, 1),
        }

    # ------------------------------------------------------------------
    # Far-coupling metrics
    # ------------------------------------------------------------------

    def _compute_far_metrics(
        self,
        model: BottleneckMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        """C_far, R_far, and D_far for pairs (v,w) that cross the bottleneck.

        Only pairs with idx_v < bottleneck_pos < idx_w.
        """
        layer_names = model.get_layer_names()
        bneck_pos = model.bottleneck_position
        cross_pairs = [
            (i, j)
            for i in range(len(layer_names))
            for j in range(i + 1, len(layer_names))
            if i < bneck_pos < j
        ]

        # All pairs for the full C_ij matrix
        all_pairs = [
            (i, j)
            for i in range(len(layer_names))
            for j in range(i + 1, len(layer_names))
        ]

        if not cross_pairs:
            return {
                "C_far": 0.0,
                "R_far": 0.0,
                "D_far": 0.0,
                "n_pairs": 0,
                "C_matrix": [],  # type: ignore[dict-item]
                "D_matrix": [],  # type: ignore[dict-item]
            }

        if self._cfg.hessian.mode == "exact":
            return self._far_exact(
                model, x, y, loss_fn, layer_names, cross_pairs, all_pairs
            )
        return self._far_stochastic(
            model, x, y, loss_fn, layer_names, cross_pairs, all_pairs
        )

    def _far_exact(
        self,
        model: BottleneckMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        layer_names: list[str],
        cross_pairs: list[tuple[int, int]],
        all_pairs: list[tuple[int, int]],
    ) -> dict[str, Any]:
        hess_calc = ExactBlockHessian(model, loss_fn)  # type: ignore[arg-type]

        # Determine all required blocks: all pairs + diagonal
        diag_indices = set(range(len(layer_names)))
        pairs_to_compute = [(layer_names[i], layer_names[j]) for i, j in all_pairs] + [
            (layer_names[k], layer_names[k]) for k in diag_indices
        ]

        result = hess_calc.compute_all_blocks(x, y, pairs_to_compute)

        # Full C_ij and D_ij matrices
        n = len(layer_names)
        c_matrix = [[0.0] * n for _ in range(n)]
        d_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            c_matrix[i][i] = 1.0
            block_ii = result.get_block(layer_names[i], layer_names[i])
            d_matrix[i][i] = compute_dimension(block_ii) if block_ii else 0.0
        for i, j in all_pairs:
            v, w = layer_names[i], layer_names[j]
            block_vw = result.get_block(v, w)
            block_vv = result.get_block(v, v)
            block_ww = result.get_block(w, w)
            if block_vw is None or block_vv is None or block_ww is None:
                continue
            c_val = compute_coupling(block_vw, block_vv, block_ww)
            c_matrix[i][j] = c_val
            c_matrix[j][i] = c_val
            d_val = compute_dimension(block_vw)
            d_matrix[i][j] = d_val
            d_matrix[j][i] = d_val

        # Cross-bottleneck metrics
        rs, cs, ds = [], [], []
        for i, j in cross_pairs:
            v, w = layer_names[i], layer_names[j]
            block_vw = result.get_block(v, w)
            if block_vw is None:
                continue
            rs.append(compute_resonance(block_vw))
            cs.append(c_matrix[i][j])
            ds.append(d_matrix[i][j])

        return {
            "C_far": float(np.mean(cs)) if cs else 0.0,
            "R_far": float(np.mean(rs)) if rs else 0.0,
            "D_far": float(np.nanmean(ds)) if ds else 0.0,
            "n_pairs": len(cs),
            "C_matrix": c_matrix,
            "D_matrix": d_matrix,
        }

    def _far_stochastic(
        self,
        model: BottleneckMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        layer_names: list[str],
        cross_pairs: list[tuple[int, int]],
        all_pairs: list[tuple[int, int]],
    ) -> dict[str, Any]:
        estimator = StochasticHessianEstimator(
            model,
            loss_fn,
            n_probes=self._cfg.hessian.n_probes,
            n_power_iter=self._cfg.hessian.n_power_iter,
        )

        # Cache all diagonal norms and stable rank
        n = len(layer_names)
        diag_r: dict[int, float] = {}
        diag_d: dict[int, float] = {}
        for k in range(n):
            name = layer_names[k]
            r_kk, d_kk = estimator.estimate_frob_and_stable_rank(x, y, name, name)
            diag_r[k] = r_kk
            diag_d[k] = d_kk

        # Full C_ij, D_ij matrices + raw norm cache
        c_matrix = [[0.0] * n for _ in range(n)]
        d_matrix = [[0.0] * n for _ in range(n)]
        raw_norms: dict[tuple[int, int], float] = {}
        for i in range(n):
            c_matrix[i][i] = 1.0
            d_matrix[i][i] = diag_d[i]
        for i, j in all_pairs:
            v, w = layer_names[i], layer_names[j]
            r, d = estimator.estimate_frob_and_stable_rank(x, y, v, w)
            raw_norms[(i, j)] = r
            c = coupling_from_norms(r, diag_r[i], diag_r[j])
            c_matrix[i][j] = c
            c_matrix[j][i] = c
            d_matrix[i][j] = d
            d_matrix[j][i] = d

        # Cross-bottleneck metrics (using cached values)
        rs, cs, ds = [], [], []
        for i, j in cross_pairs:
            cs.append(c_matrix[i][j])
            rs.append(raw_norms[(i, j)])
            ds.append(d_matrix[i][j])

        return {
            "C_far": float(np.mean(cs)) if cs else 0.0,
            "R_far": float(np.mean(rs)) if rs else 0.0,
            "D_far": float(np.nanmean(ds)) if ds else 0.0,
            "n_pairs": len(cs),
            "C_matrix": c_matrix,
            "D_matrix": d_matrix,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(raw: dict[int, Any]) -> dict[str, Any]:
        """Aggregates over seeds: mean +/- std for each (bneck_width, checkpoint)."""
        aggregated: dict[str, Any] = {}
        for bneck_w, ckpt_dict in raw.items():
            bneck_agg: dict[str, Any] = {}
            for ckpt_name, seed_data in ckpt_dict.items():
                c_vals = [v["C_far"] for v in seed_data.values()]
                r_vals = [v["R_far"] for v in seed_data.values()]
                d_vals = [v["D_far"] for v in seed_data.values()]
                loss_vals = [v["test_loss"] for v in seed_data.values()]
                acc_vals = [v["test_acc"] for v in seed_data.values()]

                entry: dict[str, Any] = {
                    "C_far_mean": float(np.mean(c_vals)),
                    "C_far_std": float(np.std(c_vals)),
                    "R_far_mean": float(np.mean(r_vals)),
                    "R_far_std": float(np.std(r_vals)),
                    "D_far_mean": float(np.nanmean(d_vals)),
                    "D_far_std": float(np.nanstd(d_vals)),
                    "test_loss_mean": float(np.mean(loss_vals)),
                    "test_loss_std": float(np.std(loss_vals)),
                    "test_acc_mean": float(np.mean(acc_vals)),
                    "test_acc_std": float(np.std(acc_vals)),
                }

                # Mean C_ij and D_ij matrices across seeds
                for mkey in ("C_matrix", "D_matrix"):
                    matrices = [v[mkey] for v in seed_data.values() if v.get(mkey)]
                    if matrices:
                        entry[mkey] = np.nanmean(matrices, axis=0).tolist()

                bneck_agg[ckpt_name] = entry
            aggregated[str(bneck_w)] = bneck_agg
        return aggregated


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"
