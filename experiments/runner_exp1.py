"""Experiment 1: Plain vs ResNet - decay of R and C with distance.

Claim (Theorem S8): in a plain network R(v,w) decays exponentially
with dist(v,w), while in a ResNet it is preserved via skip-connections.

Protocol:
  1. For each depth in depths (depth sweep):
     a. For each seed:
        i.   Create PlainMLP and ResidualMLP (depth, width).
        ii.  Train on CIFAR-10. Save checkpoints (init/mid/final).
        iii. At each checkpoint:
             - Compute R(v,w) and C(v,w) for all pairs.
             - Compute spectral norms of Jacobians ||D_{i+1<-i}||_2 (rho).
        iv.  Aggregate: R_bar(d), C_bar(d), rho_max, rho_mean.
  2. Average over seeds, plot with confidence bands.

Result:
  {depth: {arch: {checkpoint: {distance: {R_mean, R_std, C_mean, C_std},
                                "rho": {rho_max_mean, rho_max_std, rho_mean_mean, rho_mean_std}
                               }}}}
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.config import Exp1Config
from experiments.data import get_cifar10_loaders
from experiments.models import PlainMLP, ResidualMLP, _ResidualMLPBlock
from experiments.training import Trainer
from experiments.utils import set_seed
from hessian import ExactBlockHessian, StochasticHessianEstimator
from hessian.metrics import compute_coupling, compute_resonance, coupling_from_norms

logger = logging.getLogger(__name__)


class Exp1Runner:
    """Runner for Experiment 1 (with depth sweep)."""

    def __init__(self, config: Exp1Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {depth_str: {arch: {ckpt: {dist: metrics, "rho": rho_stats}}}}."""
        all_results: dict[str, Any] = {}
        train_loader, val_loader = get_cifar10_loaders(self._cfg.training.batch_size)

        for depth in self._cfg.depths:
            logger.info("=== Depth %d ===", depth)
            raw: dict[str, Any] = {"plain": {}, "residual": {}}

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)

                for arch_name, model_cls in [
                    ("plain", PlainMLP),
                    ("residual", ResidualMLP),
                ]:
                    logger.info("    Architecture: %s", arch_name)
                    set_seed(seed)
                    kwargs: dict[str, Any] = {
                        "width": self._cfg.width,
                        "depth": depth,
                    }
                    if model_cls is PlainMLP:
                        kwargs["use_layernorm"] = self._cfg.use_layernorm
                        kwargs["use_spectral_norm"] = self._cfg.use_spectral_norm
                    model = model_cls(**kwargs)
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

                        x_batch, y_batch = self._sample_hessian_batch(val_loader)

                        metrics = self._compute_metrics(
                            model,
                            x_batch,
                            y_batch,
                            trainer.loss_fn,
                        )
                        rho = self._compute_jacobian_spectral_norms(model, x_batch)
                        lyap = self._compute_lyapunov_exponent(model, x_batch)
                        eval_result = trainer.evaluate(val_loader)

                        ckpt_name = _epoch_to_label(ep, epochs)
                        key = f"seed_{seed}"
                        raw[arch_name].setdefault(ckpt_name, {})[key] = {
                            "distances": metrics,
                            "rho": rho,
                            "lyapunov": lyap,
                            "test_acc": eval_result["test_acc"],
                            "test_loss": eval_result["test_loss"],
                        }

            all_results[str(depth)] = self._aggregate(raw)

        return all_results

    def _resolve_checkpoint_epochs(self, total_epochs: int) -> list[int]:
        """Converts string checkpoint labels to epoch numbers."""
        mapping = {
            "init": 0,
            "mid": total_epochs // 2,
            "final": total_epochs,
        }
        return [
            mapping.get(label, total_epochs) for label in self._cfg.checkpoint_epochs
        ]

    # ------------------------------------------------------------------
    # Jacobian spectral norms (rho)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_jacobian_spectral_norms(
        self,
        model: nn.Module,
        x: Tensor,
    ) -> dict[str, Any]:
        """Computes spectral norms of Jacobians ||J_block||_2 between segments.

        For _MLPBlock (Plain): full Jacobian via autograd, correctly accounting
        for LayerNorm (forward = act(LN(linear(h)))).
        For _ResidualMLPBlock: J = alpha * diag(sigma'(z)) * W + I (no LayerNorm,
        the analytic formula is exact).

        Returns {"rho_max": max_i ||J_i||_2, "rho_mean": mean_i ||J_i||_2,
                 "rho_list": [||J_0||, ...]}.
        """
        norms: list[float] = []

        # Run activations layer by layer through model.blocks
        # (segments[0] = projection + block_0 is a closure, so we work directly)
        if not hasattr(model, "blocks") or not hasattr(model, "projection"):
            return {"rho_max": 0.0, "rho_mean": 0.0, "rho_list": []}

        h = model.projection(x.flatten(1))  # type: ignore[operator]  # (B, width)
        for block in model.blocks:  # type: ignore[union-attr]
            if isinstance(block, _ResidualMLPBlock):
                # _ResidualMLPBlock: no LayerNorm, analytic formula is exact.
                # J = alpha * diag(sigma'(z)) * W + I
                W = block.linear.weight  # (out, in)
                z = block.linear(h)  # pre-activation (B, out)
                z_req = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    a = block.act(z_req)
                    D = torch.autograd.grad(a.sum(), z_req)[0].detach()  # (B, out)
                J = block.alpha * (D.unsqueeze(-1) * W.unsqueeze(0)) + torch.eye(
                    W.shape[0],
                    device=W.device,
                ).unsqueeze(0)
            else:
                # _MLPBlock: forward = act(LN(linear(h))); full autograd Jacobian
                # correctly accounts for LayerNorm.
                B_sz, d_in = h.shape
                d_out = block.linear.weight.shape[0]
                h_req = h.detach().requires_grad_(True)
                with torch.enable_grad():
                    out = block(h_req)
                    J = torch.zeros(B_sz, d_out, d_in, device=h.device)
                    for j in range(d_out):
                        g = torch.autograd.grad(
                            out[:, j].sum(),
                            h_req,
                            retain_graph=(j < d_out - 1),
                        )[0]
                        J[:, j, :] = g

            # sigma_max per sample, averaged over the batch
            sv = torch.linalg.svdvals(J)  # (B, min(out,in))
            s_max = sv[:, 0].mean().item()
            norms.append(s_max)

            # Forward through the full block for the next iteration
            h = block(h)

        if not norms:
            return {"rho_max": 0.0, "rho_mean": 0.0, "rho_list": []}

        return {
            "rho_max": max(norms),
            "rho_mean": sum(norms) / len(norms),
            "rho_list": norms,
        }

    # ------------------------------------------------------------------
    # Lyapunov exponent
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_lyapunov_exponent(
        self,
        model: nn.Module,
        x: Tensor,
    ) -> dict[str, Any]:
        """Computes the maximum Lyapunov exponent of the Jacobian chain.

        Definition:
            lambda_1 = (1/L) * log ||J_L ... J_1||_2

        This is the geometric-mean growth rate of the Jacobian product norm,
        controlling the actual (not worst-case) rate of signal decay/amplification
        through the network.

        For numerical stability, computed via QR re-factorization at each step
        (standard Benettin algorithm):
            P_k = J_k * Q_{k-1}, then QR: P_k = Q_k * R_k
            lambda_1 = (1/L) * sum_k log |R_k[0,0]|

        Returns {"lyapunov_max": lambda_1, "lyapunov_all": [lambda_1, lambda_2, ...]}.
        """
        if not hasattr(model, "blocks") or not hasattr(model, "projection"):
            return {"lyapunov_max": 0.0, "lyapunov_all": []}

        blocks = list(model.blocks)  # type: ignore[union-attr, arg-type]
        L = len(blocks)
        if L == 0:
            return {"lyapunov_max": 0.0, "lyapunov_all": []}

        h = model.projection(x.flatten(1))  # type: ignore[operator]  # (B, width)
        d = h.shape[1]

        # Initialization: Q is the identity matrix (B, d, d)
        B_sz = h.shape[0]
        Q = torch.eye(d, device=h.device).unsqueeze(0).expand(B_sz, -1, -1).clone()
        log_diag_sum = torch.zeros(B_sz, d, device=h.device)

        for block in blocks:
            if isinstance(block, _ResidualMLPBlock):
                W = block.linear.weight
                z = block.linear(h)
                z_req = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    a = block.act(z_req)
                    D = torch.autograd.grad(a.sum(), z_req)[0].detach()
                J = block.alpha * (D.unsqueeze(-1) * W.unsqueeze(0)) + torch.eye(
                    W.shape[0],
                    device=W.device,
                ).unsqueeze(0)
            else:
                B_sz_cur, d_in = h.shape
                d_out = block.linear.weight.shape[0]
                h_req = h.detach().requires_grad_(True)
                with torch.enable_grad():
                    out = block(h_req)
                    J = torch.zeros(B_sz_cur, d_out, d_in, device=h.device)
                    for j in range(d_out):
                        g = torch.autograd.grad(
                            out[:, j].sum(),
                            h_req,
                            retain_graph=(j < d_out - 1),
                        )[0]
                        J[:, j, :] = g

            # Benettin algorithm: P = J @ Q, then QR
            P = torch.bmm(J, Q)  # (B, d, d)
            Q, R = torch.linalg.qr(P)
            # Accumulate log|diag(R)| for each exponent
            log_diag_sum += torch.log(
                R.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-30)
            )

            h = block(h)

        # lambda_k = (1/L) * sum_i log|R_i[k,k]|, averaged over the batch
        lyap_all = (log_diag_sum / L).mean(dim=0)  # (d,)
        lyap_list = lyap_all.sort(descending=True).values.tolist()

        return {
            "lyapunov_max": lyap_list[0],
            "lyapunov_all": lyap_list,
        }

    # ------------------------------------------------------------------
    # Hessian metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
    ) -> dict[int, dict[str, float]]:
        """Computes R and C for all layer pairs, aggregated by distance."""
        layer_names = model.get_layer_names()  # type: ignore[attr-defined, operator]

        if self._cfg.hessian.mode == "exact":
            return self._compute_exact(model, x, y, loss_fn, layer_names)
        return self._compute_stochastic(model, x, y, loss_fn, layer_names)

    def _compute_exact(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        layer_names: list[str],
    ) -> dict[int, dict[str, float]]:
        hess_calc = ExactBlockHessian(model, loss_fn)  # type: ignore[arg-type]
        n = len(layer_names)
        pairs = [
            (layer_names[i], layer_names[j]) for i in range(n) for j in range(i, n)
        ]
        result = hess_calc.compute_all_blocks(x, y, pairs)

        by_distance: dict[int, list[dict[str, float]]] = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                dist = j - i
                v, w = layer_names[i], layer_names[j]
                block_vw = result.get_block(v, w)
                block_vv = result.get_block(v, v)
                block_ww = result.get_block(w, w)
                if block_vw is None or block_vv is None or block_ww is None:
                    continue
                r = compute_resonance(block_vw)
                c = compute_coupling(block_vw, block_vv, block_ww)
                by_distance[dist].append({"R": r, "C": c})

        return _average_by_distance(by_distance)

    def _compute_stochastic(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        layer_names: list[str],
    ) -> dict[int, dict[str, float]]:
        estimator = StochasticHessianEstimator(
            model,  # type: ignore[arg-type]
            loss_fn,
            n_probes=self._cfg.hessian.n_probes,
        )
        n = len(layer_names)

        # Cache diagonal R(v,v) for coupling computation
        diag_r: dict[str, float] = {}
        for name in layer_names:
            diag_r[name] = estimator.estimate_block_frob_norm(x, y, name, name)

        by_distance: dict[int, list[dict[str, float]]] = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                dist = j - i
                v, w = layer_names[i], layer_names[j]
                if i == j:
                    # Diagonal: R from cache, C = 1.0 by definition
                    by_distance[dist].append({"R": diag_r[v], "C": 1.0})
                else:
                    r = estimator.estimate_block_frob_norm(x, y, v, w)
                    c = coupling_from_norms(r, diag_r[v], diag_r[w])
                    by_distance[dist].append({"R": r, "C": c})

        return _average_by_distance(by_distance)

    def _sample_hessian_batch(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        """Takes the first hessian_batch_size examples from the loader."""
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
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(raw: dict[str, Any]) -> dict[str, Any]:
        """Aggregates over seeds: mean +/- std for each (arch, checkpoint, distance + rho)."""
        import numpy as np

        aggregated: dict[str, Any] = {}
        for arch in ("plain", "residual"):
            aggregated[arch] = {}
            for ckpt_name, seed_data in raw[arch].items():
                # seed_data: {seed_key: {"distances": {...}, "rho": {...}}}

                # 1) Aggregate R/C by distance
                all_dists: dict[int, dict[str, list[float]]] = defaultdict(
                    lambda: defaultdict(list)
                )
                for _seed_key, payload in seed_data.items():
                    for dist, metrics in payload["distances"].items():
                        for metric_name, val in metrics.items():
                            all_dists[dist][metric_name].append(val)

                ckpt_agg: dict[str, Any] = {}
                ckpt_agg["distances"] = {}
                for dist, metric_lists in sorted(all_dists.items()):
                    d_agg: dict[str, float] = {}
                    for metric_name, vals in metric_lists.items():
                        arr = np.array(vals)
                        d_agg[f"{metric_name}_mean"] = float(arr.mean())
                        d_agg[f"{metric_name}_std"] = float(arr.std())
                    ckpt_agg["distances"][dist] = d_agg

                # 2) Aggregate rho over seeds
                rho_maxes: list[float] = []
                rho_means: list[float] = []
                for _seed_key, payload in seed_data.items():
                    rho_maxes.append(payload["rho"]["rho_max"])
                    rho_means.append(payload["rho"]["rho_mean"])

                ckpt_agg["rho"] = {
                    "rho_max_mean": float(np.mean(rho_maxes)),
                    "rho_max_std": float(np.std(rho_maxes)),
                    "rho_mean_mean": float(np.mean(rho_means)),
                    "rho_mean_std": float(np.std(rho_means)),
                }

                # 3) Aggregate Lyapunov exponents over seeds
                lyap_maxes: list[float] = []
                for _seed_key, payload in seed_data.items():
                    if "lyapunov" in payload:
                        lyap_maxes.append(payload["lyapunov"]["lyapunov_max"])
                if lyap_maxes:
                    ckpt_agg["lyapunov"] = {
                        "lyapunov_max_mean": float(np.mean(lyap_maxes)),
                        "lyapunov_max_std": float(np.std(lyap_maxes)),
                    }

                # 4) Aggregate test accuracy over seeds
                test_accs: list[float] = []
                test_losses: list[float] = []
                for _seed_key, payload in seed_data.items():
                    if "test_acc" in payload:
                        test_accs.append(payload["test_acc"])
                    if "test_loss" in payload:
                        test_losses.append(payload["test_loss"])
                if test_accs:
                    ckpt_agg["test_acc_mean"] = float(np.mean(test_accs))
                    ckpt_agg["test_acc_std"] = float(np.std(test_accs))
                if test_losses:
                    ckpt_agg["test_loss_mean"] = float(np.mean(test_losses))
                    ckpt_agg["test_loss_std"] = float(np.std(test_losses))

                aggregated[arch][ckpt_name] = ckpt_agg

        return aggregated


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"


def _average_by_distance(
    by_distance: dict[int, list[dict[str, float]]],
) -> dict[int, dict[str, float]]:
    """Averages R and C over pairs with the same distance."""
    import numpy as np

    result: dict[int, dict[str, float]] = {}
    for dist, entries in sorted(by_distance.items()):
        rs = [e["R"] for e in entries if not np.isnan(e["R"])]
        cs = [e["C"] for e in entries if not np.isnan(e["C"])]
        result[dist] = {
            "R": float(np.mean(rs)) if rs else float("nan"),
            "C": float(np.mean(cs)) if cs else float("nan"),
        }
    return result
