"""Experiment 4: Diamond (multi-branch) MLP - activation of the tensor term.

Claim: in sequential architectures (PlainMLP, ResidualMLP), term (2) of formula (1)
  T_{u;v,w}  (mixed tensor for fan-in >= 2)
is identically zero, since Ch(v) and Ch(w) are disjoint for v != w at different depths.
Diamond MLP with nonlinear merge and a smooth activation activates T_{merge;A,B} != 0.

Hypotheses:
  H4.1: Within each branch R_bar_{AA}(d) decays with distance (analogous to Exp.1).
  H4.2a: At initialization (rho ~= 1) the cross-branch R_bar_{AB}(d_graph) decays
         with graph distance (Theorem S8, s*rho < 1).
  H4.2b: After training (rho > 1) the decay breaks - R grows with distance,
         as predicted by theory when s*rho > 1.
  H4.3: Linear merge (sum) -> T=0; nonlinear merge (cat) + sigma''!=0 (SiLU) -> T!=0.

Protocol:
  Sweep: branch_depth x merge_type x activation x seed x checkpoint.
  Metrics: within-branch R, cross-branch R, rho_max, GN-Gap at merge node, test accuracy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.config import Exp4Config
from experiments.data import get_cifar10_loaders
from experiments.models import DiamondMLP
from experiments.training import Trainer
from experiments.utils import set_seed
from hessian.exact import ExactBlockHessian

logger = logging.getLogger(__name__)


class Exp4Runner:
    """Runner for Experiment 4 (Diamond MLP)."""

    def __init__(self, config: Exp4Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {branch_depth: {config_key: {ckpt: metrics_agg}}}."""
        all_results: dict[str, Any] = {}
        train_loader, val_loader = get_cifar10_loaders(self._cfg.training.batch_size)

        for depth in self._cfg.branch_depths:
            logger.info("=== Branch depth %d ===", depth)
            depth_results: dict[str, Any] = {}

            for merge in self._cfg.merge_types:
                for act in self._cfg.activations:
                    config_key = f"{merge}_{act}"
                    logger.info("  Config: %s", config_key)
                    raw: dict[str, dict[str, Any]] = {}

                    for seed in self._cfg.training.seeds:
                        logger.info("    Seed %d", seed)
                        set_seed(seed)

                        model = DiamondMLP(
                            width=self._cfg.width,
                            branch_depth=depth,
                            activation=act,
                            merge=merge,
                        )
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
                            metrics = self._compute_diamond_metrics(
                                model,
                                x_batch,
                                y_batch,
                                trainer.loss_fn,
                            )
                            metrics["rho"] = self._compute_jacobian_spectral_norms(
                                model,
                                x_batch,
                            )
                            eval_result = trainer.evaluate(val_loader)
                            metrics["test_acc"] = eval_result["test_acc"]
                            metrics["test_loss"] = eval_result["test_loss"]

                            ckpt_name = _epoch_to_label(ep, epochs)
                            raw.setdefault(ckpt_name, {})[f"seed_{seed}"] = metrics

                    depth_results[config_key] = self._aggregate(raw)

            all_results[str(depth)] = depth_results

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
    # Jacobian spectral norms (rho)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_jacobian_spectral_norms(
        self,
        model: DiamondMLP,
        x: Tensor,
    ) -> dict[str, Any]:
        """Spectral norms of Jacobians ||J_block||_2 for Diamond MLP.

        For _MLPBlock (branches), computes the full Jacobian via autograd,
        correctly accounting for LayerNorm. For cat-merge (Sequential(Linear, Act),
        no LayerNorm), uses the analytic formula diag(sigma'(z)) * W.
        For sum-merge the node is linear (J = I/sqrt(2)), sigma_max = 1/sqrt(2).

        Returns:
            {"rho_max", "rho_mean", "rho_list"}
        """
        norms: list[float] = []
        h = model.stem(x.flatten(1))

        h_a, h_b = h, h
        for block in model.branch_a:
            norms.append(self._block_spectral_norm(block, h_a))
            h_a = block(h_a)
        for block in model.branch_b:
            norms.append(self._block_spectral_norm(block, h_b))
            h_b = block(h_b)

        if model._merge_type == "cat":
            # merge_layer = Sequential(Linear, Act)
            h_cat = torch.cat([h_a, h_b], dim=-1)
            linear = model.merge_layer[0]  # type: ignore[index]
            act = model.merge_layer[1]  # type: ignore[index]
            z = linear(h_cat)  # type: ignore[operator]
            z_req = z.detach().requires_grad_(True)
            with torch.enable_grad():
                a = act(z_req)  # type: ignore[operator]
                D = torch.autograd.grad(a.sum(), z_req)[0].detach()  # type: ignore[operator]
            J = D.unsqueeze(-1) * linear.weight.unsqueeze(0)  # type: ignore[operator]
            sv = torch.linalg.svdvals(J)
            norms.append(sv[:, 0].mean().item())
        else:
            norms.append(model._inv_sqrt2)

        if not norms:
            return {"rho_max": 0.0, "rho_mean": 0.0, "rho_list": []}

        return {
            "rho_max": max(norms),
            "rho_mean": sum(norms) / len(norms),
            "rho_list": norms,
        }

    @staticmethod
    def _block_spectral_norm(block: nn.Module, h: Tensor) -> float:
        """sigma_max(J_block) per sample, averaged over the batch for _MLPBlock.

        Computes the full Jacobian J = d block(h)/dh via autograd,
        correctly accounting for LayerNorm (if present in the block).
        """
        B, d_in = h.shape
        d_out = block.linear.weight.shape[0]  # type: ignore[union-attr, index]
        h_req = h.detach().requires_grad_(True)
        with torch.enable_grad():
            out = block(h_req)  # type: ignore[operator]
            J = torch.zeros(B, d_out, d_in, device=h.device)  # type: ignore[arg-type]
            for j in range(d_out):
                g = torch.autograd.grad(
                    out[:, j].sum(),
                    h_req,
                    retain_graph=bool(j < d_out - 1),
                )[0]
                J[:, j, :] = g
        sv = torch.linalg.svdvals(J)
        return sv[:, 0].mean().item()

    # ------------------------------------------------------------------
    # Diamond hessian metrics (autograd, without SegmentedModel)
    # ------------------------------------------------------------------

    def _compute_diamond_metrics(
        self,
        model: DiamondMLP,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, Any]:
        """Computes R for all DAG node pairs and GN-Gap at the merge node.

        The diamond topology is not a chain, so computations go directly
        via autograd (forward_with_intermediates) rather than through
        ExactBlockHessian.
        """
        # Forward with graph for autograd: stem without graph, then with graph
        with torch.no_grad():
            stem_cached = model.stem(x.flatten(1))
        stem_leaf = stem_cached.detach().requires_grad_(True)

        intermediates: dict[str, Tensor] = {}
        h_a, h_b = stem_leaf, stem_leaf
        for i, block in enumerate(model.branch_a):
            h_a = block(h_a)
            intermediates[f"A_{i}"] = h_a
        for i, block in enumerate(model.branch_b):
            h_b = block(h_b)
            intermediates[f"B_{i}"] = h_b

        h_merged = model._merge(h_a, h_b)
        intermediates["merge"] = h_merged
        logits = model.head(h_merged)
        loss = loss_fn(logits, y)

        node_names = model.get_node_names()
        n = len(node_names)

        # Compute H^f blocks for all pairs
        blocks: dict[tuple[str, str], Tensor] = {}
        for i in range(n):
            for j in range(i, n):
                v, w = node_names[i], node_names[j]
                fv, fw = intermediates[v], intermediates[w]
                if i == j:
                    H = ExactBlockHessian._hessian_of_loss_wrt_tensor(loss, fv)
                else:
                    H = ExactBlockHessian._cross_hessian(loss, fv, fw)
                blocks[(v, w)] = H.detach()
                if i != j:
                    blocks[(w, v)] = H.detach().t()

        # Within-branch R(d) for branches A and B
        branch_depth = model._branch_depth
        within_a = self._within_branch_resonance(blocks, "A", branch_depth)
        within_b = self._within_branch_resonance(blocks, "B", branch_depth)

        # Cross-branch R by total graph distance through merge
        cross_ab = self._cross_branch_resonance(blocks, branch_depth)

        # GN-Gap at the merge node: for pairs (A_last, B_last) through merge
        gn_gap = self._compute_merge_gn_gap(
            blocks,
            intermediates,
            logits,
            branch_depth,
        )

        return {
            "within_A": within_a,
            "within_B": within_b,
            "cross_AB": cross_ab,
            "merge_gn_gap": gn_gap,
        }

    @staticmethod
    def _within_branch_resonance(
        blocks: dict[tuple[str, str], Tensor],
        branch: str,
        depth: int,
    ) -> dict[int, dict[str, float]]:
        """R_bar(d) within a single branch, aggregated by distance."""
        by_dist: dict[int, list[float]] = defaultdict(list)
        for i in range(depth):
            for j in range(i, depth):
                v, w = f"{branch}_{i}", f"{branch}_{j}"
                H = blocks.get((v, w))
                if H is None:
                    continue
                r = torch.linalg.norm(H, ord="fro").item()
                by_dist[j - i].append(r)

        result: dict[int, dict[str, float]] = {}
        for d, vals in sorted(by_dist.items()):
            arr = np.array(vals)
            result[d] = {"R_mean": float(arr.mean())}
        return result

    @staticmethod
    def _cross_branch_resonance(
        blocks: dict[tuple[str, str], Tensor],
        depth: int,
    ) -> dict[int, dict[str, float]]:
        """R_bar(d_graph) for cross-branch pairs (A_i, B_j).

        Graph distance through merge: d_graph(A_i, B_j) = (k-1-i) + (k-1-j) + 2,
        where k = branch_depth (path A_i -> ... -> A_{k-1} -> merge -> B_{k-1} -> ... -> B_j).
        """
        by_dist: dict[int, list[float]] = defaultdict(list)
        k = depth
        for i in range(k):
            for j in range(k):
                v, w = f"A_{i}", f"B_{j}"
                H = blocks.get((v, w))
                if H is None:
                    continue
                r = torch.linalg.norm(H, ord="fro").item()
                d_graph = (k - 1 - i) + (k - 1 - j) + 2
                by_dist[d_graph].append(r)

        result: dict[int, dict[str, float]] = {}
        for d, vals in sorted(by_dist.items()):
            arr = np.array(vals)
            result[d] = {"R_mean": float(arr.mean())}
        return result

    @staticmethod
    def _compute_merge_gn_gap(
        blocks: dict[tuple[str, str], Tensor],
        intermediates: dict[str, Tensor],
        logits: Tensor,
        branch_depth: int,
    ) -> dict[str, float]:
        """GN-Gap = ||H^T||_F / (||H^GN||_F + eps) for the cross-branch pair through merge.

        Takes the pair (A_{k-1}, B_{k-1}) - last branch nodes closest to merge.
        H^GN_{v,w} = (1/B) sum_b J_v^T (diag(p) - pp^T) J_w.
        H^T_{v,w} = H^f_{v,w} - H^GN_{v,w}.
        """
        v_name = f"A_{branch_depth - 1}"
        w_name = f"B_{branch_depth - 1}"

        H_full = blocks.get((v_name, w_name))
        if H_full is None:
            return {"gap": 0.0, "frob_tensor": 0.0, "frob_gn": 0.0}

        fv = intermediates[v_name]
        fw = intermediates[w_name]
        B, d_out = logits.shape
        d_v = fv.shape[1]
        d_w = fw.shape[1]

        # J_v = d logits/d f_v, J_w = d logits/d f_w
        jac_v = torch.zeros(B, d_out, d_v, device=fv.device)
        jac_w = torch.zeros(B, d_out, d_w, device=fw.device)
        for c in range(d_out):
            gv = torch.autograd.grad(logits[:, c].sum(), fv, retain_graph=True)[0]
            jac_v[:, c, :] = gv
            gw = torch.autograd.grad(logits[:, c].sum(), fw, retain_graph=True)[0]
            jac_w[:, c, :] = gw

        # H^GN = (1/B) sum_b J_v^T (diag(p) - pp^T) J_w
        p = F.softmax(logits.detach(), dim=-1)
        pJ_w = p.unsqueeze(-1) * jac_w.detach()
        pTJ_w = torch.bmm(p.unsqueeze(1), jac_w.detach())
        L2_Jw = pJ_w - p.unsqueeze(-1) * pTJ_w
        H_gn = torch.bmm(jac_v.detach().transpose(1, 2), L2_Jw).sum(0) / B

        H_tensor = H_full - H_gn
        frob_gn = torch.linalg.norm(H_gn, ord="fro").item()
        frob_tensor = torch.linalg.norm(H_tensor, ord="fro").item()

        return {
            "gap": frob_tensor / (frob_gn + 1e-12),
            "frob_tensor": frob_tensor,
            "frob_gn": frob_gn,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Aggregates over seeds: mean +/- std for each checkpoint."""
        aggregated: dict[str, Any] = {}

        for ckpt_name, seed_data in raw.items():
            ckpt_agg: dict[str, Any] = {}

            # within-branch R by distance
            for branch_key in ("within_A", "within_B"):
                all_dists: dict[int, list[float]] = defaultdict(list)
                for payload in seed_data.values():
                    for d, metrics in payload[branch_key].items():
                        all_dists[d].append(metrics["R_mean"])
                dist_agg: dict[int, dict[str, float]] = {}
                for d, vals in sorted(all_dists.items()):
                    arr = np.array(vals)
                    dist_agg[d] = {
                        "R_mean": float(arr.mean()),
                        "R_std": float(arr.std()),
                    }
                ckpt_agg[branch_key] = dist_agg

            # cross-branch R by graph distance
            cross_dists: dict[int, list[float]] = defaultdict(list)
            for payload in seed_data.values():
                for d, metrics in payload["cross_AB"].items():
                    cross_dists[d].append(metrics["R_mean"])
            cross_agg: dict[int, dict[str, float]] = {}
            for d, vals in sorted(cross_dists.items()):
                arr = np.array(vals)
                cross_agg[d] = {
                    "R_mean": float(arr.mean()),
                    "R_std": float(arr.std()),
                }
            ckpt_agg["cross_AB"] = cross_agg

            # GN-Gap at merge
            gap_vals = [v["merge_gn_gap"]["gap"] for v in seed_data.values()]
            frob_t = [v["merge_gn_gap"]["frob_tensor"] for v in seed_data.values()]
            frob_g = [v["merge_gn_gap"]["frob_gn"] for v in seed_data.values()]
            ckpt_agg["merge_gn_gap"] = {
                "gap_mean": float(np.mean(gap_vals)),
                "gap_std": float(np.std(gap_vals)),
                "frob_tensor_mean": float(np.mean(frob_t)),
                "frob_tensor_std": float(np.std(frob_t)),
                "frob_gn_mean": float(np.mean(frob_g)),
                "frob_gn_std": float(np.std(frob_g)),
            }

            # rho (Jacobian spectral norms)
            rho_maxes = [v["rho"]["rho_max"] for v in seed_data.values()]
            rho_means = [v["rho"]["rho_mean"] for v in seed_data.values()]
            ckpt_agg["rho"] = {
                "rho_max_mean": float(np.mean(rho_maxes)),
                "rho_max_std": float(np.std(rho_maxes)),
                "rho_mean_mean": float(np.mean(rho_means)),
                "rho_mean_std": float(np.std(rho_means)),
            }

            # Test accuracy / loss
            test_accs = [v["test_acc"] for v in seed_data.values()]
            test_losses = [v["test_loss"] for v in seed_data.values()]
            ckpt_agg["test_acc_mean"] = float(np.mean(test_accs))
            ckpt_agg["test_acc_std"] = float(np.std(test_accs))
            ckpt_agg["test_loss_mean"] = float(np.mean(test_losses))
            ckpt_agg["test_loss_std"] = float(np.std(test_losses))

            aggregated[ckpt_name] = ckpt_agg

        return aggregated


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"
