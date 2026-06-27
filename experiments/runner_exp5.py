"""Experiment 5: Toy-Attention vs ReLU-MLP - verification of H^T_{Q,K} != 0.

Claim (Example 6 of the paper): Softmax-attention has sigma'' != 0 in attention weights,
so the tensor component H^T_{Q,K} does not vanish.
For a piecewise-linear per-position ReLU-MLP: sigma'' = 0 a.e. => H^T = 0 => GN-Gap ~= 0.

Protocol:
  1. Synthetic regression: fixed teacher y = tanh(X @ W).mean(1) @ w + eps.
  2. For each seed, train both models; at each checkpoint
     compute H^f, H^GN, GN-Gap via autograd for pairs:
       Attention: (Q, K),  MLP: (block_0, block_1).
  3. Aggregate over seeds: mean +/- std.

Result:
  {model_type: {checkpoint: {gap_mean, gap_std, frob_full_mean, ...}}}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from experiments.config import Exp5Config
from experiments.models import ToyAttentionModel, ToyReluMLP
from experiments.utils import set_seed

logger = logging.getLogger(__name__)

_TEACHER_SEED = 0


class Exp5Runner:
    """Runner for Experiment 5 (Toy-Attention vs ReLU-MLP)."""

    def __init__(self, config: Exp5Config, device: torch.device) -> None:
        self._cfg = config
        self._device = device

    def run(self) -> dict[str, Any]:
        """Returns {model_type: {ckpt: {metric_mean, metric_std, ...}}}."""
        all_results: dict[str, Any] = {}

        for model_type in ("attention", "relu_mlp"):
            logger.info("=== Model: %s ===", model_type)
            raw: dict[str, dict[str, Any]] = {}

            for seed in self._cfg.training.seeds:
                logger.info("  Seed %d", seed)
                set_seed(seed)

                train_loader, val_loader = self._make_synthetic_data(seed)

                if model_type == "attention":
                    model: nn.Module = ToyAttentionModel(
                        d_model=self._cfg.d_model,
                        seq_len=self._cfg.seq_len,
                    )
                else:
                    model = ToyReluMLP(
                        d_model=self._cfg.d_model,
                        seq_len=self._cfg.seq_len,
                    )

                model = model.to(self._device)
                loss_fn = nn.MSELoss()

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self._cfg.training.lr,
                    weight_decay=self._cfg.training.weight_decay,
                )
                scheduler = None
                if self._cfg.training.scheduler == "cosine":
                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=self._cfg.training.epochs
                    )

                epochs = self._cfg.training.epochs
                ckpt_epochs = self._resolve_checkpoint_epochs(epochs)
                checkpoints: dict[int, dict[str, Tensor]] = {}

                # Init checkpoint
                if 0 in ckpt_epochs:
                    checkpoints[0] = self._snapshot(model)

                for epoch in range(epochs):
                    self._train_epoch(model, train_loader, loss_fn, optimizer)
                    if scheduler is not None:
                        scheduler.step()
                    ep_num = epoch + 1
                    if ep_num in ckpt_epochs:
                        checkpoints[ep_num] = self._snapshot(model)

                if epochs in ckpt_epochs and epochs not in checkpoints:
                    checkpoints[epochs] = self._snapshot(model)

                # Evaluate at each checkpoint
                for ep, state in checkpoints.items():
                    model.load_state_dict(state)
                    model.to(self._device)
                    model.eval()

                    x_batch, y_batch = self._sample_hessian_batch(val_loader)

                    node_v = "Q" if model_type == "attention" else "block_0"
                    node_w = "K" if model_type == "attention" else "block_1"
                    metrics = self._compute_gn_gap(
                        model, x_batch, y_batch, loss_fn, node_v, node_w
                    )

                    # Evaluate test loss
                    eval_loss = self._evaluate(model, val_loader, loss_fn)
                    metrics["test_loss"] = eval_loss

                    ckpt_name = _epoch_to_label(ep, epochs)
                    raw.setdefault(ckpt_name, {})[f"seed_{seed}"] = metrics

            all_results[model_type] = self._aggregate(raw)

        return all_results

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------

    def _make_synthetic_data(self, seed: int) -> tuple[DataLoader, DataLoader]:
        """Synthetic regression with a fixed nonlinear teacher.

        Teacher: y = tanh(X @ W_teacher).mean(dim=1) @ w_readout.
        Targets are standardized (mean=0, std=1) using the training set.
        """
        rng = torch.Generator().manual_seed(seed)
        S, d = self._cfg.seq_len, self._cfg.d_model

        X_train = torch.randn(self._cfg.n_train, S, d, generator=rng)
        X_val = torch.randn(self._cfg.n_val, S, d, generator=rng)

        # Fixed teacher (seed=0, independent of training seed)
        teacher_gen = torch.Generator().manual_seed(_TEACHER_SEED)
        W_teacher = torch.randn(d, d, generator=teacher_gen) / d**0.5
        w_readout = torch.randn(d, 1, generator=teacher_gen) / d**0.5

        def teacher_fn(X: Tensor) -> Tensor:
            return torch.tanh(X @ W_teacher).mean(dim=1) @ w_readout

        y_train = teacher_fn(X_train)
        y_val = teacher_fn(X_val)

        # Noise
        y_train = y_train + self._cfg.noise_std * torch.randn_like(
            y_train, generator=rng
        )
        y_val = y_val + self._cfg.noise_std * torch.randn_like(y_val, generator=rng)

        # Standardize using training statistics
        y_mu = y_train.mean()
        y_sigma = y_train.std().clamp(min=1e-8)
        y_train = (y_train - y_mu) / y_sigma
        y_val = (y_val - y_mu) / y_sigma

        bs = self._cfg.training.batch_size
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=bs, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=bs, shuffle=False
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _train_epoch(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        model.train()
        device = next(model.parameters()).device
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    @staticmethod
    def _evaluate(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        model.eval()
        device = next(model.parameters()).device
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
        return total_loss / max(n, 1)

    @staticmethod
    def _snapshot(model: nn.Module) -> dict[str, Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
    # GN-Gap computation (unified for both models)
    # ------------------------------------------------------------------

    def _compute_gn_gap(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        loss_fn: nn.Module,
        node_v: str,
        node_w: str,
    ) -> dict[str, Any]:
        """H^f_{v,w}, H^GN_{v,w}, GN-Gap via autograd.

        Works for both models: Attention (Q, K) and MLP (block_0, block_1).
        Intermediate activations (B, S, d) are obtained via forward_with_intermediates.
        H^GN = (2/B) sum_b J_v[b]^T J_w[b]  (MSE with mean reduction).
        """
        output, intermediates = model.forward_with_intermediates(x)  # type: ignore[operator]
        fv = intermediates[node_v]
        fw = intermediates[node_w]
        loss = loss_fn(output, y)  # type: ignore[operator]

        B = x.shape[0]
        d_v = fv[0].numel()  # S * d_model
        d_w = fw[0].numel()
        d_out = output.shape[1]  # 1

        H_full = self._cross_hessian(loss, fv, fw)

        jac_v = self._batch_jacobian(output, fv, B, d_out, d_v)
        jac_w = self._batch_jacobian(output, fw, B, d_out, d_w)
        H_gn = torch.bmm(jac_v.transpose(1, 2), jac_w).sum(0) * (2.0 / B)

        H_tensor = H_full - H_gn
        frob_full = torch.linalg.norm(H_full, ord="fro").item()
        frob_gn = torch.linalg.norm(H_gn, ord="fro").item()
        frob_tensor = torch.linalg.norm(H_tensor, ord="fro").item()

        gap = frob_tensor / (frob_gn + 1e-12)

        return {
            "gap": gap,
            "frob_full": frob_full,
            "frob_gn": frob_gn,
            "frob_tensor": frob_tensor,
        }

    # ------------------------------------------------------------------
    # Autograd helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_hessian(loss: Tensor, fv: Tensor, fw: Tensor) -> Tensor:
        """d^2 L / df_v df_w -> (numel_v, numel_w).

        Works for any-shape intermediates (flattened per sample).
        """
        B = fv.shape[0]
        d_v = fv[0].numel()
        d_w = fw[0].numel()

        grad_fw = torch.autograd.grad(loss, fw, create_graph=True)[0]
        grad_fw_flat = grad_fw.reshape(B, d_w)

        H = torch.zeros(d_v, d_w, device=fv.device)
        for j in range(d_w):
            col_sum = grad_fw_flat[:, j].sum()
            row = torch.autograd.grad(col_sum, fv, retain_graph=True)[0]
            H[:, j] = row.reshape(B, d_v).sum(0)

        return H

    @staticmethod
    def _batch_jacobian(
        output: Tensor,
        f: Tensor,
        B: int,
        d_out: int,
        d_f: int,
    ) -> Tensor:
        """d output/d f -> (B, d_out, d_f)."""
        jac = torch.zeros(B, d_out, d_f, device=f.device)
        for k in range(d_out):
            grad_k = torch.autograd.grad(
                output[:, k].sum(),
                f,
                retain_graph=True,
            )[0]
            jac[:, k, :] = grad_k.reshape(B, d_f)
        return jac

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Aggregates over seeds: mean +/- std for each checkpoint."""
        keys = ("gap", "frob_full", "frob_gn", "frob_tensor", "test_loss")
        aggregated: dict[str, Any] = {}
        for ckpt_name, seed_data in raw.items():
            vals = {k: [v[k] for v in seed_data.values()] for k in keys}
            agg: dict[str, float] = {}
            for k, vs in vals.items():
                agg[f"{k}_mean"] = float(np.mean(vs))
                agg[f"{k}_std"] = float(np.std(vs))
            aggregated[ckpt_name] = agg
        return aggregated


def _epoch_to_label(epoch: int, total: int) -> str:
    if epoch == 0:
        return "init"
    if epoch == total:
        return "final"
    return "mid"
