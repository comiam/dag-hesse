"""Training loop with checkpoint saving."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import TrainingConfig


class Trainer:
    """Trains a model and saves state_dict at specified epochs."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._loss_fn = nn.CrossEntropyLoss()

        optimizer: torch.optim.SGD | torch.optim.Adam
        if config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        self._optimizer = optimizer

        self._scheduler = None
        if config.scheduler == "cosine":
            self._scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    @property
    def loss_fn(self) -> nn.Module:
        return self._loss_fn

    def train(
        self,
        train_loader: DataLoader,
        checkpoint_epochs: list[int] | None = None,
    ) -> dict[int, dict[str, Tensor]]:
        """Trains the model, returns {epoch: state_dict} for the requested epochs.

        checkpoint_epochs: list of epoch numbers at which to save state_dict.
            Default: [0 (init), epochs//2 (mid), epochs-1 (final)].
        """
        epochs = self._config.epochs
        if checkpoint_epochs is None:
            checkpoint_epochs = [0, epochs // 2, epochs - 1]

        checkpoints: dict[int, dict[str, Tensor]] = {}

        # Checkpoint at init (epoch 0 - before training)
        if 0 in checkpoint_epochs:
            checkpoints[0] = self._snapshot()

        for epoch in range(epochs):
            self._train_epoch(train_loader)
            if self._scheduler is not None:
                self._scheduler.step()

            ep_num = epoch + 1  # 1-indexed for compatibility
            if ep_num in checkpoint_epochs:
                checkpoints[ep_num] = self._snapshot()

        # Final
        if epochs in checkpoint_epochs and epochs not in checkpoints:
            checkpoints[epochs] = self._snapshot()

        return checkpoints

    def restore(self, state_dict: dict[str, Tensor]) -> None:
        """Restores the model from a snapshot."""
        self._model.load_state_dict(state_dict)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Computes accuracy and loss on the given loader."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y in loader:
            x, y = x.to(self._device), y.to(self._device)
            logits = self._model(x)
            loss = self._loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        return {
            "test_acc": correct / max(n, 1),
            "test_loss": total_loss / max(n, 1),
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(self._device), y.to(self._device)
            self._optimizer.zero_grad()
            logits = self._model(x)
            loss = self._loss_fn(logits, y)
            loss.backward()
            clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / max(n, 1)

    def _snapshot(self) -> dict[str, Tensor]:
        return {
            k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()
        }
