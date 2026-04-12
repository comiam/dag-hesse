"""Common data loading utilities."""

from __future__ import annotations

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cifar10_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-10."""
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=transform
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def get_cifar100_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-100."""
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR100(
        data_root, train=True, download=True, transform=transform
    )
    val_ds = torchvision.datasets.CIFAR100(
        data_root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


# CIFAR-10 with proper augmentation (for convolutional models)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_image_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-10 with augmentation.

    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + per-channel normalization.
    Val: per-channel normalization only.
    """
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    val_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    train_ds = torchvision.datasets.CIFAR10(
        data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_root,
        train=False,
        download=True,
        transform=val_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader
