"""Common data loading utilities."""

from __future__ import annotations

import math
import os
import random

import kagglehub
import scipy.io as sio
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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


# CIFAR-100 with proper augmentation (for convolutional models)
_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_image_loaders(
    batch_size: int,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for CIFAR-100 with augmentation.

    The CIFAR-100 counterpart of `get_cifar10_image_loaders` (same conv-friendly
    pipeline, CIFAR-100 normalization): train uses RandomCrop(32, padding=4) +
    RandomHorizontalFlip + per-channel normalization; val uses normalization only.
    """
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    val_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    train_ds = torchvision.datasets.CIFAR100(
        data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_ds = torchvision.datasets.CIFAR100(
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


# Stanford Cars (fine-grained, 196 classes) - Stage-B B1 finetune
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CARS_KAGGLE_ID = "eduardo4jesus/stanford-cars-dataset"


def _cars_transforms(image_size: int, *, train: bool) -> T.Compose:
    """ImageNet-normalized transforms; fine-grained augmentation on the train split.

    Train: RandomResizedCrop(scale in [0.5, 1.0]) + horizontal flip - mild for a
    fine-grained task where over-aggressive cropping would discard the discriminative
    object. Eval: Resize(256/224 * image_size) + CenterCrop, the standard ImageNet
    protocol matching the IMAGENET1K_V2 backbone weights.
    """
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    resize = int(round(image_size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _stratified_split(
    labels: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic per-class split of sample indices into (train, val).

    Stratification keeps every class represented in both splits - essential for a
    fine-grained 196-way problem where a plain random split could starve rare classes.
    Within each class the indices are shuffled with a fixed-seed RNG and the first
    ceil(val_fraction * n_c) (clipped to leave >=1 for training) go to validation; a
    singleton class stays entirely in training.
    """
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_class):
        idxs = by_class[label][:]
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 1:
            train_idx.extend(idxs)
            continue
        n_val = min(max(1, math.ceil(val_fraction * n)), n - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


class _StanfordCarsSplit(Dataset):
    """A (path, label) list with a transform; loads RGB images lazily via PIL."""

    def __init__(self, samples: list[tuple[str, int]], transform: T.Compose) -> None:
        self._samples = samples
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self._samples[index]
        with Image.open(path) as img:
            image = self._transform(img.convert("RGB"))
        return image, label


def _load_cars_train_samples(root: str) -> list[tuple[str, int]]:
    """Parse `cars_train_annos.mat` into [(image_path, label)] with labels in [0, 195]."""
    annos_path = os.path.join(root, "car_devkit", "devkit", "cars_train_annos.mat")
    images_dir = os.path.join(root, "cars_train", "cars_train")
    if not os.path.isfile(annos_path) or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Unexpected Stanford Cars layout under {root!r}; expected "
            "car_devkit/devkit/cars_train_annos.mat and cars_train/cars_train/."
        )
    annotations = sio.loadmat(annos_path)["annotations"][0]
    samples: list[tuple[str, int]] = []
    for record in annotations:
        fname = str(record["fname"][0])
        label = int(record["class"][0, 0]) - 1  # MATLAB 1-based -> 0-based
        samples.append((os.path.join(images_dir, fname), label))
    return samples


def get_stanford_cars_loaders(
    batch_size: int,
    *,
    image_size: int = 224,
    augment: bool = True,
    val_fraction: float = 0.2,
    num_workers: int = 4,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for Stanford Cars (196 classes).

    The dataset is fetched from Kaggle via `kagglehub`, which authenticates through the
    standard KAGGLE_USERNAME / KAGGLE_KEY environment variables or ~/.kaggle/kaggle.json
    (no credentials are read or stored here). This Kaggle mirror ships labels only for
    the 8144-image training set - the official test annotations are unlabeled - so the
    validation split is carved from the labeled training set with a deterministic,
    per-class stratified split (see `_stratified_split`).

    Images are ImageNet-normalized; the train split uses fine-grained augmentation when
    `augment` is set, matching the IMAGENET1K_V2 backbone used in Stage-B B1.
    """
    root = kagglehub.dataset_download(_CARS_KAGGLE_ID)
    samples = _load_cars_train_samples(root)
    labels = [label for _, label in samples]
    train_idx, val_idx = _stratified_split(labels, val_fraction, seed)

    train_ds = _StanfordCarsSplit(
        [samples[i] for i in train_idx],
        _cars_transforms(image_size, train=augment),
    )
    val_ds = _StanfordCarsSplit(
        [samples[i] for i in val_idx],
        _cars_transforms(image_size, train=False),
    )
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader
