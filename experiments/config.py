"""Experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    lr: float = 0.01
    batch_size: int = 128
    epochs: int = 50
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "none" | "cosine"


@dataclass
class HessianConfig:
    """Hessian computation mode selection."""

    mode: str = "exact"  # "exact" | "stochastic"
    n_probes: int = 30  # for stochastic mode
    n_power_iter: int = 20  # power iteration steps for ||H||_2
    hessian_batch_size: int = 32  # subsample from validation set


@dataclass
class Exp1Config:
    """Experiment 1: Plain vs ResNet - decay of R and C with distance."""

    depths: list[int] = field(default_factory=lambda: [8, 10, 12])
    width: int = 64
    use_layernorm: bool = True
    use_spectral_norm: bool = False
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hessian: HessianConfig = field(default_factory=HessianConfig)
    # Checkpoint epochs for metric measurement: init (0), mid, final
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class Exp2Config:
    """Experiment 2: Bottleneck ablation - sensitivity of D/C to a narrow layer.

    CIFAR-100 (K=100): ensures rank(nabla^2 l) <= K-1 = 99,
    making the bottleneck width the sole rank constraint
    for d_u in {4, 8, 16, 32, 64} (all < 99).
    With CIFAR-10 (K=10) the head layer capped the rank at 9,
    making the bottleneck indistinguishable.
    """

    dataset: str = "cifar100"  # "cifar10" | "cifar100"
    num_classes: int = 100
    depths: list[int] = field(default_factory=lambda: [6, 8])
    base_width: int = 256
    bottleneck_widths: list[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256]
    )  # 256 = control (no bottleneck); 512 is run separately with base_width=512
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hessian: HessianConfig = field(default_factory=HessianConfig)
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class Exp3Config:
    """Experiment 3: ReLU vs GELU - selective relevance of GN-Gap."""

    activations: list[str] = field(
        default_factory=lambda: ["relu", "leaky_relu", "softplus", "silu", "gelu"],
    )
    depth: int = 6
    width: int = 64
    use_layernorm: bool = False  # False for validation of H^T=0 (piecewise linearity)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hessian: HessianConfig = field(
        default_factory=lambda: HessianConfig(
            mode="exact",
            hessian_batch_size=16,
        )
    )
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class Exp4Config:
    """Experiment 4: Diamond (multi-branch) MLP - activation of the tensor term T_{u;v,w}.

    A two-branch MLP with a merge node activates all 4 terms of formula (1),
    including the mixed tensor T_{u;v,w} (term 2), which is identically zero
    in sequential and ResNet architectures.

    Sweep: branch_depth x merge_type x activation.
    """

    branch_depths: list[int] = field(default_factory=lambda: [1, 2, 3])
    width: int = 32
    merge_types: list[str] = field(default_factory=lambda: ["sum", "cat"])
    activations: list[str] = field(default_factory=lambda: ["relu", "silu"])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hessian: HessianConfig = field(default_factory=HessianConfig)
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class Exp5Config:
    """Experiment 5: Toy-Attention vs ReLU-MLP - verification of H^T_{Q,K} != 0.

    Single-head self-attention with Softmax has sigma'' != 0 => H^T_{Q,K} != 0.
    Control ReLU-MLP with the same parameter count has H^T = 0.
    Synthetic regression: y = f(X) + eps, MSE-loss.
    """

    d_model: int = 16
    seq_len: int = 8
    n_train: int = 2048
    n_val: int = 512
    noise_std: float = 0.1
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=1e-3,
            epochs=30,
            optimizer="adam",
            scheduler="cosine",
            seeds=[42, 43, 44, 45, 46],
        )
    )
    hessian: HessianConfig = field(
        default_factory=lambda: HessianConfig(
            mode="exact",
            hessian_batch_size=64,
        )
    )
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class Exp6Config:
    """Experiment 6: ResNet-18 on CIFAR-10 - GN-Gap and R/C decay in a convolutional DAG.

    Variants:
      A) ReLU (piecewise linear): H^T = 0, Gap ~= 0 - theory confirmation.
      B) SiLU (smooth): sigma'' != 0, Gap > 0 - control.

    Architectures: resnet18 (with skip), plain_resnet18 (without skip).
    """

    activations: list[str] = field(default_factory=lambda: ["relu", "silu"])
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=0.1,
            batch_size=128,
            epochs=100,
            seeds=[42, 43, 44, 45, 46],
            optimizer="sgd",
            momentum=0.9,
            weight_decay=5e-4,
            scheduler="cosine",
        )
    )
    hessian: HessianConfig = field(
        default_factory=lambda: HessianConfig(
            mode="stochastic",
            n_probes=30,
            n_power_iter=20,
            hessian_batch_size=32,
        )
    )
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class Exp7Config:
    """Exp7: COUPLE-FAC overlay optimization (the repair experiment; "Stage B" in the plan).

    COUPLE-FAC augments a block-diagonal K-FAC step with a low-rank cross-block
    Newton correction restricted to the strongly-coupled subset
    S = {(v, w) : C_{vw} > tau}, then accepts it only inside a trust region when an
    exact-HVP quadratic model improves, m(d_overlay) <= m(d_KFAC) (no-harm safeguard).
    Hyperparameters follow the revision plan: overlay rank r <= 4, coupling-measurement
    period T >= 20, and at least five seeds (inherited from `TrainingConfig`).

    Two regimes share this schema (the "adaptive" fac/dag split):
      B1 (fine-grained finetune, the headline) - an ImageNet-pretrained ResNet-50 fully
        finetuned on Stanford Cars (196 classes, ~8k train) at 224x224; see `b1_cars`.
      B2 (large-batch from scratch, the control) - a ResNet trained from random init
        with large batches B in {1024, 4096} on ImageNet-32 (32x32); see `b2_largebatch`.

    The CIFAR loaders and `TrainingConfig` defaults are reused unchanged; only the
    `dataset`/resolution/optimizer knobs are adapted per regime.
    """

    # Data / backbone
    dataset: str = (
        "stanford_cars"  # "stanford_cars" | "cifar10" | "cifar100" | "imagenet32"
    )
    num_classes: int = 196
    model: str = "resnet50"  # backbone registry key
    pretrained: bool = True  # ImageNet init (B1); False for from-scratch (B2)
    image_size: int = 224
    augment: bool = True

    # COUPLE-FAC overlay + trust-region safeguard + K-FAC provider
    tau: float = 0.1  # coupling-selection threshold S = {C_{vw} > tau}
    overlay_rank: int = 4  # rank r <= 4 of the 2x2 block-Newton correction
    overlay_n_power_iter: int = 12  # power iterations for the low-rank overlay build
    measure_period: int = 20  # T >= 20: re-measure coupling C every T steps
    coupling_n_probes: int = 30  # Hutchinson probes for the coupling estimate
    tr_radius: float = float("inf")  # trust-region radius for the no-harm safeguard
    damping: float = 1e-2  # K-FAC Tikhonov damping
    fisher_type: str = "type-2"  # curvlinops K-FAC Fisher type

    # Comparison + demonstration
    methods: list[str] = field(
        default_factory=lambda: [
            "sgd",
            "adam",
            "kfac",
            "kfac+overlay",
            "newton_cg",
        ]
    )
    target_acc: float = 0.9  # validation accuracy for the "speed to target" metric
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=0.01,
            batch_size=64,
            epochs=30,
            optimizer="sgd",
            momentum=0.9,
            weight_decay=1e-4,
            scheduler="cosine",
        )
    )

    @classmethod
    def b1_cars(cls) -> Exp7Config:
        """B1 headline: ImageNet-pretrained ResNet-50 full finetune on Stanford Cars."""
        return cls()

    @classmethod
    def b2_largebatch(cls, batch_size: int = 1024) -> Exp7Config:
        """B2 control: large-batch ResNet trained from scratch on ImageNet-32 (32x32).

        Learning rate follows the linear scaling rule lr = 0.1 * B / 256
        (Goyal et al., 2017), keeping the per-sample step size batch-invariant.
        """
        return cls(
            dataset="imagenet32",
            num_classes=1000,
            pretrained=False,
            image_size=32,
            target_acc=0.5,
            training=TrainingConfig(
                lr=0.1 * batch_size / 256,
                batch_size=batch_size,
                epochs=50,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=5e-4,
                scheduler="cosine",
            ),
        )
