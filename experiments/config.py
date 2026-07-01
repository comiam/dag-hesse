"""Experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


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
    # EKFAC rebuilds its eigenbasis only every this many steps - the eigendecomposition is
    # its dominant cost, and periodic factor refresh is standard practice (K-FAC's cheaper
    # factored step stays per-step). Only the EKFAC methods read this.
    ekfac_refresh_period: int = 10

    # Comparison + demonstration
    methods: list[str] = field(
        default_factory=lambda: [
            "sgd",
            "adam",
            "kfac",
            "ekfac",
            "kfac+overlay",
            "ekfac+overlay",
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
    def b1_mlp(cls) -> Exp7Config:
        """Optimizer-agnostic control on a full-rank (Linear) MLP - the clean EKFAC setting.

        The convolutional headline (`b1_cars`) has rank-deficient Kronecker factors, where
        EKFAC's eigenvalue correction is numerically delicate and close to K-FAC. This
        profile reruns the same optimizer comparison on a low-dimensional synthetic task
        whose Linear factors are full rank, so EKFAC is well conditioned and genuinely
        distinct - evidence that the overlay is agnostic to the block-diagonal base
        preconditioner, not specific to K-FAC.
        """
        return cls(
            dataset="synthetic",
            num_classes=10,
            model="mlp",
            pretrained=False,
            augment=False,
            # The label-noise ceiling is ~85%, and the from-scratch MLP has small initial
            # curvature, so the damped-Newton family needs a larger damping than the
            # pretrained-ResNet default (1e-2 diverges here) and a reachable target.
            damping=0.1,
            target_acc=0.7,
            ekfac_refresh_period=1,
            training=TrainingConfig(
                lr=0.05,
                batch_size=256,
                epochs=20,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=1e-4,
                scheduler="cosine",
            ),
        )

    @classmethod
    def b2_largebatch(cls, batch_size: int = 1024) -> Exp7Config:
        """B2 control: large-batch ResNet trained from scratch on ImageNet-32 (32x32).

        Learning rate follows the linear scaling rule lr = 0.1 * B / 256
        (Goyal et al., 2017), keeping the per-sample step size batch-invariant.
        """
        return cls(
            dataset="imagenet32",
            num_classes=1001,  # HF mirror keeps the index-0 background class
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


@dataclass
class Exp8Config:
    """Exp8: Stage-A Phi diagnostic - the K-FAC blind-spot map (the diagnosis experiment).

    Measures the discarded-coupling fraction Phi (Delta-M1) and where its mass sits -
    per-block diagonal norms ||H_{vv}||_F plus the coupling field C(v, w) (Delta-M2) -
    on a real convolutional DAG, with NO change to training. It is the cheap diagnosis
    that motivates the exp7 repair. The headline is the inversion Phi_ResNet >> Phi_Plain:
    skip connections preserve inter-layer coupling, so a block-diagonal preconditioner
    (K-FAC / EKFAC) discards more curvature on a ResNet than on its skip-free counterpart.

    Two regimes share this schema ("Stage A" in the plan):
      A1 (from scratch, the inversion headline) - `SegmentedResNet18` vs its skip-free
        control `SegmentedPlainResNet18` on CIFAR-100, Phi tracked at init / mid / final;
        see `a1_from_scratch`.
      A2 (pretrained, source (P)) - an ImageNet-pretrained ResNet-50 measured pre-
        finetune on Stanford Cars, where correlated pretrained Jacobians push Phi even
        higher; measured init-only (no training); see `a2_pretrained`.
    """

    # Backbones (each exposes group-major `get_param_groups`) + data
    archs: list[str] = field(default_factory=lambda: ["resnet18", "plain_resnet18"])
    dataset: str = "cifar100"  # "cifar100" | "cifar10" | "imagenet32" | "stanford_cars"
    num_classes: int = 100
    pretrained: bool = False  # ImageNet init (A2); False for from-scratch (A1)
    image_size: int = 32
    # DataLoader workers for the heavy decode paths (ImageNet-32 / Stanford Cars); raise
    # for the production ImageNet-32 anchor to clear the CPU-decode bottleneck. The light
    # CIFAR loaders keep their own small default.
    num_workers: int = 4

    # Phi estimation (Hutchinson on the parameter-space cross-block HVP)
    n_probes: int = 30
    hessian_batch_size: int = 32  # minibatch subsampled for the Hessian measurement

    # Checkpoints at which Phi is measured (init = pre-training, then mid, final)
    checkpoint_epochs: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            lr=0.1,
            batch_size=128,
            epochs=100,
            optimizer="sgd",
            momentum=0.9,
            weight_decay=5e-4,
            scheduler="cosine",
        )
    )

    @classmethod
    def a1_from_scratch(cls) -> Exp8Config:
        """A1 headline: ResNet-18 vs plain ResNet-18 from scratch on CIFAR-100."""
        return cls()

    @classmethod
    def a1_width(cls) -> Exp8Config:
        """A1 width/depth axis: ResNet-18 at 0.5x / 1x / 2x width + ResNet-34 on CIFAR-100.

        Complements a1's skip-vs-plain inversion with the capacity axis the plan ties to
        R#2 #6: how Phi scales with width (at fixed depth) and with depth (ResNet-34), from
        scratch on CIFAR-100. Same schedule / seeds / checkpoints as `a1_from_scratch`.
        """
        return cls(archs=["resnet18_w0.5", "resnet18", "resnet18_w2.0", "resnet34"])

    @classmethod
    def a1_imagenet32(cls) -> Exp8Config:
        """A1 anchor on ImageNet-32: ResNet-18 vs plain ResNet-18 (1001 classes), on-demand.

        The plan's scale anchor for the inversion headline - the same skip-vs-plain
        comparison as `a1_from_scratch`, on downsampled ImageNet-1k (32x32) instead of
        CIFAR-100. Heavy (1.28M images, fetched from the Hugging Face hub on demand), so it
        is not part of the default sweep. The mirror keeps the index-0 background class, so
        the head has 1001 outputs.
        """
        base = cls(dataset="imagenet32", num_classes=1001)
        # ImageNet-32 is 1.28M images: 30 epochs is already a full cosine schedule
        # (~300k steps), so this lighter budget is complete training, not a shortcut
        # (100 epochs, the CIFAR-100 budget, would be wasteful overkill at this scale).
        return replace(base, training=replace(base.training, epochs=30))

    @classmethod
    def a2_pretrained(cls) -> Exp8Config:
        """A2: ImageNet-pretrained ResNet-50, Phi measured pre-finetune on Stanford Cars.

        Diagnostic only - no training (epochs = 0 => Phi is read off the freshly loaded
        pretrained weights, at the single ``init`` checkpoint), isolating source (P):
        correlated pretrained Jacobians inflate the off-diagonal mass before any update.
        """
        return cls(
            archs=["resnet50"],
            dataset="stanford_cars",
            num_classes=196,
            pretrained=True,
            image_size=224,
            checkpoint_epochs=["init"],
            training=TrainingConfig(
                lr=0.01,
                batch_size=64,
                epochs=0,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=1e-4,
                scheduler="none",
            ),
        )


@dataclass
class Exp9Config:
    """Exp9: Stage-A "A3" frozen-transformer Phi ladder (the diagnosis, in language models).

    Carries exp8's parameter-space blind-spot diagnostic onto decoder-only transformers:
    with NO training, for each frozen model it measures the discarded-coupling fraction Phi
    (Delta-M1) and the coupling field C(v, w) that locates it (Delta-M2). Attention is kept
    *visible* (``attn_granularity = "qkv"``) so the coupling the paper isolates in attention
    (H^T_{Q,K} != 0, exp5) is read off directly. The "A3" question is how Phi climbs the
    model-size ladder and whether pretraining (source (P)) inflates it.

    Two backends share this schema:
      A3-toy (local, dependency-free) - `SegmentedTransformer` at growing sizes, measured
        on a random token minibatch; this is the default sweep and what the tests run; see
        `a3_toy`.
      A3-llm (on-demand) - frozen Hugging Face causal LMs (Qwen2.5 size sweep + a Llama-3.1
        cross-check), loaded in bf16 with gradients enabled; the larger rungs need on-demand
        accelerator memory; see `a3_llm`.

    The model is built (toy) or loaded (hf) once per id under a fixed build seed; ``seeds``
    then vary only the Hutchinson probes, so their spread reports measurement precision.
    """

    backend: str = "toy"  # "toy" | "hf"
    # Toy size presets (see experiments.llm._TOY_PRESETS) or Hugging Face model ids.
    models: list[str] = field(default_factory=lambda: ["tiny", "small", "base"])
    seq_len: int = 64
    hessian_batch_size: int = 8  # minibatch subsampled for the Hessian measurement
    n_probes: int = 8  # Hutchinson probes per block (raised per publication)
    dtype: str = "float32"  # "float32" | "bfloat16"
    attn_granularity: str = "qkv"  # "qkv" (attention visible) | "block" (coarse)
    # Coupling-field scope: None measures every block pair (O(L^2); the toy ladder). An
    # int k keeps only pairs within k blocks in group order plus all intra-layer pairs, so
    # the on-demand LLM ladder stays O(L) without losing the attention hotspots.
    coupling_band: int | None = None
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])

    @classmethod
    def a3_toy(cls) -> Exp9Config:
        """A3-toy: local, dependency-free ladder over toy decoder sizes (tiny->small->base)."""
        return cls()

    @classmethod
    def a3_llm(cls) -> Exp9Config:
        """A3-llm: on-demand frozen-LLM ladder (Qwen2.5 size sweep + a Llama-3.1 cross-check).

        Loaded in bf16 with gradients enabled for the parameter-space measurement; the
        larger rungs need on-demand accelerator memory. 4-bit weights are out of scope
        (not differentiable). Requires the optional ``transformers`` dependency.
        """
        return cls(
            backend="hf",
            models=[
                "Qwen/Qwen2.5-0.5B",
                "Qwen/Qwen2.5-1.5B",
                "Qwen/Qwen2.5-3B",
                "Qwen/Qwen2.5-7B",
                "Qwen/Qwen2.5-14B",
                "Qwen/Qwen2.5-32B",
                "meta-llama/Llama-3.1-8B",
            ],
            seq_len=128,
            hessian_batch_size=4,
            n_probes=8,
            dtype="bfloat16",
            coupling_band=8,
        )
