"""Entry point: run DAG-Hesse experiments.

Subcommands:
  exp1  - Plain vs ResNet: decay of R and C with distance.
  exp2  - Bottleneck ablation: sensitivity of R/C to a narrow layer.
  exp3  - ReLU vs GELU: selective relevance of GN-Gap.
  exp4  - Diamond MLP: activation of the tensor term T_{u;v,w}.
  exp5  - Toy-Attention vs ReLU-MLP: verification of H^T_{Q,K} != 0.
  exp6  - ResNet-18 conv: GN-Gap and R/C decay in a convolutional DAG.
  exp7  - COUPLE-FAC overlay optimization (K-FAC + coupling-gated correction).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from experiments.config import (
    Exp1Config,
    Exp2Config,
    Exp3Config,
    Exp4Config,
    Exp5Config,
    Exp6Config,
    Exp7Config,
    HessianConfig,
    TrainingConfig,
)
from experiments.plotting import (
    plot_exp1_all_checkpoints,
    plot_exp1_depth_sweep,
    plot_exp2_acc_delta,
    plot_exp2_all_checkpoints,
    plot_exp2_depth_sweep,
    plot_exp2_diff_heatmap,
    plot_exp2_dual_metric,
    plot_exp2_triple_metric,
    plot_exp3_composite,
    plot_exp3_gn_gap,
    plot_exp3_per_distance,
    plot_exp4_cross_branch_decay,
    plot_exp4_gn_gap_heatmap,
    plot_exp4_rho_summary,
    plot_exp6_all_checkpoints,
    plot_exp6_gn_gap,
    plot_exp6_gn_gap_per_distance,
    plot_rho_summary,
)
from experiments.runner_exp1 import Exp1Runner
from experiments.runner_exp2 import Exp2Runner
from experiments.runner_exp3 import Exp3Runner
from experiments.runner_exp4 import Exp4Runner
from experiments.runner_exp5 import Exp5Runner
from experiments.runner_exp6 import Exp6Runner
from experiments.runner_exp7 import Exp7Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Common CLI arguments
# ======================================================================


def _add_device_args(parser: argparse.ArgumentParser) -> None:
    """Registers device / output arguments shared by every subcommand."""
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help="CUDA GPU index (e.g. --gpu 1). Overrides --device to cuda.",
    )
    parser.add_argument("--output-dir", type=str, default=None)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Registers arguments shared by the diagnostic experiments (exp1-exp6)."""
    _add_device_args(parser)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--hessian-mode",
        type=str,
        default="stochastic",
        choices=["exact", "stochastic"],
    )
    parser.add_argument("--n-probes", type=int, default=30)
    parser.add_argument("--n-power-iter", type=int, default=20)
    parser.add_argument("--hessian-batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
    )


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        return torch.device("cuda")
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def _make_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        seeds=args.seeds,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
    )


def _make_hessian_config(args: argparse.Namespace) -> HessianConfig:
    return HessianConfig(
        mode=args.hessian_mode,
        n_probes=args.n_probes,
        n_power_iter=args.n_power_iter,
        hessian_batch_size=args.hessian_batch_size,
    )


def _save_results(results: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info("Results saved to %s/results.json", out_dir)


# ======================================================================
# Experiment 1
# ======================================================================


def _add_exp1_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--use-layernorm",
        action="store_true",
        default=True,
        help="Use LayerNorm in PlainMLP blocks (default: True)",
    )
    parser.add_argument("--no-layernorm", dest="use_layernorm", action="store_false")
    parser.add_argument(
        "--use-spectral-norm",
        action="store_true",
        default=False,
        help="Apply spectral normalization to all PlainMLP Linear layers",
    )


def _run_exp1(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    config = Exp1Config(
        depths=args.depths,
        width=args.width,
        use_layernorm=args.use_layernorm,
        use_spectral_norm=args.use_spectral_norm,
        training=_make_training_config(args),
        hessian=_make_hessian_config(args),
    )

    runner = Exp1Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp1")
    _save_results(results, out_dir)

    # Plots: depth sweep
    for metric in ("R", "C"):
        plot_exp1_depth_sweep(
            results,
            metric=metric,
            save_path=str(out_dir / f"exp1_depth_sweep_{metric}.png"),
        )
        logger.info(
            "Depth sweep plot saved: %s", out_dir / f"exp1_depth_sweep_{metric}.png"
        )

    # Plots: per-depth 3-panel
    for depth in args.depths:
        for metric in ("R", "C"):
            plot_exp1_all_checkpoints(
                results,
                metric=metric,
                depth=depth,
                save_path=str(out_dir / f"exp1_d{depth}_{metric}.png"),
            )

    # rho summary
    plot_rho_summary(results, save_path=str(out_dir / "exp1_rho.png"))
    logger.info("Rho summary plot saved: %s", out_dir / "exp1_rho.png")


# ======================================================================
# Experiment 1b: theorem verification under spectral normalization
# ======================================================================


def _add_exp1b_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--width", type=int, default=64)


def _run_exp1b(args: argparse.Namespace) -> None:
    """Exp.1b: same protocol as Exp.1, but with spectral normalization.

    Spectral norm guarantees ||W||_2 = 1 for all Linear layers,
    which gives rho <= 1 for ReLU; for a chain s = max|Ch| = 1, s*rho = rho <= 1.
    This is a direct verification of Theorem 6 in its own regime.
    """
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    config = Exp1Config(
        depths=args.depths,
        width=args.width,
        use_layernorm=False,  # no LayerNorm - pure piecewise linearity
        use_spectral_norm=True,  # rho <= 1 by construction
        training=_make_training_config(args),
        hessian=_make_hessian_config(args),
    )

    runner = Exp1Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp1b")
    _save_results(results, out_dir)

    for metric in ("R", "C"):
        plot_exp1_depth_sweep(
            results,
            metric=metric,
            save_path=str(out_dir / f"exp1b_depth_sweep_{metric}.png"),
        )

    for depth in args.depths:
        for metric in ("R", "C"):
            plot_exp1_all_checkpoints(
                results,
                metric=metric,
                depth=depth,
                save_path=str(out_dir / f"exp1b_d{depth}_{metric}.png"),
            )

    plot_rho_summary(results, save_path=str(out_dir / "exp1b_rho.png"))
    logger.info("Exp1b done. Results in %s", out_dir)


# ======================================================================
# Experiment 2
# ======================================================================


def _add_exp2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--base-width", type=int, default=256)
    parser.add_argument(
        "--bottleneck-widths",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
    )


def _run_exp2(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    num_classes = {"cifar10": 10, "cifar100": 100}[args.dataset]

    config = Exp2Config(
        dataset=args.dataset,
        num_classes=num_classes,
        depths=args.depths,
        base_width=args.base_width,
        bottleneck_widths=args.bottleneck_widths,
        training=_make_training_config(args),
        hessian=_make_hessian_config(args),
    )

    runner = Exp2Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp2")
    _save_results(results, out_dir)

    # Plots: depth sweep
    for metric in ("C_far", "R_far", "D_far"):
        plot_exp2_depth_sweep(
            results,
            metric=metric,
            save_path=str(out_dir / f"exp2_depth_sweep_{metric}.png"),
        )
        logger.info(
            "Depth sweep plot saved: %s", out_dir / f"exp2_depth_sweep_{metric}.png"
        )

    # Plots: per-depth 3-panel
    for depth in args.depths:
        for metric in ("C_far", "R_far", "D_far"):
            plot_exp2_all_checkpoints(
                results,
                metric=metric,
                depth=depth,
                save_path=str(out_dir / f"exp2_d{depth}_{metric}.png"),
            )

    # Plots: difference heatmaps delta_C_ij and delta_D_ij
    for depth in args.depths:
        for ckpt in ("init", "final"):
            for hm_metric in ("C", "D"):
                plot_exp2_diff_heatmap(
                    results,
                    depth=depth,
                    checkpoint=ckpt,
                    metric=hm_metric,
                    save_path=str(
                        out_dir / f"exp2_diff_heatmap_{hm_metric}_d{depth}_{ckpt}.png"
                    ),
                )
                logger.info(
                    "Diff heatmap saved: %s",
                    out_dir / f"exp2_diff_heatmap_{hm_metric}_d{depth}_{ckpt}.png",
                )

    # Plots: per-depth accuracy/loss delta from bottleneck
    for ckpt in ("mid", "final"):
        plot_exp2_acc_delta(
            results,
            checkpoint=ckpt,
            save_path=str(out_dir / f"exp2_acc_delta_{ckpt}.png"),
        )
        logger.info("Acc delta plot saved: %s", out_dir / f"exp2_acc_delta_{ckpt}.png")

    # Plots: dual metric C_far + R_far (backward compat)
    for ckpt in ("init", "mid", "final"):
        plot_exp2_dual_metric(
            results,
            checkpoint=ckpt,
            save_path=str(out_dir / f"exp2_dual_{ckpt}.png"),
        )
        logger.info("Dual metric plot saved: %s", out_dir / f"exp2_dual_{ckpt}.png")

    # Plots: triple metric C_far + R_far + D_far
    for ckpt in ("init", "mid", "final"):
        plot_exp2_triple_metric(
            results,
            checkpoint=ckpt,
            save_path=str(out_dir / f"exp2_triple_{ckpt}.png"),
        )
        logger.info("Triple metric plot saved: %s", out_dir / f"exp2_triple_{ckpt}.png")


# ======================================================================
# Experiment 3
# ======================================================================


def _add_exp3_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--activations",
        type=str,
        nargs="+",
        default=["relu", "leaky_relu", "softplus", "silu", "gelu"],
    )
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--use-layernorm",
        action="store_true",
        default=False,
        help="Use LayerNorm in MLP blocks (disable for piecewise-linear validation)",
    )


def _run_exp3(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    config = Exp3Config(
        activations=args.activations,
        depth=args.depth,
        width=args.width,
        use_layernorm=args.use_layernorm,
        training=_make_training_config(args),
        hessian=_make_hessian_config(args),
    )

    runner = Exp3Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp3")
    _save_results(results, out_dir)

    # Plots
    plot_exp3_gn_gap(
        results,
        save_path=str(out_dir / "exp3_gn_gap.png"),
    )
    logger.info("GN-Gap bar plot saved: %s", out_dir / "exp3_gn_gap.png")

    plot_exp3_per_distance(
        results,
        save_path=str(out_dir / "exp3_gn_gap_per_distance.png"),
    )
    logger.info(
        "GN-Gap per-distance plot saved: %s", out_dir / "exp3_gn_gap_per_distance.png"
    )

    plot_exp3_composite(
        results,
        checkpoint="mid",
        save_path=str(out_dir / "exp3_composite.png"),
    )
    logger.info("Composite plot saved: %s", out_dir / "exp3_composite.png")


# ======================================================================
# Experiment 4
# ======================================================================


def _add_exp4_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--branch-depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--merge-types",
        type=str,
        nargs="+",
        default=["sum", "cat"],
    )
    parser.add_argument(
        "--activations",
        type=str,
        nargs="+",
        default=["relu", "silu"],
    )


def _run_exp4(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    config = Exp4Config(
        branch_depths=args.branch_depths,
        width=args.width,
        merge_types=args.merge_types,
        activations=args.activations,
        training=_make_training_config(args),
        hessian=_make_hessian_config(args),
    )

    runner = Exp4Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp4")
    _save_results(results, out_dir)

    # Plots
    plot_exp4_gn_gap_heatmap(
        results,
        checkpoint="final",
        save_path=str(out_dir / "exp4_gn_gap_heatmap.png"),
    )
    logger.info("GN-Gap heatmap saved: %s", out_dir / "exp4_gn_gap_heatmap.png")

    plot_exp4_cross_branch_decay(
        results,
        save_path=str(out_dir / "exp4_cross_branch_decay.png"),
    )
    logger.info(
        "Cross-branch decay plot saved: %s", out_dir / "exp4_cross_branch_decay.png"
    )

    plot_exp4_rho_summary(
        results,
        save_path=str(out_dir / "exp4_rho.png"),
    )
    logger.info("Rho summary plot saved: %s", out_dir / "exp4_rho.png")


# ======================================================================
# Experiment 5: Toy-Attention vs ReLU-MLP
# ======================================================================


def _add_exp5_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--noise-std", type=float, default=0.1)


def _run_exp5(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    training_cfg = _make_training_config(args)
    training_cfg.optimizer = "adam"
    training_cfg.lr = 1e-3

    config = Exp5Config(
        d_model=args.d_model,
        seq_len=args.seq_len,
        n_train=args.n_train,
        n_val=args.n_val,
        noise_std=args.noise_std,
        training=training_cfg,
        hessian=_make_hessian_config(args),
    )

    runner = Exp5Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp5")
    _save_results(results, out_dir)
    logger.info("Exp5 done. Results in %s", out_dir)


# ======================================================================
# Experiment 6: ResNet-18 conv
# ======================================================================


def _add_exp6_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--activations",
        type=str,
        nargs="+",
        default=["relu", "silu"],
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.set_defaults(weight_decay=5e-4)


def _run_exp6(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    training_cfg = _make_training_config(args)
    training_cfg.lr = args.lr
    training_cfg.batch_size = args.batch_size

    config = Exp6Config(
        activations=args.activations,
        training=training_cfg,
        hessian=_make_hessian_config(args),
    )

    runner = Exp6Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or "results/exp6")
    _save_results(results, out_dir)

    # Plots: decay per activation
    for act in args.activations:
        for metric in ("R", "C", "D"):
            plot_exp6_all_checkpoints(
                results,
                activation=act,
                metric=metric,
                save_path=str(out_dir / f"exp6_{act}_{metric}.png"),
            )
            logger.info("Plot saved: %s", out_dir / f"exp6_{act}_{metric}.png")

    # Plots: GN-Gap bar chart
    for ckpt in ("mid", "final"):
        plot_exp6_gn_gap(
            results,
            checkpoint=ckpt,
            save_path=str(out_dir / f"exp6_gn_gap_{ckpt}.png"),
        )
        logger.info("GN-Gap plot saved: %s", out_dir / f"exp6_gn_gap_{ckpt}.png")

    # Plots: GN-Gap per distance
    plot_exp6_gn_gap_per_distance(
        results,
        checkpoint="final",
        save_path=str(out_dir / "exp6_gn_gap_per_distance.png"),
    )
    logger.info(
        "GN-Gap per-distance plot saved: %s", out_dir / "exp6_gn_gap_per_distance.png"
    )

    logger.info("Exp6 done. Results in %s", out_dir)


# ======================================================================
# Exp7: COUPLE-FAC overlay optimization (the repair experiment)
# ======================================================================

# Class count per exp7 dataset; keeps the backbone head consistent when
# --dataset overrides the profile default.
_EXP7_NUM_CLASSES = {
    "stanford_cars": 196,
    "cifar10": 10,
    "cifar100": 100,
    "imagenet32": 1000,
}


def _add_exp7_args(parser: argparse.ArgumentParser) -> None:
    _add_device_args(parser)
    parser.add_argument(
        "--profile",
        type=str,
        default="b1",
        choices=["b1", "b2"],
        help="b1: pretrained ResNet-50 finetune on Stanford Cars (headline); "
        "b2: large-batch ResNet from scratch on ImageNet-32 (control).",
    )
    # Optional overrides; a value of None leaves the profile default untouched.
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["stanford_cars", "cifar10", "cifar100", "imagenet32"],
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=["resnet50", "resnet18"]
    )
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--overlay-rank", type=int, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--measure-period", type=int, default=None)
    parser.add_argument("--target-acc", type=float, default=None)


def _make_exp7_config(args: argparse.Namespace) -> Exp7Config:
    """Selects the regime profile, then applies any explicit CLI overrides."""
    if args.profile == "b1":
        base = Exp7Config.b1_cars()
    else:
        base = Exp7Config.b2_largebatch()

    training = replace(
        base.training,
        **{
            key: value
            for key, value in {
                "epochs": args.epochs,
                "seeds": args.seeds,
                "batch_size": args.batch_size,
                "lr": args.lr,
            }.items()
            if value is not None
        },
    )

    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "model": args.model,
            "methods": args.methods,
            "tau": args.tau,
            "overlay_rank": args.overlay_rank,
            "damping": args.damping,
            "measure_period": args.measure_period,
            "target_acc": args.target_acc,
        }.items()
        if value is not None
    }
    # A dataset switch implies its class count (keeps the head consistent).
    if "dataset" in overrides:
        overrides["num_classes"] = _EXP7_NUM_CLASSES[overrides["dataset"]]

    return replace(base, training=training, **overrides)


def _run_exp7(args: argparse.Namespace) -> None:
    device = _resolve_device(args)
    logger.info("Device: %s", device)

    config = _make_exp7_config(args)
    runner = Exp7Runner(config, device)
    results = runner.run()

    out_dir = Path(args.output_dir or f"results/exp7_{args.profile}")
    _save_results(results, out_dir)
    logger.info("Exp7 done. Results in %s", out_dir)


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAG-Hesse experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    # exp1
    p1 = subparsers.add_parser("exp1", help="Plain vs ResNet resonance decay")
    _add_common_args(p1)
    _add_exp1_args(p1)

    # exp1b
    p1b = subparsers.add_parser(
        "exp1b",
        help="Exp.1 with spectral norm (theorem verification regime)",
    )
    _add_common_args(p1b)
    _add_exp1b_args(p1b)

    # exp2
    p2 = subparsers.add_parser("exp2", help="Bottleneck ablation")
    _add_common_args(p2)
    _add_exp2_args(p2)

    # exp3
    p3 = subparsers.add_parser("exp3", help="ReLU vs GELU GN-Gap")
    _add_common_args(p3)
    _add_exp3_args(p3)

    # exp4
    p4 = subparsers.add_parser("exp4", help="Diamond MLP: tensor term activation")
    _add_common_args(p4)
    _add_exp4_args(p4)

    # exp5
    p5 = subparsers.add_parser("exp5", help="Toy-Attention vs ReLU-MLP GN-Gap")
    _add_common_args(p5)
    _add_exp5_args(p5)

    # exp6
    p6 = subparsers.add_parser("exp6", help="ResNet-18 conv: GN-Gap + R/C/D decay")
    _add_common_args(p6)
    _add_exp6_args(p6)

    # exp7
    p7 = subparsers.add_parser(
        "exp7", help="COUPLE-FAC overlay optimization (repair experiment)"
    )
    _add_exp7_args(p7)

    args = parser.parse_args()

    dispatch = {
        "exp1": _run_exp1,
        "exp1b": _run_exp1b,
        "exp2": _run_exp2,
        "exp3": _run_exp3,
        "exp4": _run_exp4,
        "exp5": _run_exp5,
        "exp6": _run_exp6,
        "exp7": _run_exp7,
    }
    dispatch[args.experiment](args)


def _make_serializable(obj):
    """Recursively converts dicts with int keys to str keys for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    return obj


if __name__ == "__main__":
    main()
