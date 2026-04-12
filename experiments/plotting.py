"""Visualization of DAG-Hesse experiments.

Experiment 1:
  1. plot_exp1 - single panel R_bar(d) or C_bar(d) vs distance.
  2. plot_exp1_all_checkpoints - 3 panels (init/mid/final).
  3. plot_exp1_depth_sweep - rows=depths, columns=checkpoints.
  4. plot_rho_summary - rho_max by depth x checkpoint x arch.

Experiment 4:
  1. plot_exp4_gn_gap_heatmap - GN-Gap heatmap: merge x activation.
  2. plot_exp4_cross_branch_decay - R_bar_{AB}(d_graph) by checkpoint.
  3. plot_exp4_rho_summary - rho_max by depth x config x checkpoint.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_ARCH_STYLES = {
    "plain": {"color": "#d62728", "marker": "o", "label": "Plain MLP"},
    "residual": {"color": "#1f77b4", "marker": "s", "label": "Residual MLP"},
}


def _plot_one_panel(
    depth_data: dict[str, Any],
    checkpoint: str,
    metric: str,
    ax: Axes,
    log_scale: bool = True,
) -> None:
    """Draws a single R_bar(d) or C_bar(d) panel for two architectures."""
    for arch in ("plain", "residual"):
        ckpt_data = depth_data[arch].get(checkpoint, {})
        dist_data = ckpt_data.get("distances", ckpt_data)
        distances = sorted(int(d) for d in dist_data.keys())
        if not distances:
            continue
        means = [dist_data[d][f"{metric}_mean"] for d in distances]
        stds = [dist_data[d][f"{metric}_std"] for d in distances]

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        dist_arr = np.array(distances)

        style = _ARCH_STYLES[arch]
        ax.plot(
            dist_arr,
            means_arr,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(
            dist_arr,
            np.maximum(means_arr - stds_arr, 1e-15 if log_scale else 0),
            means_arr + stds_arr,
            alpha=0.2,
            color=style["color"],
        )

    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_exp1(
    results: dict[str, Any],
    checkpoint: str = "final",
    metric: str = "R",
    ax: Axes | None = None,
    log_scale: bool = True,
    depth: int | None = None,
) -> Figure:
    """Draws a single R_bar(d) or C_bar(d) panel.

    Args:
        results: Output of Exp1Runner.run().
            Supports both the new format {depth: {arch: ...}}
            and legacy {arch: {ckpt: {dist: ...}}}.
        depth: Depth (for multi-depth format). If None, uses the first one.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    depth_data = _resolve_depth_data(results, depth)
    _plot_one_panel(depth_data, checkpoint, metric, ax, log_scale)

    metric_label = (
        r"$\bar{\mathcal{R}}(d)$" if metric == "R" else r"$\bar{\mathcal{C}}(d)$"
    )
    ax.set_xlabel("Layer distance $d$")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs distance ({checkpoint})")
    ax.legend()

    return fig


def plot_exp1_all_checkpoints(
    results: dict[str, Any],
    metric: str = "R",
    save_path: str | None = None,
    depth: int | None = None,
) -> Figure:
    """3-panel plot: init / mid / final for one depth."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    depth_data = _resolve_depth_data(results, depth)

    for i, ckpt in enumerate(("init", "mid", "final")):
        _plot_one_panel(depth_data, ckpt, metric, axes[i])
        metric_label = (
            r"$\bar{\mathcal{R}}(d)$" if metric == "R" else r"$\bar{\mathcal{C}}(d)$"
        )
        axes[i].set_xlabel("Layer distance $d$")
        axes[i].set_title(f"{ckpt}")
        if i == 0:
            axes[i].set_ylabel(metric_label)
            axes[i].legend()

    depth_label = depth if depth else list(results.keys())[0]
    fig.suptitle(
        (
            rf"Inter-layer resonance $\mathcal{{R}}$: depth={depth_label}"
            if metric == "R"
            else rf"Geometric coupling $\mathcal{{C}}$: depth={depth_label}"
        ),
        fontsize=13,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_exp1_depth_sweep(
    results: dict[str, Any],
    metric: str = "R",
    save_path: str | None = None,
) -> Figure:
    """Grid of plots: rows = depths, columns = checkpoints (init/mid/final).

    Visually confirms that as depth increases, decay in plain becomes
    more pronounced while residual remains stable.
    """
    depths = sorted(int(d) for d in results.keys())
    checkpoints = ["init", "mid", "final"]

    n_rows = len(depths)
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for r, depth in enumerate(depths):
        depth_data = results[str(depth)]
        for c, ckpt in enumerate(checkpoints):
            ax = axes[r][c]
            _plot_one_panel(depth_data, ckpt, metric, ax)

            if r == 0:
                ax.set_title(ckpt, fontsize=12)
            if c == 0:
                metric_sym = (
                    r"$\bar{\mathcal{R}}$" if metric == "R" else r"$\bar{\mathcal{C}}$"
                )
                ax.set_ylabel(f"depth={depth}\n{metric_sym}", fontsize=11)
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("Layer distance $d$")
            else:
                ax.set_xlabel("")
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=8)

    metric_name = (
        r"Inter-layer resonance $\mathcal{R}$"
        if metric == "R"
        else r"Geometric coupling $\mathcal{C}$"
    )
    fig.suptitle(f"{metric_name}: depth sweep", fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_rho_summary(
    results: dict[str, Any],
    save_path: str | None = None,
) -> Figure:
    """Visualization of rho_max and rho_mean by depth x checkpoint.

    Two subplots (plain / residual), each showing
    rho_max(checkpoint) for different depths - annotation for Theorem S8.
    """
    depths = sorted(int(d) for d in results.keys())
    checkpoints = ["init", "mid", "final"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, arch in zip(axes, ("plain", "residual"), strict=False):
        for depth in depths:
            depth_data = results[str(depth)]
            rho_maxes: list[float] = []
            for ckpt in checkpoints:
                ckpt_data = depth_data[arch].get(ckpt, {})
                rho_info = ckpt_data.get("rho", {})
                rho_maxes.append(rho_info.get("rho_max_mean", 0.0))

            ax.plot(
                checkpoints,
                rho_maxes,
                marker="o",
                label=f"depth={depth}",
                linewidth=1.5,
            )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label=r"$\rho=1$")
        ax.set_title(_ARCH_STYLES[arch]["label"])
        ax.set_ylabel(r"$\rho_{\max}$")
        ax.set_xlabel("Checkpoint")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Jacobian spectral norm $\rho_{\max} = \max_i \|J_i\|_2$ across training",
        fontsize=13,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def _resolve_depth_data(
    results: dict[str, Any],
    depth: int | None = None,
) -> dict[str, Any]:
    """Extracts data for a specific depth from results.

    Also supports the legacy format (without depth key).
    """
    # New format: {depth_str: {arch: ...}}
    first_key = next(iter(results))
    if first_key in ("plain", "residual"):
        # Legacy
        return results

    if depth is not None:
        return results[str(depth)]
    return results[first_key]


# ======================================================================
# Experiment 2: Bottleneck ablation
# ======================================================================

_EXP2_STYLE = {"color": "#2ca02c", "marker": "D"}

_EXP2_METRIC_LABELS = {
    "C_far": (
        r"$\bar{\mathcal{C}}_{\mathrm{far}}$",
        r"Far coupling $\mathcal{C}_{\mathrm{far}}$",
    ),
    "R_far": (
        r"$\bar{\mathcal{R}}_{\mathrm{far}}$",
        r"Far resonance $\mathcal{R}_{\mathrm{far}}$",
    ),
    "D_far": (
        r"$\bar{\mathcal{D}}_{\mathrm{far}}$",
        r"Far stable rank $\mathcal{D}_{\mathrm{far}}$",
    ),
}


def _plot_exp2_panel(
    depth_data: dict[str, Any],
    checkpoint: str,
    metric: str,
    ax: Axes,
) -> None:
    """Draws a single C_far (or R_far) vs bottleneck width panel."""
    widths_sorted = sorted(int(w) for w in depth_data.keys())
    means, stds = [], []
    for w in widths_sorted:
        ckpt_data = depth_data[str(w)].get(checkpoint, {})
        means.append(ckpt_data.get(f"{metric}_mean", 0.0))
        stds.append(ckpt_data.get(f"{metric}_std", 0.0))

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    w_arr = np.array(widths_sorted)

    ax.plot(
        w_arr,
        means_arr,
        marker=_EXP2_STYLE["marker"],
        color=_EXP2_STYLE["color"],
        linewidth=1.5,
        markersize=6,
    )
    ax.fill_between(
        w_arr,
        np.maximum(means_arr - stds_arr, 0),
        means_arr + stds_arr,
        alpha=0.2,
        color=_EXP2_STYLE["color"],
    )
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)


def plot_exp2(
    results: dict[str, Any],
    checkpoint: str = "final",
    metric: str = "C_far",
    ax: Axes | None = None,
    depth: int | None = None,
    save_path: str | None = None,
) -> Figure:
    """Line plot: bottleneck width vs C_far (or R_far).

    x-axis: bottleneck width (log2 scale)
    y-axis: C_far or R_far (mean +/- std)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    depth_data = _resolve_depth_data(results, depth)
    _plot_exp2_panel(depth_data, checkpoint, metric, ax)

    metric_label = _EXP2_METRIC_LABELS.get(metric, (metric, metric))[0]
    ax.set_xlabel("Bottleneck width")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs bottleneck width ({checkpoint})")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp2_all_checkpoints(
    results: dict[str, Any],
    metric: str = "C_far",
    save_path: str | None = None,
    depth: int | None = None,
) -> Figure:
    """3-panel plot: init / mid / final for one depth."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    depth_data = _resolve_depth_data(results, depth)

    for i, ckpt in enumerate(("init", "mid", "final")):
        _plot_exp2_panel(depth_data, ckpt, metric, axes[i])
        axes[i].set_xlabel("Bottleneck width")
        axes[i].set_title(ckpt)
        if i == 0:
            axes[i].set_ylabel(_EXP2_METRIC_LABELS.get(metric, (metric, metric))[0])

    depth_label = depth if depth else list(results.keys())[0]
    metric_name = _EXP2_METRIC_LABELS.get(metric, (metric, metric))[1]
    fig.suptitle(f"{metric_name}: depth={depth_label}", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp2_depth_sweep(
    results: dict[str, Any],
    metric: str = "C_far",
    save_path: str | None = None,
) -> Figure:
    """Grid: rows = depths, columns = checkpoints (init/mid/final)."""
    depths = sorted(int(d) for d in results.keys())
    checkpoints = ["init", "mid", "final"]

    n_rows = len(depths)
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for r, d in enumerate(depths):
        depth_data = results[str(d)]
        for c, ckpt in enumerate(checkpoints):
            ax = axes[r][c]
            _plot_exp2_panel(depth_data, ckpt, metric, ax)

            if r == 0:
                ax.set_title(ckpt, fontsize=12)
            if c == 0:
                metric_sym = _EXP2_METRIC_LABELS.get(metric, (metric, metric))[0]
                ax.set_ylabel(f"depth={d}\n{metric_sym}", fontsize=11)
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("Bottleneck width")
            else:
                ax.set_xlabel("")

    metric_name = _EXP2_METRIC_LABELS.get(metric, (metric, metric))[1]
    fig.suptitle(f"{metric_name}: depth sweep", fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp2_acc_delta(
    results: dict[str, Any],
    checkpoint: str = "final",
    save_path: str | None = None,
) -> Figure:
    """Per-depth grouped bar chart: relative accuracy & loss change vs bottleneck width.

    For each depth shows Delta_acc = acc(w) - acc(w_max) and
    Delta_loss = loss(w) - loss(w_max). This cleanly isolates the
    bottleneck effect within each depth without mixing depths together.
    """
    depths = sorted(int(d) for d in results.keys())
    _depth_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, depth in enumerate(depths):
        depth_data = results[str(depth)]
        widths = sorted(int(w) for w in depth_data.keys())
        w_max = max(widths)  # control (no bottleneck)

        ctrl_ckpt = depth_data[str(w_max)].get(checkpoint, {})
        acc_ctrl = ctrl_ckpt.get("test_acc_mean", 0.0)
        loss_ctrl = ctrl_ckpt.get("test_loss_mean", 0.0)

        # Exclude control from bars
        bneck_widths = [w for w in widths if w != w_max]
        if not bneck_widths:
            continue

        d_acc = []
        d_loss = []
        labels = []
        for w in bneck_widths:
            ckpt_data = depth_data[str(w)].get(checkpoint, {})
            d_acc.append(ckpt_data.get("test_acc_mean", 0.0) - acc_ctrl)
            d_loss.append(ckpt_data.get("test_loss_mean", 0.0) - loss_ctrl)
            labels.append(str(w))

        color = _depth_palette[idx % len(_depth_palette)]
        n_bars = len(bneck_widths)
        bar_w = 0.8 / max(len(depths), 1)
        x_pos = np.arange(n_bars)
        offset = (idx - (len(depths) - 1) / 2) * bar_w

        ax_acc.bar(
            x_pos + offset,
            d_acc,
            bar_w,
            color=color,
            alpha=0.8,
            label=f"depth={depth}",
        )
        ax_loss.bar(
            x_pos + offset,
            d_loss,
            bar_w,
            color=color,
            alpha=0.8,
            label=f"depth={depth}",
        )

    # Common x-labels: bottleneck widths (from first depth that has them)
    for depth in depths:
        depth_data = results[str(depth)]
        widths = sorted(int(w) for w in depth_data.keys())
        w_max = max(widths)
        bneck_widths = [w for w in widths if w != w_max]
        if bneck_widths:
            x_labels = [str(w) for w in bneck_widths]
            x_pos = np.arange(len(bneck_widths))
            break
    else:
        x_labels, x_pos = [], np.arange(0)

    for ax, ylabel, title in [
        (ax_acc, r"$\Delta$ accuracy", "Accuracy drop from bottleneck"),
        (ax_loss, r"$\Delta$ loss", "Loss increase from bottleneck"),
    ]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Bottleneck width")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({checkpoint})")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Bottleneck impact on generalisation ({checkpoint})",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp2_diff_heatmap(
    results: dict[str, Any],
    depth: int,
    checkpoint: str = "init",
    metric: str = "C",
    save_path: str | None = None,
) -> Figure:
    r"""Difference heatmaps $\Delta M_{ij}$.

    Args:
        metric: ``"C"`` for coupling, ``"D"`` for stable rank.

    Row of panels: one per bottleneck\_width (excluding control).
    Shows the *loss* of connectivity / dimensionality from narrowing.
    """
    matrix_key = f"{metric}_matrix"
    depth_data = results[str(depth)]
    widths = sorted((int(w) for w in depth_data.keys()), reverse=True)
    w_max = widths[0]  # control (no real bottleneck)

    ctrl_matrix = depth_data[str(w_max)].get(checkpoint, {}).get(matrix_key)
    if ctrl_matrix is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(
            0.5,
            0.5,
            f"No {matrix_key} for control",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig
    ctrl_mat = np.array(ctrl_matrix)

    bneck_widths = [w for w in widths if w != w_max]
    if not bneck_widths:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(
            0.5,
            0.5,
            "Only control width present",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    n_cols = len(bneck_widths)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), squeeze=False)

    bneck_pos = depth // 2
    im = None

    # Pre-compute shared vmax across all panels for consistent colorbar
    global_vmax = 0.0
    for w in bneck_widths:
        c_matrix = depth_data[str(w)].get(checkpoint, {}).get(matrix_key)
        if c_matrix is not None:
            diff = ctrl_mat - np.array(c_matrix)
            global_vmax = max(global_vmax, float(np.abs(diff).max()))
    global_vmax = max(global_vmax, 1e-6)

    for c_idx, w in enumerate(bneck_widths):
        ax = axes[0][c_idx]
        ckpt_data = depth_data[str(w)].get(checkpoint, {})
        c_matrix = ckpt_data.get(matrix_key)

        if c_matrix is None:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"bneck={w}")
            continue

        diff = ctrl_mat - np.array(c_matrix)
        im = ax.imshow(
            diff,
            cmap="RdBu_r",
            vmin=-global_vmax,
            vmax=global_vmax,
            origin="upper",
            aspect="equal",
        )

        ax.axhline(y=bneck_pos, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(x=bneck_pos, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

        n = diff.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"{i}" for i in range(n)], fontsize=6)
        ax.set_yticklabels([f"{i}" for i in range(n)], fontsize=6)
        ax.set_title(f"bneck={w}", fontsize=10)
        if c_idx == 0:
            ax.set_ylabel("Layer $i$")
        ax.set_xlabel("Layer $j$")

    if im is not None:
        metric_sym = r"$\mathcal{C}$" if metric == "C" else r"$\mathcal{D}$"
        fig.colorbar(
            im,
            ax=list(axes.ravel()),
            fraction=0.02,
            pad=0.02,
            label=rf"$\Delta{metric_sym[1:-1]}(i,j)$",
        )
    metric_name = "coupling" if metric == "C" else "stable rank"
    fig.suptitle(
        rf"$\Delta {metric}$: {metric_name} loss from bottleneck"
        + f" (depth={depth}, {checkpoint})",
        fontsize=13,
    )
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp2_dual_metric(
    results: dict[str, Any],
    checkpoint: str = "final",
    save_path: str | None = None,
) -> Figure:
    """Dual plot: C_far and R_far vs bottleneck width for each depth.

    Two subplots side by side. Shows that even if C_far is compensated,
    R_far can remain suppressed.
    """
    depths = sorted(int(d) for d in results.keys())

    _depth_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    for idx, depth in enumerate(depths):
        depth_data = results[str(depth)]
        widths = sorted(int(w) for w in depth_data.keys())
        c_means, c_stds, r_means, r_stds = [], [], [], []
        for w in widths:
            ckpt_data = depth_data[str(w)].get(checkpoint, {})
            c_means.append(ckpt_data.get("C_far_mean", 0.0))
            c_stds.append(ckpt_data.get("C_far_std", 0.0))
            r_means.append(ckpt_data.get("R_far_mean", 0.0))
            r_stds.append(ckpt_data.get("R_far_std", 0.0))

        color = _depth_palette[idx % len(_depth_palette)]
        w_arr = np.array(widths)
        c_arr = np.array(c_means)
        c_s = np.array(c_stds)
        r_arr = np.array(r_means)
        r_s = np.array(r_stds)

        ax_c.plot(
            w_arr,
            c_arr,
            marker="D",
            color=color,
            linewidth=1.5,
            markersize=6,
            label=f"depth={depth}",
        )
        ax_c.fill_between(
            w_arr, np.maximum(c_arr - c_s, 0), c_arr + c_s, alpha=0.15, color=color
        )

        ax_r.plot(
            w_arr,
            r_arr,
            marker="D",
            color=color,
            linewidth=1.5,
            markersize=6,
            label=f"depth={depth}",
        )
        ax_r.fill_between(
            w_arr, np.maximum(r_arr - r_s, 0), r_arr + r_s, alpha=0.15, color=color
        )

    for ax, label, name in [
        (ax_c, r"$\bar{\mathcal{C}}_{\mathrm{far}}$", "Far coupling"),
        (ax_r, r"$\bar{\mathcal{R}}_{\mathrm{far}}$", "Far resonance"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Bottleneck width")
        ax.set_ylabel(label)
        ax.set_title(f"{name} ({checkpoint})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Coupling vs Resonance: bottleneck sensitivity" + f" ({checkpoint})",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp2_triple_metric(
    results: dict[str, Any],
    checkpoint: str = "final",
    save_path: str | None = None,
) -> Figure:
    """Triple plot: C_far, R_far, D_far vs bottleneck width for each depth.

    Three subplots side by side. D_far (stable rank) complements C_far and R_far,
    directly reflecting the rank constraint from Corollary S7.
    """
    depths = sorted(int(d) for d in results.keys())
    _depth_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    metrics_spec: list[tuple[str, str, str]] = [
        ("C_far", r"$\bar{\mathcal{C}}_{\mathrm{far}}$", "Far coupling"),
        ("R_far", r"$\bar{\mathcal{R}}_{\mathrm{far}}$", "Far resonance"),
        ("D_far", r"$\bar{\mathcal{D}}_{\mathrm{far}}$", "Far stable rank"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for idx, depth in enumerate(depths):
        depth_data = results[str(depth)]
        widths = sorted(int(w) for w in depth_data.keys())
        color = _depth_palette[idx % len(_depth_palette)]
        w_arr = np.array(widths)

        for col, (mkey, _, _) in enumerate(metrics_spec):
            means = []
            stds = []
            for w in widths:
                ckpt_data = depth_data[str(w)].get(checkpoint, {})
                means.append(ckpt_data.get(f"{mkey}_mean", 0.0))
                stds.append(ckpt_data.get(f"{mkey}_std", 0.0))

            m_arr = np.array(means)
            s_arr = np.array(stds)
            axes[col].plot(
                w_arr,
                m_arr,
                marker="D",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=f"depth={depth}",
            )
            axes[col].fill_between(
                w_arr,
                np.maximum(m_arr - s_arr, 0),
                m_arr + s_arr,
                alpha=0.15,
                color=color,
            )

    for col, (_, label, name) in enumerate(metrics_spec):
        axes[col].set_xscale("log", base=2)
        axes[col].set_xlabel("Bottleneck width")
        axes[col].set_ylabel(label)
        axes[col].set_title(f"{name} ({checkpoint})")
        axes[col].legend(fontsize=8)
        axes[col].grid(True, alpha=0.3)

    fig.suptitle(
        r"$(\mathcal{C},\;\mathcal{R},\;\mathcal{D})$: bottleneck sensitivity"
        + f" ({checkpoint})",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ======================================================================
# Experiment 3: ReLU vs GELU GN-Gap
# ======================================================================

_ACT_STYLES = {
    "relu": {"color": "#d62728", "label": "ReLU", "marker": "o"},
    "leaky_relu": {"color": "#ff7f0e", "label": "LeakyReLU", "marker": "s"},
    "softplus": {"color": "#2ca02c", "label": "Softplus", "marker": "^"},
    "silu": {"color": "#9467bd", "label": "Swish", "marker": "D"},
    "gelu": {"color": "#1f77b4", "label": "GELU", "marker": "v"},
}


def plot_exp3_gn_gap(
    results: dict[str, Any],
    save_path: str | None = None,
) -> Figure:
    """Grouped bar plot: GN-Gap by checkpoint x activation.

    3 groups (init/mid/final) x 2 bars (ReLU, GELU).
    y-axis: aggregated GN-Gap, error bars: +/-std over seeds.
    """
    checkpoints = ["init", "mid", "final"]
    activations = [act for act in _ACT_STYLES if act in results]
    n_act = len(activations)

    x_pos = np.arange(len(checkpoints))
    bar_width = 0.8 / max(n_act, 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    bars_by_act: dict[str, Any] = {}
    for i, act in enumerate(activations):
        means, stds = [], []
        for ckpt in checkpoints:
            ckpt_data = results[act].get(ckpt, {})
            means.append(ckpt_data.get("gap_global_mean", 0.0))
            stds.append(ckpt_data.get("gap_global_std", 0.0))

        style = _ACT_STYLES[act]
        offset = (i - (n_act - 1) / 2) * bar_width
        container = ax.bar(
            x_pos + offset,
            means,
            bar_width,
            yerr=stds,
            label=style["label"],
            color=style["color"],
            alpha=0.8,
            capsize=4,
        )
        bars_by_act[act] = (container, means)

    # Value annotations above bars
    for act, (container, means) in bars_by_act.items():
        for bar, val in zip(container, means, strict=False):
            if val < 1e-6:
                ax.annotate(
                    r"$\approx 0$",
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=_ACT_STYLES[act]["color"],
                )
            else:
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(checkpoints)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(r"$\overline{\mathrm{Gap}}_{GN}$")
    ax.set_title(r"Aggregated GN-Gap by activation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def _plot_exp3_distance_panel(
    results: dict[str, Any],
    checkpoint: str,
    ax: Axes,
) -> None:
    """Single panel GN-Gap(d) for all activations."""
    activations = [act for act in _ACT_STYLES if act in results]

    for act in activations:
        ckpt_data = results[act].get(checkpoint, {})
        dist_data = ckpt_data.get("distances", {})
        distances = sorted(int(d) for d in dist_data.keys())
        if not distances:
            continue

        means = [dist_data[d]["gap_mean"] for d in distances]
        stds = [dist_data[d]["gap_std"] for d in distances]

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        dist_arr = np.array(distances)

        style = _ACT_STYLES[act]
        ax.plot(
            dist_arr,
            means_arr,
            marker="o",
            color=style["color"],
            label=style["label"],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(
            dist_arr,
            np.maximum(means_arr - stds_arr, 0),
            means_arr + stds_arr,
            alpha=0.2,
            color=style["color"],
        )

    ax.grid(True, alpha=0.3)


def plot_exp3_per_distance(
    results: dict[str, Any],
    save_path: str | None = None,
) -> Figure:
    """3-panel: init / mid / final - GN-Gap(d) for ReLU and GELU."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, ckpt in enumerate(("init", "mid", "final")):
        _plot_exp3_distance_panel(results, ckpt, axes[i])
        axes[i].set_xlabel("Layer distance $d$")
        axes[i].set_title(ckpt)
        axes[i].set_ylabel(r"$\mathrm{Gap}_{GN}(d)$")
        if i == 0:
            axes[i].legend()

    fig.suptitle(r"GN-Gap per distance", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp3_composite(
    results: dict[str, Any],
    checkpoint: str = "mid",
    save_path: str | None = None,
) -> Figure:
    """Composite Figure 1c: bar plot (left) + scatter Gap vs E[sigma''(z)^2] (right).

    Left panel: grouped bar - GN-Gap by activation for one checkpoint.
    Right panel: scatter - correlation Gap vs E[sigma''(z)^2] with linear fit.
    """
    from scipy import stats as sp_stats  # noqa: PLC0415

    activations = [act for act in _ACT_STYLES if act in results]

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left panel: bar plot ---
    means, stds, colors, labels = [], [], [], []
    for act in activations:
        ckpt_data = results[act].get(checkpoint, {})
        m = ckpt_data.get("gap_global_mean", 0.0)
        s = ckpt_data.get("gap_global_std", 0.0)
        means.append(m)
        stds.append(s)
        colors.append(_ACT_STYLES[act]["color"])
        labels.append(_ACT_STYLES[act]["label"])

    x_pos = np.arange(len(activations))
    bars = ax_bar.bar(
        x_pos,
        means,
        yerr=stds,
        color=colors,
        alpha=0.8,
        capsize=4,
    )
    for bar, val, act in zip(bars, means, activations, strict=False):
        if val < 1e-6:
            ax_bar.annotate(
                r"$\approx 0$",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_ACT_STYLES[act]["color"],
            )
        else:
            ax_bar.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel(r"$\overline{\mathrm{Gap}}_{GN}$")
    ax_bar.set_title(f"GN-Gap ({checkpoint})")
    ax_bar.grid(True, alpha=0.3, axis="y")

    # --- Right panel: scatter Gap vs E[sigma''(z)^2] ---
    gap_vals, sigma_vals = [], []
    for act in activations:
        ckpt_data = results[act].get(checkpoint, {})
        g = ckpt_data.get("gap_global_mean", 0.0)
        s = ckpt_data.get("sigma_pp_sq_mean", 0.0)
        gap_vals.append(g)
        sigma_vals.append(s)

        ax_scatter.scatter(
            s,
            g,
            color=_ACT_STYLES[act]["color"],
            marker=_ACT_STYLES[act]["marker"],
            s=80,
            zorder=5,
        )
        ax_scatter.annotate(
            _ACT_STYLES[act]["label"],
            xy=(s, g),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    # Linear fit (if >= 3 non-degenerate points)
    sx = np.array(sigma_vals)
    sy = np.array(gap_vals)
    if len(sx) >= 3 and sx.max() > 1e-12:
        slope, intercept, r_value, _, _ = sp_stats.linregress(sx, sy)
        x_fit = np.linspace(0, sx.max() * 1.1, 50)
        ax_scatter.plot(
            x_fit,
            slope * x_fit + intercept,
            "--",
            color="gray",
            alpha=0.7,
            label=f"$R^2 = {r_value**2:.3f}$",
        )
        ax_scatter.legend(fontsize=9)

    ax_scatter.set_xlabel(r"$\mathbb{E}[\sigma''(z)^2]$")
    ax_scatter.set_ylabel(r"$\overline{\mathrm{Gap}}_{GN}$")
    ax_scatter.set_title(f"GN-Gap vs activation curvature ({checkpoint})")
    ax_scatter.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ======================================================================
# Experiment 4: Diamond MLP - tensor term
# ======================================================================

_EXP4_CONFIG_STYLES = {
    "sum_relu": {"color": "#1f77b4", "marker": "o", "label": "sum + ReLU"},
    "sum_silu": {"color": "#ff7f0e", "marker": "s", "label": "sum + SiLU"},
    "cat_relu": {"color": "#2ca02c", "marker": "^", "label": "cat + ReLU"},
    "cat_silu": {"color": "#d62728", "marker": "D", "label": "cat + SiLU"},
}

_EXP4_CKPT_STYLES = {
    "init": {"color": "#1f77b4", "linestyle": "-", "label": "init"},
    "mid": {"color": "#ff7f0e", "linestyle": "--", "label": "mid"},
    "final": {"color": "#d62728", "linestyle": "-.", "label": "final"},
}


def plot_exp4_gn_gap_heatmap(
    results: dict[str, Any],
    checkpoint: str = "final",
    save_path: str | None = None,
) -> Figure:
    """GN-Gap heatmap: merge_type x activation, one panel per depth.

    Args:
        results: Output of Exp4Runner.run() - {depth: {config_key: {ckpt: ...}}}.
    """
    depths = sorted(int(d) for d in results.keys())
    n_depths = len(depths)

    fig, axes = plt.subplots(1, n_depths, figsize=(5 * n_depths, 4), squeeze=False)

    merge_types = ["sum", "cat"]
    activations = ["relu", "silu"]

    for idx, depth in enumerate(depths):
        ax = axes[0][idx]
        depth_data = results[str(depth)]

        matrix = np.zeros((len(merge_types), len(activations)))
        annot = np.empty_like(matrix, dtype=object)

        for r, merge in enumerate(merge_types):
            for c, act in enumerate(activations):
                key = f"{merge}_{act}"
                ckpt_data = depth_data.get(key, {}).get(checkpoint, {})
                gap = ckpt_data.get("merge_gn_gap", {}).get("gap_mean", 0.0)
                gap_std = ckpt_data.get("merge_gn_gap", {}).get("gap_std", 0.0)
                matrix[r, c] = gap
                annot[r, c] = f"{gap:.2f}\n+/-{gap_std:.2f}"

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(activations)
        ax.set_yticks(range(len(merge_types)))
        ax.set_yticklabels(merge_types)
        ax.set_title(f"depth={depth}")

        for r in range(len(merge_types)):
            for c in range(len(activations)):
                ax.text(c, r, annot[r, c], ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        rf"$\mathrm{{Gap}}_{{GN}}$ on merge node ({checkpoint})",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp4_cross_branch_decay(
    results: dict[str, Any],
    save_path: str | None = None,
) -> Figure:
    """R_bar_{AB}(d_graph) for each depth: panels = configs, curves = checkpoints.

    Visually shows decay at init and its breakdown after training.
    """
    depths = sorted(int(d) for d in results.keys())
    configs = list(_EXP4_CONFIG_STYLES.keys())
    checkpoints = ["init", "mid", "final"]

    n_rows = len(depths)
    n_cols = len(configs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for r, depth in enumerate(depths):
        depth_data = results[str(depth)]
        for c, cfg in enumerate(configs):
            ax = axes[r][c]
            cfg_data = depth_data.get(cfg, {})

            for ckpt in checkpoints:
                ckpt_data = cfg_data.get(ckpt, {})
                cross = ckpt_data.get("cross_AB", {})
                if not cross:
                    continue

                dists = sorted(int(d) for d in cross.keys())
                means = [cross[d]["R_mean"] for d in dists]
                stds = [cross[d].get("R_std", 0.0) for d in dists]

                means_arr = np.array(means)
                stds_arr = np.array(stds)
                dist_arr = np.array(dists)

                style = _EXP4_CKPT_STYLES[ckpt]
                ax.plot(
                    dist_arr,
                    means_arr,
                    marker="o",
                    color=style["color"],
                    linestyle=style["linestyle"],
                    label=style["label"],
                    linewidth=1.5,
                    markersize=5,
                )
                ax.fill_between(
                    dist_arr,
                    np.maximum(means_arr - stds_arr, 1e-15),
                    means_arr + stds_arr,
                    alpha=0.15,
                    color=style["color"],
                )

            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(_EXP4_CONFIG_STYLES[cfg]["label"], fontsize=11)
            if c == 0:
                ax.set_ylabel(rf"depth={depth}" + "\n" + r"$\bar{\mathcal{R}}_{AB}$")
            if r == n_rows - 1:
                ax.set_xlabel(r"Graph distance $d_{\mathrm{graph}}$")
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=8)

    fig.suptitle(
        r"Cross-branch resonance $\bar{\mathcal{R}}_{AB}(d_{\mathrm{graph}})$",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp4_rho_summary(
    results: dict[str, Any],
    save_path: str | None = None,
) -> Figure:
    r"""rho_max by config x checkpoint for each depth.

    One panel per depth; curves are configurations (merge x activation).
    Horizontal line rho=1 marks the decay threshold.
    """
    depths = sorted(int(d) for d in results.keys())
    checkpoints = ["init", "mid", "final"]
    n_depths = len(depths)

    fig, axes = plt.subplots(1, n_depths, figsize=(5 * n_depths, 4), squeeze=False)

    for idx, depth in enumerate(depths):
        ax = axes[0][idx]
        depth_data = results[str(depth)]

        for cfg, style in _EXP4_CONFIG_STYLES.items():
            cfg_data = depth_data.get(cfg, {})
            rho_vals: list[float] = []
            for ckpt in checkpoints:
                rho_info = cfg_data.get(ckpt, {}).get("rho", {})
                rho_vals.append(rho_info.get("rho_max_mean", 0.0))

            ax.plot(
                checkpoints,
                rho_vals,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=1.5,
            )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label=r"$\rho=1$")
        ax.set_title(f"depth={depth}")
        ax.set_ylabel(r"$\rho_{\max}$")
        ax.set_xlabel("Checkpoint")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Jacobian spectral norm $\rho_{\max}$ for Diamond MLP",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ======================================================================
# Experiment 6: ResNet-18 conv
# ======================================================================

_EXP6_ARCH_STYLES = {
    "resnet18": {"color": "#1f77b4", "marker": "s", "label": "ResNet-18"},
    "plain_resnet18": {"color": "#d62728", "marker": "o", "label": "Plain ResNet-18"},
}


def plot_exp6_decay(
    results: dict[str, Any],
    activation: str,
    checkpoint: str = "final",
    metric: str = "R",
    save_path: str | None = None,
) -> Figure:
    """R_bar(d) or C_bar(d) for ResNet-18 vs PlainResNet-18 for a given activation."""
    act_data = results[activation]

    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, style in _EXP6_ARCH_STYLES.items():
        ckpt_data = act_data.get(arch, {}).get(checkpoint, {})
        dist_data = ckpt_data.get("distances", {})
        distances = sorted(int(d) for d in dist_data.keys())
        if not distances:
            continue
        means = [dist_data[d][f"{metric}_mean"] for d in distances]
        stds = [dist_data[d][f"{metric}_std"] for d in distances]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        dist_arr = np.array(distances)

        ax.plot(
            dist_arr,
            means_arr,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(
            dist_arr,
            np.maximum(means_arr - stds_arr, 1e-15 if metric != "D" else 0),
            means_arr + stds_arr,
            alpha=0.2,
            color=style["color"],
        )

    if metric != "D":
        ax.set_yscale("log")
    ax.set_xlabel("Distance $d$")
    metric_labels = {"R": r"$\bar{R}(d)$", "C": r"$\bar{C}(d)$", "D": r"$\bar{D}(d)$"}
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f"{metric} decay - {activation.upper()}, {checkpoint}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp6_all_checkpoints(
    results: dict[str, Any],
    activation: str,
    metric: str = "R",
    save_path: str | None = None,
) -> Figure:
    """3 panels (init/mid/final) for one activation with R_bar(d) or C_bar(d)."""
    act_data = results[activation]
    checkpoints = ["init", "mid", "final"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, ckpt in enumerate(checkpoints):
        ax = axes[idx]
        for arch, style in _EXP6_ARCH_STYLES.items():
            ckpt_data = act_data.get(arch, {}).get(ckpt, {})
            dist_data = ckpt_data.get("distances", {})
            distances = sorted(int(d) for d in dist_data.keys())
            if not distances:
                continue
            means = [dist_data[d][f"{metric}_mean"] for d in distances]
            stds = [dist_data[d][f"{metric}_std"] for d in distances]
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            dist_arr = np.array(distances)

            ax.plot(
                dist_arr,
                means_arr,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=1.5,
                markersize=5,
            )
            ax.fill_between(
                dist_arr,
                np.maximum(means_arr - stds_arr, 1e-15 if metric != "D" else 0),
                means_arr + stds_arr,
                alpha=0.2,
                color=style["color"],
            )

        if metric != "D":
            ax.set_yscale("log")
        ax.set_xlabel("Distance $d$")
        ax.set_title(ckpt)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            metric_labels = {
                "R": r"$\bar{R}(d)$",
                "C": r"$\bar{C}(d)$",
                "D": r"$\bar{D}(d)$",
            }
            ax.set_ylabel(metric_labels.get(metric, metric))
        if idx == 2:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Exp 6: {metric} decay - {activation.upper()}, ResNet-18 vs Plain",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp6_gn_gap(
    results: dict[str, Any],
    checkpoint: str = "final",
    save_path: str | None = None,
) -> Figure:
    """Bar chart: global GN-Gap for each (activation x arch)."""
    activations = list(results.keys())
    archs = list(_EXP6_ARCH_STYLES.keys())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(activations))
    width = 0.35

    for i, arch in enumerate(archs):
        style = _EXP6_ARCH_STYLES[arch]
        means, stds = [], []
        for act in activations:
            ckpt = results[act].get(arch, {}).get(checkpoint, {})
            means.append(ckpt.get("gn_gap_global_mean", 0.0))
            stds.append(ckpt.get("gn_gap_global_std", 0.0))
        ax.bar(
            x + i * width - width / 2,
            means,
            width,
            yerr=stds,
            color=style["color"],
            label=style["label"],
            alpha=0.85,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.set_ylabel(r"Global GN-Gap $\|\|H^T\|\|_F / \|\|H^{GN}\|\|_F$")
    ax.set_title(f"Exp 6: GN-Gap - {checkpoint}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_exp6_gn_gap_per_distance(
    results: dict[str, Any],
    checkpoint: str = "final",
    save_path: str | None = None,
) -> Figure:
    """GN-Gap(d) per distance for each activation - ResNet vs Plain."""
    activations = list(results.keys())
    n_act = len(activations)

    fig, axes = plt.subplots(1, n_act, figsize=(6 * n_act, 4), squeeze=False)
    for idx, act in enumerate(activations):
        ax = axes[0][idx]
        act_data = results[act]
        for arch, style in _EXP6_ARCH_STYLES.items():
            ckpt_data = act_data.get(arch, {}).get(checkpoint, {})
            gn_dists = ckpt_data.get("gn_gap_distances", {})
            distances = sorted(int(d) for d in gn_dists.keys())
            if not distances:
                continue
            gaps = [gn_dists[d].get("gap_mean", 0.0) for d in distances]
            ax.plot(
                distances,
                gaps,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=1.5,
            )
        ax.set_xlabel("Distance $d$")
        ax.set_ylabel("GN-Gap$(d)$")
        ax.set_title(f"{act.upper()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Exp 6: GN-Gap per distance - {checkpoint}", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
