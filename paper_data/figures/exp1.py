"""TikZ figures for Experiment 1 / 1b.

Generated figures
-----------------
- fig:coupling-C      - C(d) groupplot L=8, L=12
- fig:decay-R         - R(d) groupplot L=8, L=12
- fig:decay-R-sn-supp - R(d) spectral norm L=8, 12, 16
"""

from __future__ import annotations

from typing import Any

from ..formatting import tikz_coord

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coords_block(
    data: dict[str, Any],
    depth: int,
    archs: list[tuple[str, str]],
    phases: list[tuple[str, str]],
    metric: str,
) -> list[str]:
    """Build addplot blocks for a given depth.

    Parameters
    ----------
    archs  : [(arch_key, legend_label), ...]
    phases : [(phase_key, style_suffix), ...]
    metric : key in distances dict (R_mean, C_mean, ...)
    """
    d_str = str(depth)
    n_d = depth  # max distance = depth - 1 => indices 0..depth-1
    std_key = metric.replace("_mean", "_std")
    blocks: list[str] = []
    for arch_key, arch_legend in archs:
        for phase_key, style in phases:
            entry = data[d_str][arch_key][phase_key]
            pts: list[str] = []
            for d in range(n_d):
                dist = entry["distances"][str(d)]
                pts.append(tikz_coord(d, dist[metric], dist.get(std_key)))
            coords_str = "\n        ".join(pts)
            legend = f"{arch_legend} {phase_key}"
            blocks.append(
                f"      \\addplot[{style},"
                f"\n      error bars/.cd, y dir=both, y explicit]"
                f" coordinates {{\n        {coords_str}\n"
                f"      }}; \\addlegendentry{{{legend}}}"
            )
    return blocks


# ---------------------------------------------------------------------------
# fig:coupling-C
# ---------------------------------------------------------------------------


def fig_coupling_C(exp1: dict[str, Any]) -> str:
    """Generate Figure 1: coupling C(d) for L=8,12."""
    styles = {
        ("plain", "init"): "blue, thick, mark=triangle*, mark size=2",
        ("plain", "final"): "blue, thick, dashed, mark=triangle, mark size=2",
        ("residual", "init"): "red, thick, mark=square*, mark size=2",
        ("residual", "final"): "red, thick, dashed, mark=square, mark size=2",
    }

    panels: list[str] = []
    for depth in [8, 12]:
        d_str = str(depth)
        blocks: list[str] = []
        for arch, arch_label in [("plain", "Plain"), ("residual", r"Res.\\")]:
            for phase in ["init", "final"]:
                entry = exp1[d_str][arch][phase]
                pts: list[str] = []
                for d in range(depth):
                    dist = entry["distances"][str(d)]
                    pts.append(tikz_coord(d, dist["C_mean"], dist["C_std"]))
                coords_str = "\n        ".join(pts)
                style = styles[(arch, phase)]
                legend_label = f"{arch_label.rstrip(chr(92))} {phase}"
                blocks.append(
                    f"      \\addplot[{style},\n"
                    f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
                    f"        {coords_str}\n"
                    f"      }}; \\addlegendentry{{{legend_label}}}"
                )
        panel = f"      \\nextgroupplot[title={{$L={depth}$}}]\n" + "\n".join(blocks)
        panels.append(panel)

    plots = "\n".join(panels)
    return rf"""\begin{{figure}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{groupplot}}[
        group style={{
          group size=1 by 2,
          vertical sep=1.5cm,
        }},
        width=0.95\columnwidth,
        height=5.5cm,
        xlabel={{Distance $d$}},
        ylabel={{$\bar{{\mathcal{{C}}}}(d)$}},
        ymin=0, ymax=1.05,
        grid=major,
        legend style={{font=\footnotesize, at={{(0.02,0.02)}},
        anchor=south west, legend columns=2}},
      ]
{plots}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Exp.~1: Geometric coupling
    $\bar{{\mathcal{{C}}}}(d)$ vs.\ distance~$d$
    (error bars: $\pm 1\sigma$ over 5~seeds).
    \textbf{{(a)}}~$L\!=\!8$, \textbf{{(b)}}~$L\!=\!12$.
    Plain~MLP: $\mathcal{{C}}$ decays monotonically from~1 to
    0.24 ($L\!=\!12$, init), reflecting loss of geometric
    coherence between distant layers.
    Residual~MLP: $\mathcal{{C}}\!>\!0.93$ at all distances;
    skip connections preserve coupling.
    After training both architectures shift upward (curvature
  becomes more uniform).}}
  \label{{fig:coupling-C}}
\end{{figure}}"""


# ---------------------------------------------------------------------------
# fig:decay-R
# ---------------------------------------------------------------------------


def fig_decay_R(exp1: dict[str, Any]) -> str:
    """Generate Figure 2: R(d) groupplot L=8,12."""
    styles = {
        ("plain", "init"): "blue, thick, mark=triangle*, mark size=2",
        ("plain", "final"): "blue, thick, dashed, mark=triangle, mark size=2",
        ("residual", "init"): "red, thick, mark=square*, mark size=2",
        ("residual", "final"): "red, thick, dashed, mark=square, mark size=2",
    }
    legend_labels = {
        ("plain", "init"): "Plain init",
        ("plain", "final"): "Plain final",
        ("residual", "init"): r"Res.\ init",
        ("residual", "final"): r"Res.\ final",
    }

    panels: list[str] = []
    for depth in [8, 12]:
        d_str = str(depth)
        xtick = ",".join(str(i) for i in range(0, depth, max(1, depth // 6)))
        blocks: list[str] = []
        for arch in ["plain", "residual"]:
            for phase in ["init", "final"]:
                entry = exp1[d_str][arch][phase]
                pts: list[str] = []
                for d in range(depth):
                    dist = entry["distances"][str(d)]
                    pts.append(tikz_coord(d, dist["R_mean"], dist["R_std"]))
                coords_str = "\n        ".join(pts)
                style = styles[(arch, phase)]
                label = legend_labels[(arch, phase)]
                blocks.append(
                    f"      \\addplot[{style},\n"
                    f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
                    f"        {coords_str}\n"
                    f"      }}; \\addlegendentry{{{label}}}"
                )
        panel = (
            f"      \\nextgroupplot[title={{$L={depth}$}}, "
            f"xtick={{{xtick}}}]\n" + "\n".join(blocks)
        )
        panels.append(panel)

    plots = "\n".join(panels)
    return rf"""\begin{{figure*}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{groupplot}}[
        group style={{
          group size=2 by 1,
          horizontal sep=1.8cm,
        }},
        width=0.48\textwidth,
        height=5.5cm,
        xlabel={{Distance $d$}},
        ylabel={{$\bar{{\mathcal{{R}}}}(d)$}},
        ymode=log,
        grid=major,
        grid style={{gray!30}},
        legend style={{font=\footnotesize, at={{(1.12,1.18)}}, anchor=south, legend columns=4}},
        tick label style={{font=\footnotesize}},
        label style={{font=\small}},
      ]
{plots}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Exp.\,1: Mean resonance $\bar{{\mathcal{{R}}}}(d)$ vs.\
    distance~$d$ between layers (log scale on~$y$; error
    bars~$\pm 1\sigma$ over 5~seeds).
    \textbf{{(a)}}~$L\!=\!8$, \textbf{{(b)}}~$L\!=\!12$.
    Plain~MLP exhibits exponential decay (straight lines on
    log scale, $R^2\!>\!0.91$); Residual~MLP shows stabilization
    ($b\!\approx\!0.02$).
    Scale differs: init $\sim 10^{{-1}}$, final $\sim 10^{{0}}$
  (reflects overall curvature growth during training).}}
  \label{{fig:decay-R}}
\end{{figure*}}"""


# ---------------------------------------------------------------------------
# fig:decay-R-sn-supp (spectral normalization)
# ---------------------------------------------------------------------------


def fig_decay_R_sn(exp1b: dict[str, Any]) -> str:
    """Generate appendix Figure: R(d) under spectral norm."""
    color_map = {8: "blue", 12: "red", 16: "green!50!black"}
    mark_init = {8: "*", 12: "square*", 16: "triangle*"}
    mark_final = {8: "o", 12: "square", 16: ""}  # no L=16 final by default

    blocks: list[str] = []
    for depth in [8, 12, 16]:
        d_str = str(depth)
        color = color_map[depth]
        # --- init ---
        if d_str in exp1b and "plain" in exp1b[d_str]:
            entry = exp1b[d_str]["plain"]["init"]
            pts: list[str] = []
            max_d = depth
            for d in range(max_d):
                if str(d) in entry.get("distances", {}):
                    dist = entry["distances"][str(d)]
                    pts.append(tikz_coord(d, dist["R_mean"], dist["R_std"]))
            if pts:
                coords_str = "\n        ".join(pts)
                mark = mark_init[depth]
                blocks.append(
                    f"      \\addplot[{color}, thick, mark={mark}, mark size=1.5,\n"
                    f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
                    f"        {coords_str}\n"
                    f"      }}; \\addlegendentry{{$L\\!=\\!{depth}$ init}}"
                )
            # --- final ---
            entry_f = exp1b[d_str]["plain"].get("final")
            if entry_f and "distances" in entry_f:
                pts_f: list[str] = []
                for d in range(max_d):
                    if str(d) in entry_f["distances"]:
                        dist = entry_f["distances"][str(d)]
                        pts_f.append(tikz_coord(d, dist["R_mean"], dist["R_std"]))
                if pts_f:
                    coords_str_f = "\n        ".join(pts_f)
                    mark_f = mark_final.get(depth, "o")
                    blocks.append(
                        f"      \\addplot[{color}, thick, dashed, mark={mark_f}, mark size=1.5,\n"
                        f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
                        f"        {coords_str_f}\n"
                        f"      }}; \\addlegendentry{{$L\\!=\\!{depth}$ final}}"
                    )

    plots = "\n".join(blocks)
    return rf"""\begin{{figure}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
        width=0.95\columnwidth,
        height=5.5cm,
        xlabel={{Distance $d$}},
        ylabel={{$\bar{{\mathcal{{R}}}}(d)$}},
        ymode=log,
        grid=major,
        grid style={{gray!30}},
        legend style={{font=\scriptsize, at={{(0.97,0.97)}},
        anchor=north east}},
        tick label style={{font=\footnotesize}},
        label style={{font=\small}},
        xmin=-0.5, xmax=16,
      ]
{plots}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Exp.~1b: resonance $\bar{{\mathcal{{R}}}}(d)$ under
    spectral normalization (Plain~SN).
    Solid curves --- initialization
    ($\rho\!\approx\!0.91\!<\!1$): exponential decay over
    ${{\sim}}5.5$ orders of magnitude at $L\!=\!16$, directly
    verifying inequality~\eqref{{eq:resonance-decay-main}}.
    Dashed --- after training:
    at $L\!=\!8$ ($\rho\!\to\!1.0$) $R$ plateaus at
    ${{\approx}}0.94$ (boundary of the theorem condition);
    at $L\!=\!12$ decay persists ($\lambda_1\!<\!0$).
  Error bands: $\pm 1\sigma$ over 5~seeds.}}
  \label{{fig:decay-R-sn-supp}}
\end{{figure}}"""
