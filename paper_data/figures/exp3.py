"""TikZ figures for Experiment 3: GN-Gap vs activation curvature.

Generated figures
-----------------
- fig:gn-gap-activation - scatter Gap vs sigma''^2
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ..formatting import round_str

_ACTIVATIONS = ["relu", "leaky_relu", "softplus", "silu", "gelu"]
_ACT_LABELS = {
    "relu": "ReLU",
    "leaky_relu": "LeakyReLU",
    "softplus": "Softplus",
    "silu": "SiLU",
    "gelu": "GELU",
}
# Plot label positions (manual, matches the paper layout).
_LABEL_OFFSETS = {
    "relu": (0.01, 0.003),
    "leaky_relu": None,  # omitted to avoid clutter (overlaps with ReLU)
    "softplus": (0.065, 0.115),
    "silu": (0.255, 0.218),
    "gelu": (0.50, 0.365),
}


def fig_gn_gap_activation(exp3: dict[str, Any]) -> str:
    """Generate Figure 4: GN-Gap (init) scatter vs sigma''^2."""
    # Collect scatter data
    xs: list[float] = []
    ys: list[float] = []
    pts: list[str] = []
    for act in _ACTIVATIONS:
        sigma = exp3[act]["init"]["sigma_pp_sq_mean"]
        gap = exp3[act]["init"]["gap_global_mean"]
        xs.append(sigma)
        ys.append(gap)
        pts.append(f"        ({round_str(sigma, 3)}, {gap:.4e})")
    scatter = "\n".join(pts)

    # Compute linear fit and Spearman correlation from data
    x_arr, y_arr = np.asarray(xs), np.asarray(ys)
    slope, intercept, r_value, _, _ = sp_stats.linregress(x_arr, y_arr)
    r_squared = r_value**2
    rho_s, p_spearman = sp_stats.spearmanr(x_arr, y_arr)

    # Format p-value as standard significance threshold
    if p_spearman < 0.001:
        p_str = "0.001"
    elif p_spearman < 0.01:
        p_str = "0.01"
    elif p_spearman < 0.05:
        p_str = "0.05"
    else:
        p_str = f"{p_spearman:.2f}"

    # Build labels
    labels: list[str] = []
    for act in _ACTIVATIONS:
        offset = _LABEL_OFFSETS.get(act)
        if offset is None:
            continue
        labels.append(
            f"      \\node[font=\\scriptsize, anchor=south west] "
            f"at (axis cs:{offset[0]},{offset[1]}) {{{_ACT_LABELS[act]}}};"
        )
    label_str = "\n".join(labels)

    return rf"""\begin{{figure}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
        width=0.95\columnwidth,
        height=5cm,
        xlabel={{$\mathbb{{E}}[\sigma''(z)^2]$}},
        ylabel={{$\mathrm{{Gap}}_{{GN}}$ (init)}},
        grid=major,
        grid style={{gray!30}},
        tick label style={{font=\footnotesize}},
        label style={{font=\small}},
        legend style={{font=\footnotesize, at={{(0.03,0.97)}}, anchor=north west}},
        xmin=-0.05, xmax=0.7,
        ymin=-0.03, ymax=0.42,
      ]
      % Scatter: 5 activations
      \addplot[only marks, mark=*, mark size=3, blue] coordinates {{
{scatter}
      }};
      % Linear fit: computed from data
      \addplot[red, thick, dashed, domain=0:0.68, samples=2]
      {{{slope:.4f}*x + {intercept:.4f}}};
      \addlegendentry{{$R^2\!=\!{r_squared:.3f}$, $\rho_s\!=\!{rho_s:.2f}$}}
      % Labels
{label_str}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Exp.\,3: GN-Gap (init) vs.\
    $\mathbb{{E}}[\sigma''(z)^2]$ for 5~activations
    (isolation protocol, LayerNorm off).
    Linear fit $R^2\!=\!{r_squared:.3f}$; Spearman rank correlation
    $\\rho_s\!=\!{rho_s:.2f}$ ($p\!<\!{p_str}$).}}
  \label{{fig:gn-gap-activation}}
\end{{figure}}"""
