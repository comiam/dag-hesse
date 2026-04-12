"""TikZ figures for Experiment 4: Diamond MLP.

Generated figures
-----------------
- fig:diamond-gap    - bar chart of GN-Gap (k=2)
- fig:cross-branch-R - cross-branch R_AB vs d_graph
"""

from __future__ import annotations

from typing import Any

from ..formatting import tikz_coord

_CONFIGS = ["sum_relu", "sum_silu", "cat_relu", "cat_silu"]
_CFG_COMBINED = {
    "sum_relu": "sum+ReLU",
    "sum_silu": "sum+SiLU",
    "cat_relu": "cat+ReLU",
    "cat_silu": "cat+SiLU",
}


# ---------------------------------------------------------------------------
# fig:diamond-gap
# ---------------------------------------------------------------------------


def fig_diamond_gap(exp4: dict[str, Any]) -> str:
    """Generate Figure 5: bar chart of GN-Gap at merge (k=2)."""
    k = "2"
    init_pts: list[str] = []
    final_pts: list[str] = []
    for cfg in _CONFIGS:
        sym = _CFG_COMBINED[cfg]
        gi = exp4[k][cfg]["init"]["merge_gn_gap"]["gap_mean"]
        gf = exp4[k][cfg]["final"]["merge_gn_gap"]["gap_mean"]
        init_pts.append(f"({sym}, {gi:.2e})")
        final_pts.append(f"({sym}, {gf:.2e})")

    init_str = " ".join(init_pts)
    final_str = " ".join(final_pts)

    return rf"""\begin{{figure}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
        width=0.95\columnwidth,
        height=5.5cm,
        ybar=3pt,
        bar width=7pt,
        ymode=log,
        ylabel={{$\mathrm{{Gap}}_{{GN}}$}},
        symbolic x coords={{sum+ReLU, sum+SiLU, cat+ReLU, cat+SiLU}},
        xtick=data,
        x tick label style={{font=\footnotesize, rotate=20, anchor=east}},
        tick label style={{font=\footnotesize}},
        label style={{font=\small}},
        grid=major,
        grid style={{gray!30}},
        ymin=1e-9, ymax=1e1,
        legend style={{font=\footnotesize, at={{(0.03,0.97)}}, anchor=north west}},
        enlarge x limits=0.2,
      ]
      \addplot[fill=blue!60, draw=blue!80] coordinates {{
        {init_str}
      }}; \addlegendentry{{init}}
      \addplot[fill=red!50, draw=red!70] coordinates {{
        {final_str}
      }}; \addlegendentry{{final}}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Exp.\,4: GN-Gap at the merge node of Diamond~MLP ($k\!=\!2$).
    Logarithmic scale highlights the ${{\sim}}7$~order-of-magnitude
    separation between cat$\,{{+}}\,$SiLU
    ($\sigma''\!\neq\!0$, nonlinear merge) and the remaining
    configurations (linear merge or $\sigma''\!=\!0$ a.e.).
    Sum-merge yields $T\!=\!0$ by construction ---
  $\mathrm{{Gap}}\!\approx\!0$ regardless of~$\sigma$.}}
  \label{{fig:diamond-gap}}
\end{{figure}}"""


# ---------------------------------------------------------------------------
# fig:cross-branch-R
# ---------------------------------------------------------------------------


def fig_cross_branch_R(exp4: dict[str, Any]) -> str:
    """Generate appendix Figure: cross-branch R_AB vs d_graph."""
    k = "2"
    color_map = {
        "sum_relu": "blue",
        "cat_relu": "red",
        "sum_silu": "blue!60!green",
        "cat_silu": "orange",
    }
    mark_init = {
        "sum_relu": "triangle*",
        "cat_relu": "square*",
        "sum_silu": "diamond*",
        "cat_silu": "pentagon*",
    }
    mark_final = {
        "sum_relu": "triangle",
        "cat_relu": "square",
        "sum_silu": "diamond",
        "cat_silu": "pentagon",
    }

    blocks: list[str] = []
    # Init curves (solid)
    for cfg in _CONFIGS:
        color = color_map[cfg]
        mark = mark_init[cfg]
        cab = exp4[k][cfg]["init"]["cross_AB"]
        pts: list[str] = []
        for d in sorted(cab.keys(), key=int):
            pts.append(tikz_coord(int(d), cab[d]["R_mean"], cab[d]["R_std"]))
        coords = "\n        ".join(pts)
        label = _CFG_COMBINED[cfg] + " init"
        blocks.append(
            f"      \\addplot[{color}, thick, mark={mark}, mark size=2,\n"
            f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
            f"        {coords}\n"
            f"      }}; \\addlegendentry{{{label}}}"
        )

    # Final curves (dashed)
    for cfg in _CONFIGS:
        color = color_map[cfg]
        mark = mark_final[cfg]
        cab = exp4[k][cfg]["final"]["cross_AB"]
        pts_final: list[str] = []
        for d in sorted(cab.keys(), key=int):
            pts_final.append(tikz_coord(int(d), cab[d]["R_mean"], cab[d]["R_std"]))
        coords = "\n        ".join(pts_final)
        label = _CFG_COMBINED[cfg] + " final"
        blocks.append(
            f"      \\addplot[{color}, thick, dashed, mark={mark},\n"
            f"        mark size=2,\n"
            f"      error bars/.cd, y dir=both, y explicit] coordinates {{\n"
            f"        {coords}\n"
            f"      }}; \\addlegendentry{{{label}}}"
        )

    plots = "\n".join(blocks)
    return rf"""\begin{{figure}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
        width=0.95\columnwidth,
        height=5.5cm,
        xlabel={{Graph distance $d_{{\mathrm{{graph}}}}$}},
        ylabel={{$\bar{{\mathcal{{R}}}}_{{AB}}(d_{{\mathrm{{graph}}}})$}},
        ymode=log,
        grid=major,
        grid style={{gray!30}},
        xtick={{2,3,4}},
        tick label style={{font=\footnotesize}},
        label style={{font=\small}},
        legend style={{font=\tiny, at={{(0.5,1.02)}},
        anchor=south, legend columns=4}},
      ]
{plots}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Exp.~4: cross-branch resonance
    $\bar{{\mathcal{{R}}}}_{{AB}}(d_{{\mathrm{{graph}}}})$ in
    Diamond~MLP ($k\!=\!2$).
    Solid curves --- initialization (decay with distance,
    H4.2a);
    dashed --- after training (growth, H4.2b).
    All 4 configurations show $R$ growth with
    $d_{{\mathrm{{graph}}}}$ at final
    ($3.5$--$5.7\times$ from $d\!=\!2$ to $d\!=\!4$).
  Error bands: $\pm 1\sigma$ over 5~seeds.}}
  \label{{fig:cross-branch-R}}
\end{{figure}}"""
