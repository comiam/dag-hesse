"""TikZ figures for Experiment 2: bottleneck ablation.

Generated figures
-----------------
- fig:dfar-bottleneck - D_far vs d_u (L=6 and L=8)
"""

from __future__ import annotations

from typing import Any

from ..formatting import round_str


def fig_dfar_bottleneck(exp2: dict[str, Any]) -> str:
    """Generate Figure 3: D_far vs bottleneck width."""
    widths = [4, 8, 16, 32, 64, 128, 256, 512]
    series: list[tuple[str, str, str, str]] = [
        ("6", "init", "blue, thick, mark=*, mark size=2", "$L\\!=\\!6$, init"),
        (
            "6",
            "final",
            "red, thick, dashed, mark=square*, mark size=2",
            "$L\\!=\\!6$, final",
        ),
        (
            "8",
            "init",
            "blue!50, thick, mark=triangle*, mark size=2",
            "$L\\!=\\!8$, init",
        ),
        (
            "8",
            "final",
            "red!50, thick, dashed, mark=triangle*, mark size=2",
            "$L\\!=\\!8$, final",
        ),
    ]

    blocks: list[str] = []
    for depth, phase, style, legend in series:
        pts: list[str] = []
        for w in widths:
            if str(w) in exp2.get(depth, {}):
                val = exp2[depth][str(w)][phase]["D_far_mean"]
                pts.append(f"({w},{round_str(val, 2)})")
        if pts:
            coords_str = " ".join(pts)
            blocks.append(
                f"      \\addplot[{style}] coordinates {{\n"
                f"        {coords_str}\n"
                f"      }}; \\addlegendentry{{{legend}}}"
            )

    # Theoretical bound line
    blocks.append(
        "      \\addplot[black, dotted, thick, domain=4:512, samples=50]\n"
        "      {min(x, 99)};\n"
        "      \\addlegendentry{$\\min(d_u,K{-}1)$}"
    )

    plots = "\n".join(blocks)
    return rf"""\begin{{figure}}[htbp]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
        width=0.95\columnwidth,
        height=5cm,
        xlabel={{Bottleneck width $d_u$}},
        ylabel={{$\mathcal{{D}}_{{\mathrm{{far}}}}$}},
        xmode=log,
        log basis x=2,
        xtick={{4,8,16,32,64,128,256,512}},
        xticklabels={{4,8,16,32,64,128,256,512}},
        grid=major,
        grid style={{gray!30}},
        legend style={{font=\footnotesize, at={{(0.03,0.97)}}, anchor=north west}},
        tick label style={{font=\footnotesize}},
        label style={{font=\small}},
      ]
{plots}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Exp.\,2: Stable rank $\mathcal{{D}}_{{\mathrm{{far}}}}$
    vs.\ bottleneck width~$d_u$ (CIFAR-100).
    Solid/dashed: init/final;
    bright: $L\!=\!6$, faded: $L\!=\!8$.
    Dotted: theoretical bound $\min(d_u,K{{-}}1)$.
    $d_u\!=\!512$ (${{}}^\dagger$, control:
    $d_{{\mathrm{{base}}}}\!=\!512$, no bottleneck):
  Table~\ref{{tab:exp2}}.}}
  \label{{fig:dfar-bottleneck}}
\end{{figure}}"""
