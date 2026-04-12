"""TikZ figures for Experiment 6: ResNet-18 R(d).

Generated figures
-----------------
- fig:exp6-R - R(d) groupplot (ReLU / SiLU panels)
"""

from __future__ import annotations

from typing import Any

from ..formatting import tikz_coord

_ARCHS = ["resnet18", "plain_resnet18"]
_ARCH_LEGEND = {
    ("resnet18", "init"): "ResNet init",
    ("resnet18", "final"): "ResNet final",
    ("plain_resnet18", "init"): "Plain init",
    ("plain_resnet18", "final"): "Plain final",
}
_STYLES = {
    ("resnet18", "init"): "red, thick, mark=square*, mark size=2",
    ("resnet18", "final"): "red, thick, dashed, mark=square, mark size=2",
    ("plain_resnet18", "init"): "blue, thick, mark=triangle*, mark size=2",
    ("plain_resnet18", "final"): "blue, thick, dashed, mark=triangle, mark size=2",
}
_DISTANCES = ["0", "1", "2", "3", "4"]


def fig_exp6_R(exp6: dict[str, Any]) -> str:
    """Generate Figure 6: R(d) groupplot for ResNet-18."""
    panels: list[str] = []
    for act, title in [("relu", "ReLU"), ("silu", "SiLU")]:
        ymin = "5e-6" if act == "relu" else "1e-7"
        blocks: list[str] = []
        for arch in ["resnet18", "plain_resnet18"]:
            for phase in ["init", "final"]:
                entry = exp6[act][arch][phase]
                pts: list[str] = []
                for d in _DISTANCES:
                    dist = entry["distances"][d]
                    pts.append(tikz_coord(int(d), dist["R_mean"], dist["R_std"]))
                coords = "\n        ".join(pts)
                style = _STYLES[(arch, phase)]
                legend = _ARCH_LEGEND[(arch, phase)]
                blocks.append(
                    f"    \\addplot[{style},\n"
                    f"      error bars/.cd, y dir=both, y explicit]"
                    f" coordinates {{\n        {coords}\n"
                    f"    }}; \\addlegendentry{{{legend}}}"
                )
        panel = (
            f"    \\nextgroupplot[title={{{title}}}, "
            f"ymin={ymin}, ymax=1e2]\n" + "\n".join(blocks)
        )
        panels.append(panel)

    plots = "\n".join(panels)
    return rf"""\begin{{figure*}}[t]
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
      xtick={{0,1,2,3,4}},
      xmin=-0.3, xmax=4.3,
      tick label style={{font=\footnotesize}},
      label style={{font=\small}},
      legend style={{font=\footnotesize, at={{(0.97,0.97)}}, anchor=north east}},
    ]
{plots}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Exp.\,6: Mean resonance $\bar{{\mathcal{{R}}}}(d)$ for
    ResNet-18 (CIFAR-10, log-scale on~$y$;
    $\pm 1\sigma$ over 5~seeds).
    \textbf{{(a)}}~ReLU: ResNet preserves
    $\bar{{\mathcal{{R}}}}$ at init
    ($R(0)/R(4)\!\approx\!1$); Plain decays by
    ${{\sim}}627\times$.
    \textbf{{(b)}}~SiLU: decay is stronger---Plain init
    $47\,000\times$; ResNet~$25\times$, slower than
    Plain (skip connections stabilize curvature).}}
  \label{{fig:exp6-R}}
\end{{figure*}}"""
