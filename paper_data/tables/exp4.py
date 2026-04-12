"""Tables for Experiment 4: Diamond MLP (merge-node GN-Gap).

Generated tables
----------------
- tab:exp4       - GN-Gap at merge node, k=2
- tab:exp4-sweep - GN-Gap for k=1,2,3
"""

from __future__ import annotations

from typing import Any

from ..formatting import pm, pm_sci, round_str, sci

_CONFIGS = ["sum_relu", "sum_silu", "cat_relu", "cat_silu"]
_CFG_SPLIT = {
    "sum_relu": ("sum", "ReLU"),
    "sum_silu": ("sum", "SiLU"),
    "cat_relu": ("cat", "ReLU"),
    "cat_silu": ("cat", "SiLU"),
}
_CFG_COMBINED = {
    "sum_relu": "sum+ReLU",
    "sum_silu": "sum+SiLU",
    "cat_relu": "cat+ReLU",
    "cat_silu": "cat+SiLU",
}

_ZERO_EPS = 1e-5  # values below this are treated as machine zero


def _gap_cell(val: float, std: float, *, show_pm: bool) -> str:
    """Format GN-Gap cell, choosing sci / plain depending on magnitude."""
    if abs(val) < _ZERO_EPS:
        return f"${sci(val, 1)}$"
    if not show_pm:
        dec = 2 if abs(val) >= 0.1 else 3
        return f"${round_str(val, dec)}$"
    dec = 2 if abs(val) >= 0.1 else 3
    return f"${pm(val, std, dec)}$"


def _tensor_cell(val: float, std: float, *, show_pm: bool) -> str:
    """Format Frobenius-norm cell (always sci notation)."""
    if abs(val) < _ZERO_EPS:
        return f"${sci(val, 1)}$"
    if not show_pm:
        return f"${sci(val, 1)}$"
    return f"${pm_sci(val, std, 1)}$"


# ---------------------------------------------------------------------------
# tab:exp4 (k=2)
# ---------------------------------------------------------------------------


def tab_exp4(exp4: dict[str, Any]) -> str:
    """Generate Table 6: Exp 4 GN-Gap at merge node, k=2."""
    k = "2"
    rows: list[str] = []
    for cfg in _CONFIGS:
        merge, act = _CFG_SPLIT[cfg]
        is_cs = cfg == "cat_silu"

        gi = exp4[k][cfg]["init"]["merge_gn_gap"]
        gf = exp4[k][cfg]["final"]["merge_gn_gap"]

        cells = [
            merge,
            act,
            _gap_cell(gi["gap_mean"], gi["gap_std"], show_pm=is_cs),
            _gap_cell(gf["gap_mean"], gf["gap_std"], show_pm=is_cs),
            _tensor_cell(gi["frob_tensor_mean"], gi["frob_tensor_std"], show_pm=is_cs),
            _tensor_cell(gf["frob_tensor_mean"], gf["frob_tensor_std"], show_pm=is_cs),
        ]
        rows.append("    " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.\,4: GN-Gap at the merge node of Diamond~MLP
    ($k\!=\!2$, width$\,{{=}}\,32$, CIFAR-10,
      $\mathrm{{Acc}}\approx 50\,\%$;
    mean $\pm 1\sigma$ over 5~seeds).
    $\|T_{{\mathrm{{merge}}}}\|_F$: Frobenius norm of the tensor
  component.}}
  \label{{tab:exp4}}
  \footnotesize
  \begin{{tabular}}{{@{{}}llcccc@{{}}}}
    \toprule
    Merge & $\sigma$
    & $\mathrm{{Gap}}^{{\mathrm{{init}}}}$
    & $\mathrm{{Gap}}^{{\mathrm{{final}}}}$
    & $\|T\|_F^{{\mathrm{{init}}}}$
    & $\|T\|_F^{{\mathrm{{final}}}}$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp4-sweep (k = 1, 2, 3)
# ---------------------------------------------------------------------------


def tab_exp4_sweep(exp4: dict[str, Any]) -> str:
    """Generate appendix Table: Exp 4 GN-Gap for k=1,2,3."""
    rows: list[str] = []
    for ki, k in enumerate(["1", "2", "3"]):
        if ki > 0:
            rows.append(r"      \midrule")
        for cfg in _CONFIGS:
            label = _CFG_COMBINED[cfg]
            gi = exp4[k][cfg]["init"]["merge_gn_gap"]
            gf = exp4[k][cfg]["final"]["merge_gn_gap"]

            cells = [
                k,
                label,
                _gap_cell(gi["gap_mean"], gi["gap_std"], show_pm=False),
                _gap_cell(gf["gap_mean"], gf["gap_std"], show_pm=False),
                _tensor_cell(
                    gi["frob_tensor_mean"], gi["frob_tensor_std"], show_pm=False
                ),
                _tensor_cell(
                    gf["frob_tensor_mean"], gf["frob_tensor_std"], show_pm=False
                ),
            ]
            rows.append("      " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.~4: GN-Gap at the merging node of Diamond~MLP
    for $k\!\in\!\{{1,2,3\}}$ (width$\,=32$, CIFAR-10,
  mean over 5~seeds).}}
  \label{{tab:exp4-sweep}}
  {{\footnotesize
    \begin{{tabular}}{{@{{}}clcccc@{{}}}}
      \toprule
      $k$ & Configuration
      & $\mathrm{{Gap}}^{{\mathrm{{init}}}}$
      & $\mathrm{{Gap}}^{{\mathrm{{final}}}}$
      & $\|T\|_F^{{\mathrm{{init}}}}$
      & $\|T\|_F^{{\mathrm{{final}}}}$ \\
      \midrule
{body}
      \bottomrule
  \end{{tabular}}}}
\end{{table}}"""
