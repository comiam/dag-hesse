"""Tables for Experiment 3: GN-Gap by activation.

Generated tables
----------------
- tab:exp3b       - GN-Gap with LayerNorm (standard conditions)
- tab:exp3        - GN-Gap isolation protocol (no LayerNorm)
- tab:exp3-decomp - Frobenius decomposition by distance
- tab:ablation    - exact vs stochastic estimation
"""

from __future__ import annotations

import logging
from typing import Any

from ..formatting import less_than, pm, round_str, sci, wrap_math

log = logging.getLogger(__name__)


_ACTIVATIONS = ["relu", "leaky_relu", "softplus", "silu", "gelu"]
_ACT_LABELS = {
    "relu": "ReLU",
    "leaky_relu": "LeakyReLU",
    "softplus": "Softplus",
    "silu": "SiLU",
    "gelu": "GELU",
}
_SMOOTH = ["softplus", "silu", "gelu"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gap_cell(val: float, std: float, decimals: int = 3, eps: float = 1e-6) -> str:
    """Format a GN-Gap cell, using threshold notation for ~0 values."""
    if abs(val) < eps:
        if abs(val) >= 1e-7:
            log.warning(
                "Gap threshold notation < 10^{-7} used but |val|=%.2e", abs(val)
            )
        return wrap_math(less_than(-7))
    return f"${pm(val, std, decimals)}$"


def _gap_cell_approx(val: float, std: float, eps: float = 1e-6) -> str:
    """Like _gap_cell but uses ~< for leaky-relu-style near-zero."""
    if abs(val) < eps:
        return r"${\lesssim}\,10^{-7}$"
    return f"${pm(val, std, 3)}$"


def _delta_pct(init_val: float, final_val: float) -> str:
    """Compute relative change from init to final as ``$-$X.Y`` string."""
    if init_val == 0:
        return "---"
    delta = (final_val - init_val) / init_val * 100
    if delta < 0:
        return f"$-${abs(delta):.1f}"
    return f"{delta:.1f}"


# ---------------------------------------------------------------------------
# tab:exp3b  (with LayerNorm)
# ---------------------------------------------------------------------------


def tab_exp3b(exp3_ln: dict[str, Any]) -> str:
    """Generate Table 4: Exp 3b GN-Gap with LayerNorm."""
    rows: list[str] = []
    for act in _ACTIVATIONS:
        entry = exp3_ln[act]
        gi = entry["init"]["gap_global_mean"]
        gi_s = entry["init"]["gap_global_std"]
        gf = entry["final"]["gap_global_mean"]
        gf_s = entry["final"]["gap_global_std"]
        sigma = entry["init"]["sigma_pp_sq_mean"]
        acc = entry["final"]["test_acc_mean"]
        acc_s = entry["final"]["test_acc_std"]

        gap_init = _gap_cell(gi, gi_s)
        gap_final = _gap_cell(gf, gf_s)
        delta = _delta_pct(gi, gf) if gi > 1e-6 else "---"
        sigma_s = round_str(sigma, 3) if sigma > 1e-6 else "0"
        acc_cell = f"${pm(acc * 100 if acc <= 1 else acc, acc_s * 100 if acc_s <= 1 else acc_s, 1)}$"

        rows.append(
            "    "
            + " & ".join(
                [
                    _ACT_LABELS[act],
                    gap_init,
                    gap_final,
                    delta,
                    sigma_s,
                    acc_cell,
                ]
            )
            + r" \\"
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.\,3b: GN-Gap with LayerNorm
    ($L\!=\!6$, width$\,=64$, CIFAR-10,
    mean$\,{{\pm}}\,1\sigma$ over 5~seeds).
    $\Delta$: relative change init$\,{{\to}}\,$final.}}
  \label{{tab:exp3b}}
  \footnotesize
  \begin{{tabular}}{{@{{}}lccccc@{{}}}}
    \toprule
    Activation
    & $\mathrm{{Gap}}^{{\mathrm{{init}}}}$
    & $\mathrm{{Gap}}^{{\mathrm{{final}}}}$
    & $\Delta$\,(\%)
    & $\mathbb{{E}}[\sigma''^{{\,2}}]$
    & Acc\,(\%) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp3  (isolation protocol, no LayerNorm)
# ---------------------------------------------------------------------------


def tab_exp3(exp3: dict[str, Any]) -> str:
    """Generate Table 5: Exp 3 isolation protocol."""
    rows: list[str] = []
    for act in _ACTIVATIONS:
        entry = exp3[act]
        gi = entry["init"]["gap_global_mean"]
        gi_s = entry["init"]["gap_global_std"]
        gf = entry["final"]["gap_global_mean"]
        gf_s = entry["final"]["gap_global_std"]
        sigma = entry["init"]["sigma_pp_sq_mean"]
        acc = entry["final"]["test_acc_mean"]
        acc_s = entry["final"]["test_acc_std"]

        # ReLU: <10^-7; LeakyReLU: ~<10^-7; smooth: pm
        if act == "relu":
            gap_init = wrap_math(less_than(-7))
            gap_final = wrap_math(less_than(-7))
        elif act == "leaky_relu":
            gap_init = _gap_cell_approx(gi, gi_s)
            gap_final = _gap_cell_approx(gf, gf_s)
        else:
            gap_init = f"${pm(gi, gi_s, 3)}$"
            gap_final = f"${pm(gf, gf_s, 3)}$"

        delta = _delta_pct(gi, gf) if gi > 1e-6 else "---"
        sigma_s = round_str(sigma, 3) if sigma > 1e-6 else "0"
        acc_cell = f"${pm(acc * 100 if acc <= 1 else acc, acc_s * 100 if acc_s <= 1 else acc_s, 1)}$"

        rows.append(
            "    "
            + " & ".join(
                [
                    _ACT_LABELS[act],
                    gap_init,
                    gap_final,
                    delta,
                    sigma_s,
                    acc_cell,
                ]
            )
            + r" \\"
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.\,3: GN-Gap, isolation protocol (no LayerNorm,
    $L\!=\!6$, width$\,=64$, CIFAR-10,
    mean$\,{{\pm}}\,1\sigma$ over 5~seeds).}}
  \label{{tab:exp3}}
  \footnotesize
  \begin{{tabular}}{{@{{}}lccccc@{{}}}}
    \toprule
    Activation
    & $\mathrm{{Gap}}^{{\mathrm{{init}}}}$
    & $\mathrm{{Gap}}^{{\mathrm{{final}}}}$
    & $\Delta$\,(\%)
    & $\mathbb{{E}}[\sigma''^{{\,2}}]$
    & Acc\,(\%) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp3-decomp  (Frobenius norms by distance)
# ---------------------------------------------------------------------------


def tab_exp3_decomp(exp3: dict[str, Any]) -> str:
    """Generate appendix Table: Frobenius decomposition by distance."""
    rows: list[str] = []
    for d in [0, 2, 4, 5]:
        if d > 0:
            rows.append(r"      \midrule")
        for act in _SMOOTH:
            entry = exp3[act]["init"]["distances"][str(d)]
            ht = entry["frob_tensor_mean"]
            hgn = entry["frob_gn_mean"]
            gap = entry["gap_mean"]

            if d == 5:
                # Tensor vanishes at d=L-1
                ht_s = wrap_math(less_than(-10))
                gap_s = wrap_math(less_than(-7))
            else:
                ht_s = f"${sci(ht)}$"
                gap_s = f"${round_str(gap, 3)}$"
            hgn_s = f"${sci(hgn)}$"

            rows.append(
                "      "
                + " & ".join(
                    [
                        str(d),
                        _ACT_LABELS[act],
                        ht_s,
                        hgn_s,
                        gap_s,
                    ]
                )
                + r" \\"
            )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.~3: $\|H^T\|_F$ and $\|H^{{GN}}\|_F$ by
    distance~$d$ (init, $L\!=\!6$, width$\,=64$, CIFAR-10,
    mean over 5~seeds).
    At $d\!=\!5$ the tensor component vanishes
  ($H^T_{{d=L-1}}\!\equiv\!0$).}}
  \label{{tab:exp3-decomp}}
  {{\footnotesize
    \begin{{tabular}}{{@{{}}clccc@{{}}}}
      \toprule
      $d$ & Activation
      & $\|H^T\|_F$
      & $\|H^{{GN}}\|_F$
      & $\mathrm{{Gap}}_{{GN}}$ \\
      \midrule
{body}
      \bottomrule
  \end{{tabular}}}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:ablation  (exact vs stochastic)
# ---------------------------------------------------------------------------


def tab_ablation(exp3: dict[str, Any], exp3_appendix: dict[str, Any]) -> str:
    """Generate appendix Table: exact vs stochastic GN-Gap."""
    rows: list[str] = []
    for act in _SMOOTH:
        ge_i = exp3[act]["init"]["gap_global_mean"]
        gs_i = exp3_appendix[act]["init"]["gap_global_mean"]
        ge_f = exp3[act]["final"]["gap_global_mean"]
        gs_f = exp3_appendix[act]["final"]["gap_global_mean"]

        delta_i = abs(gs_i - ge_i) / ge_i * 100 if abs(ge_i) > 1e-12 else 0.0
        delta_f = abs(gs_f - ge_f) / ge_f * 100 if abs(ge_f) > 1e-12 else 0.0

        rows.append(
            "      "
            + " & ".join(
                [
                    _ACT_LABELS[act],
                    f"${round_str(ge_i, 3)}$",
                    f"${round_str(gs_i, 3)}$",
                    f"${round_str(delta_i, 2)}$",
                    f"${round_str(delta_f, 2)}$",
                ]
            )
            + r" \\"
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Ablation: GN-Gap discrepancy between exact and
    stochastic (30~probes) estimation
  (Exp.~3, $L\!=\!6$, width$\,=64$, mean over 5~seeds).}}
  \label{{tab:ablation}}
  {{\footnotesize
    \begin{{tabular}}{{@{{}}lcccc@{{}}}}
      \toprule
      Activation
      & $\mathrm{{Gap}}_{{\mathrm{{exact}}}}^{{\mathrm{{init}}}}$
      & $\mathrm{{Gap}}_{{\mathrm{{stoch}}}}^{{\mathrm{{init}}}}$
      & $\Delta^{{\mathrm{{init}}}}$\,(\%)
      & $\Delta^{{\mathrm{{final}}}}$\,(\%) \\
      \midrule
{body}
      \bottomrule
  \end{{tabular}}}}
\end{{table}}"""
