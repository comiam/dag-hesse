"""Tables for Experiment 1 / 1b: Plain vs Residual MLP.

Generated tables
----------------
- tab:rho-accuracy  - spectral norms, Lyapunov exponents, test accuracy
- tab:exp1b-supp    - same metrics under spectral normalization
- tab:exp1-deep     - extended depths L=16,32
"""

from __future__ import annotations

from typing import Any

from ..formatting import pm, round_str, sign_str

# ---------------------------------------------------------------------------
# tab:rho-accuracy
# ---------------------------------------------------------------------------


def tab_rho_accuracy(exp1: dict[str, Any]) -> str:
    """Generate Table 2: rho_max, lambda_1, Acc (Exp 1, depths 8/10/12)."""
    rows: list[str] = []
    for depth in [8, 10, 12]:
        d = str(depth)
        for arch, label in [("plain", "Plain"), ("residual", "Res.")]:
            vals: list[str] = [str(depth), label]
            for phase in ["init", "mid", "final"]:
                rho = exp1[d][arch][phase]["rho"]["rho_max_mean"]
                vals.append(round_str(rho, 2))
            for phase in ["init", "mid", "final"]:
                lam = exp1[d][arch][phase]["lyapunov"]["lyapunov_max_mean"]
                vals.append(sign_str(lam, 2))
            acc = exp1[d][arch]["final"]["test_acc_mean"]
            vals.append(round_str(acc * 100 if acc <= 1 else acc, 1))
            rows.append("    " + " & ".join(vals) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Spectral norms of Jacobians $\rho_{{\max}}$, Lyapunov
    exponent $\lambda_1$, and test accuracy for Exp.\,1
  configurations (CIFAR-10, mean over 5~seeds).}}
  \label{{tab:rho-accuracy}}
  \footnotesize
  \begin{{tabular}}{{@{{}}llccccccc@{{}}}}
    \toprule
    $L$ & Arch
    & $\rho_{{\max}}^{{\mathrm{{init}}}}$
    & $\rho_{{\max}}^{{\mathrm{{mid}}}}$
    & $\rho_{{\max}}^{{\mathrm{{final}}}}$
    & $\lambda_1^{{\mathrm{{init}}}}$
    & $\lambda_1^{{\mathrm{{mid}}}}$
    & $\lambda_1^{{\mathrm{{final}}}}$
    & Acc\,(\%) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp1b-supp
# ---------------------------------------------------------------------------


def tab_exp1b_supp(exp1b: dict[str, Any]) -> str:
    """Generate appendix Table: Exp 1b spectral normalization."""
    rows: list[str] = []
    for depth in [8, 12, 16]:
        d = str(depth)
        for arch, label in [("plain", "Plain SN"), ("residual", "Res.")]:
            vals: list[str] = [str(depth), label]
            for phase in ["init", "final"]:
                rho = exp1b[d][arch][phase]["rho"]["rho_max_mean"]
                vals.append(round_str(rho, 2))
            for phase in ["init", "final"]:
                lam = exp1b[d][arch][phase]["lyapunov"]["lyapunov_max_mean"]
                vals.append(sign_str(lam, 2))
            acc = exp1b[d][arch]["final"]["test_acc_mean"]
            vals.append(round_str(acc * 100 if acc <= 1 else acc, 1))
            rows.append("      " + " & ".join(vals) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.~1b: spectral normalization.
    Same metrics as Table~\ref{{tab:rho-accuracy}}.
    Plain~SN guarantees $\rho\!\leq\!1$ and $\lambda_1\!<\!0$;
    Residual~MLP as control without SN
  (CIFAR-10, mean over 5~seeds).}}
  \label{{tab:exp1b-supp}}
  {{\footnotesize
    \begin{{tabular}}{{@{{}}llccccc@{{}}}}
      \toprule
      $L$ & Arch.
      & $\rho_{{\max}}^{{\mathrm{{init}}}}$
      & $\rho_{{\max}}^{{\mathrm{{final}}}}$
      & $\lambda_1^{{\mathrm{{init}}}}$
      & $\lambda_1^{{\mathrm{{final}}}}$
      & Acc\,(\%) \\
      \midrule
{body}
      \bottomrule
  \end{{tabular}}}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp1-deep
# ---------------------------------------------------------------------------


def tab_exp1_deep(exp1a: dict[str, Any]) -> str:
    """Generate appendix Table: Exp 1 depths L=16,32."""
    rows: list[str] = []
    for depth in [16, 32]:
        d = str(depth)
        max_d = str(depth - 1)
        for arch, label in [("plain", "Plain"), ("residual", "Res.")]:
            for phase in ["init", "final"]:
                entry = exp1a[d][arch][phase]
                rho = entry["rho"]["rho_max_mean"]
                rho_s = entry["rho"]["rho_max_std"]
                r0 = entry["distances"]["0"]["R_mean"]
                r0_s = entry["distances"]["0"]["R_std"]
                r_max = entry["distances"][max_d]["R_mean"]
                r_max_s = entry["distances"][max_d]["R_std"]
                c_max = entry["distances"][max_d]["C_mean"]

                vals = [
                    str(depth),
                    label,
                    phase,
                    f"${pm(rho, rho_s, 2)}$",
                    f"${pm(r0, r0_s, 3)}$",
                    f"${pm(r_max, r_max_s, 2 if r_max >= 0.1 else 3)}$",
                    f"${round_str(c_max, 3)}$",
                ]
                rows.append("      " + " & ".join(vals) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.~1: metrics for depths $L\!\in\!\{{16,32\}}$
    (CIFAR-10, width$\,=128$, mean $\pm 1\sigma$ over 5~seeds;
      stochastic estimation: 100 Rademacher probes,
    subsample~64).
    $\bar{{\mathcal{{R}}}}(0)$/$\bar{{\mathcal{{R}}}}(L{{-}}1)$ ---
    resonance of nearest/most distant pairs;
    $\bar{{\mathcal{{C}}}}(L{{-}}1)$ --- coupling at maximum
  distance.}}
  \label{{tab:exp1-deep}}
  {{\footnotesize
    \begin{{tabular}}{{@{{}}llccccc@{{}}}}
      \toprule
      $L$ & Arch. & Phase
      & $\rho_{{\max}}$
      & $\bar{{\mathcal{{R}}}}(0)$
      & $\bar{{\mathcal{{R}}}}(L{{-}}1)$
      & $\bar{{\mathcal{{C}}}}(L{{-}}1)$ \\
      \midrule
{body}
      \bottomrule
  \end{{tabular}}}}
\end{{table}}"""
