"""Tables for Experiment 2: Bottleneck ablation.

Generated tables
----------------
- tab:exp2        - D_far, C_far, Acc for L=6
- tab:exp2-depth8 - same for L=8
"""

from __future__ import annotations

from typing import Any

from ..formatting import pm, round_str


def _exp2_rows(data: dict[str, Any], depth: str, widths: list[int]) -> list[str]:
    """Build table rows for one depth level."""
    rows: list[str] = []
    for du in widths:
        du_s = str(du)
        init = data[depth][du_s]["init"]
        final = data[depth][du_s]["final"]

        d_init = f"${pm(init['D_far_mean'], init['D_far_std'], 2)}$"
        d_final = f"${pm(final['D_far_mean'], final['D_far_std'], 2)}$"
        c_init = f"${pm(init['C_far_mean'], init['C_far_std'], 2)}$"
        c_final = f"${pm(final['C_far_mean'], final['C_far_std'], 2)}$"
        acc = round_str(
            (
                final["test_acc_mean"] * 100
                if final["test_acc_mean"] <= 1
                else final["test_acc_mean"]
            ),
            1,
        )

        # Mark control row and low-du note
        du_label = str(du)
        if du == 512:
            du_label = "$512^\\dagger$"
        row_parts = [du_label, d_init, d_final, c_init, c_final, acc]
        if du == 4:
            row_parts[-1] = f"{acc}$^{{\\ddagger}}$"
        rows.append("    " + " & ".join(row_parts) + r" \\")
        # Separator before control row
        if du == 256 and 512 in widths:
            rows.append(r"    \midrule")
    return rows


# ---------------------------------------------------------------------------
# tab:exp2
# ---------------------------------------------------------------------------


def tab_exp2(exp2: dict[str, Any]) -> str:
    """Generate Table 3: Exp 2 bottleneck ablation, L=6."""
    widths = [4, 8, 16, 32, 64, 128, 256, 512]
    available = [w for w in widths if str(w) in exp2.get("6", {})]
    rows = _exp2_rows(exp2, "6", available)
    body = "\n".join(rows)
    return rf"""\begin{{table}}[!t]
  \centering
  \caption{{Exp.\,2: bottleneck ablation ($L\!=\!6$, CIFAR-100).
    Stable rank $\mathcal{{D}}_{{\mathrm{{far}}}}$ and coupling
    $\mathcal{{C}}_{{\mathrm{{far}}}}$ vs.\ narrow layer
    width~$d_u$ (mean $\pm 1\sigma$ over 5~seeds).
    ${{}}^\dagger$\,Control: $d_{{\mathrm{{base}}}}\!=\!512$
    (stochastic estimate, 100~probes, subsample~64).
    ${{}}^\ddagger$\,Acc at uninformative model level
    ($d_u\!\ll\!K$); metrics for $d_u\!\geq\!8$ are
  preferred for conclusions.}}
  \label{{tab:exp2}}
  \footnotesize
  \begin{{tabular}}{{@{{}}rccccc@{{}}}}
    \toprule
    $d_u$
    & $\mathcal{{D}}_{{\mathrm{{far}}}}^{{\mathrm{{init}}}}$
    & $\mathcal{{D}}_{{\mathrm{{far}}}}^{{\mathrm{{final}}}}$
    & $\mathcal{{C}}_{{\mathrm{{far}}}}^{{\mathrm{{init}}}}$
    & $\mathcal{{C}}_{{\mathrm{{far}}}}^{{\mathrm{{final}}}}$
    & Acc\,(\%) \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp2-depth8
# ---------------------------------------------------------------------------


def tab_exp2_depth8(exp2: dict[str, Any]) -> str:
    """Generate appendix Table: Exp 2, L=8."""
    widths = [4, 8, 16, 32, 64, 128, 256, 512]
    available = [w for w in widths if str(w) in exp2.get("8", {})]
    rows = _exp2_rows(exp2, "8", available)
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.\,2: bottleneck ablation ($L\!=\!8$, CIFAR-100).
    Same protocol as Table~\ref{{tab:exp2}}
  (mean $\pm 1\sigma$ over 5~seeds).}}
  \label{{tab:exp2-depth8}}
  {{\footnotesize
    \begin{{tabular}}{{@{{}}rccccc@{{}}}}
      \toprule
      $d_u$
      & $\mathcal{{D}}_{{\mathrm{{far}}}}^{{\mathrm{{init}}}}$
      & $\mathcal{{D}}_{{\mathrm{{far}}}}^{{\mathrm{{final}}}}$
      & $\mathcal{{C}}_{{\mathrm{{far}}}}^{{\mathrm{{init}}}}$
      & $\mathcal{{C}}_{{\mathrm{{far}}}}^{{\mathrm{{final}}}}$
      & Acc\,(\%) \\
      \midrule
{body}
      \bottomrule
  \end{{tabular}}}}
\end{{table}}"""
