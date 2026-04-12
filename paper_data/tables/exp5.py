"""Tables for Experiment 5: Toy-Attention vs ReLU-MLP.

Generated tables
----------------
- tab:exp5 - Frobenius norms and GN-Gap for Attention vs ReLU-MLP
"""

from __future__ import annotations

from typing import Any

from ..formatting import pm, sci

_MODELS = ["attention", "relu_mlp"]
_MODEL_LABELS = {"attention": "Attention", "relu_mlp": "ReLU-MLP"}
_PHASES = ["init", "mid", "final"]

# Threshold: below this, display Gap in scientific notation (no +/-).
_GAP_SCI_THRESH = 1.0


def tab_exp5(exp5: dict[str, Any]) -> str:
    """Generate Table 7: Exp 5 GN-Gap, Attention vs ReLU-MLP."""
    rows: list[str] = []
    for mi, model in enumerate(_MODELS):
        if mi > 0:
            rows.append(r"    \midrule")
        for phase in _PHASES:
            entry = exp5[model][phase]
            hf = entry["frob_full_mean"]
            hgn = entry["frob_gn_mean"]
            ht = entry["frob_tensor_mean"]
            gap = entry["gap_mean"]
            gap_s = entry["gap_std"]

            # Frobenius norms: always scientific notation
            hf_s = f"${sci(hf, 1)}$"
            hgn_s = f"${sci(hgn, 1)}$"
            ht_s = f"${sci(ht, 1)}$"

            # Gap: pm for attention (large), sci for relu_mlp (tiny)
            if gap >= _GAP_SCI_THRESH:
                gap_cell = f"${pm(gap, gap_s, 1)}$"
            else:
                gap_cell = f"${sci(gap, 1)}$"

            cells = [
                _MODEL_LABELS[model],
                phase,
                hf_s,
                hgn_s,
                ht_s,
                gap_cell,
            ]
            rows.append("    " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Exp.\,5: GN-Gap for Toy-Attention vs.\
    per-position ReLU-MLP
    ($d\!=\!16$, $S\!=\!8$, synthetic regression,
    mean$\,{{\pm}}\,1\sigma$ over 5~seeds).
    $\|H^T\|_F$: Frobenius norm of the tensor component;
    Gap $=\|H^T\|_F/\|H^{{GN}}\|_F$.}}
  \label{{tab:exp5}}
  \footnotesize
  \begin{{tabular}}{{@{{}}llcccc@{{}}}}
    \toprule
    Model & Checkpoint
    & $\|H^f\|_F$
    & $\|H^{{GN}}\|_F$
    & $\|H^T\|_F$
    & Gap \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""
