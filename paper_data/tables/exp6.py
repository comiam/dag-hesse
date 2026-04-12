"""Tables for Experiment 6: ResNet-18 (CIFAR-10).

Generated tables
----------------
- tab:exp6-gngap      - GN-Gap by distance (final)
- tab:exp6-main       - R(d) and C(d) at init
- tab:exp6-mid        - R, C, D at mid checkpoint
- tab:exp6-gngap-full - GN-Gap across all checkpoints
"""

from __future__ import annotations

import logging
from typing import Any

from ..formatting import pm, round_str, sci

log = logging.getLogger(__name__)


_ACTIVATIONS = ["relu", "silu"]
_ARCHS = ["resnet18", "plain_resnet18"]
_ARCH_SHORT = {"resnet18": "ResNet", "plain_resnet18": "Plain"}
_CONDITION_ORDER = [
    ("relu", "resnet18"),
    ("relu", "plain_resnet18"),
    ("silu", "resnet18"),
    ("silu", "plain_resnet18"),
]
_CONDITION_LABEL = {
    ("relu", "resnet18"): "ReLU ResNet",
    ("relu", "plain_resnet18"): "ReLU Plain",
    ("silu", "resnet18"): "SiLU ResNet",
    ("silu", "plain_resnet18"): "SiLU Plain",
}
_DISTANCES = ["0", "1", "2", "3", "4"]


def _gngap_cell(
    val: float, std: float, act: str, d: str, threshold: float = 1e-5
) -> str:
    """Format GN-Gap cell for exp6 tables."""
    # d=4 is always ~0 (linear head)
    if d == "4" and val < threshold:
        return r"${<}\,2{\cdot}10^{-6}$"
    if act == "relu":
        # ReLU: gap always ~0 (H^T = 0 a.e.); order matters: check tighter bound first
        gap_val = abs(val)
        if gap_val < 1e-6:
            return r"${<}\,10^{-6}$"
        if gap_val < 2e-6:
            return r"${<}\,2{\cdot}10^{-6}$"
        log.warning("ReLU gap %.2e exceeds expected threshold 2e-6", gap_val)
        return f"${sci(val, 1)}$"
    # SiLU: nonzero values
    if val < threshold:
        return r"${<}\,2{\cdot}10^{-6}$"
    return f"${pm(val, std, 2)}$"


def _gngap_cell_init(val: float, act: str, d: str, threshold: float = 1e-5) -> str:
    """Format GN-Gap cell for gngap-full table (no +/-, means only)."""
    if d == "4":
        return r"${<}10^{-5}$"
    if act == "relu":
        return r"${<}10^{-5}$"
    if val < threshold:
        return r"${<}10^{-5}$"
    return f"${round_str(val, 3)}$"


def _r_cell(val: float) -> str:
    """Format R(d) as scientific notation."""
    return f"${sci(val, 1)}$"


def _c_cell(val: float) -> str:
    """Format C(d), plain for close-to-1, threshold for small."""
    if val < 0.01:
        return r"${<}\,0.01$"
    return f"${round_str(val, 2)}$"


# ---------------------------------------------------------------------------
# tab:exp6-gngap
# ---------------------------------------------------------------------------


def tab_exp6_gngap(exp6: dict[str, Any]) -> str:
    """Generate Table 8: GN-Gap by distance (final checkpoint)."""
    rows: list[str] = []
    for d in _DISTANCES:
        d_label = "$4^\\dagger$" if d == "4" else f"${d}$"
        cells = [d_label]
        for act in _ACTIVATIONS:
            for arch in _ARCHS:
                entry = exp6[act][arch]["final"]["gn_gap_distances"][d]
                cells.append(_gngap_cell(entry["gap_mean"], entry["gap_std"], act, d))
        rows.append("    " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Exp.\,6: GN-Gap by distance~$d$ for ResNet-18
    (CIFAR-10, final checkpoint,
    mean$\,{{\pm}}\,1\sigma$ over 5~seeds).
    Row $d\!=\!4$: linear classifier
    (Remark~\ref{{rem:linear-head}}).}}
  \label{{tab:exp6-gngap}}
  \small
  \begin{{tabular}}{{@{{}}lcccc@{{}}}}
    \toprule
    $d$
    & \multicolumn{{2}}{{c}}{{ReLU}}
    & \multicolumn{{2}}{{c}}{{SiLU}} \\
    \cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
    & ResNet & Plain & ResNet & Plain \\
    \midrule
{body}
    \bottomrule
    \multicolumn{{5}}{{@{{}}l}}{{\footnotesize ${{}}^\dagger$ $H^T\!=\!0$ exactly (linear head); see Remark~\ref{{rem:linear-head}}.}}
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp6-main
# ---------------------------------------------------------------------------


def tab_exp6_main(exp6: dict[str, Any]) -> str:
    """Generate Table 9: R(d) and C(d) at initialization."""
    rows: list[str] = []
    for d in _DISTANCES:
        cells = [f"${d}$"]
        # R(d) columns: ReLU Res, ReLU Plain, SiLU Res, SiLU Plain
        for act in _ACTIVATIONS:
            for arch in _ARCHS:
                entry = exp6[act][arch]["init"]["distances"][d]
                cells.append(_r_cell(entry["R_mean"]))
        # C(d) columns: same order
        for act in _ACTIVATIONS:
            for arch in _ARCHS:
                entry = exp6[act][arch]["init"]["distances"][d]
                cells.append(_c_cell(entry["C_mean"]))
        rows.append("    " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Exp.\,6: mean resonance $\bar{{\mathcal{{R}}}}(d)$
    and coupling $\bar{{\mathcal{{C}}}}(d)$ at initialization
    (ResNet-18, CIFAR-10, mean over 5~seeds).}}
  \label{{tab:exp6-main}}
  \small
  \begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
    \toprule
    & \multicolumn{{4}}{{c}}{{$\bar{{\mathcal{{R}}}}(d)$}}
    & \multicolumn{{4}}{{c}}{{$\bar{{\mathcal{{C}}}}(d)$}} \\
    \cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}
    $d$
    & \multicolumn{{2}}{{c}}{{ReLU}} & \multicolumn{{2}}{{c}}{{SiLU}}
    & \multicolumn{{2}}{{c}}{{ReLU}} & \multicolumn{{2}}{{c}}{{SiLU}} \\
    \cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}
    & Res. & Plain & Res. & Plain
    & Res. & Plain & Res. & Plain \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp6-mid
# ---------------------------------------------------------------------------


def tab_exp6_mid(exp6: dict[str, Any]) -> str:
    """Generate appendix Table: Exp 6 metrics at mid checkpoint."""
    rows: list[str] = []
    for ci, (act, arch) in enumerate(_CONDITION_ORDER):
        if ci > 0:
            rows.append(r"    \midrule")
        label = _CONDITION_LABEL[(act, arch)]
        for d in _DISTANCES:
            entry = exp6[act][arch]["mid"]["distances"][d]
            r_val = round_str(entry["R_mean"], 2)
            c_val = round_str(entry["C_mean"], 2)
            d_val = round_str(entry["D_mean"], 2)

            cond_cell = label if d == "0" else ""
            cells = [cond_cell, d, f"${r_val}$", f"${c_val}$", f"${d_val}$"]
            rows.append("    " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    # Build accuracy note from mid data
    accs: list[str] = []
    for act, arch in _CONDITION_ORDER:
        label_short = _CONDITION_LABEL[(act, arch)]
        acc = exp6[act][arch]["mid"]["test_acc_mean"]
        acc_v = acc * 100 if acc <= 1 else acc
        accs.append(f"{label_short}~${round_str(acc_v, 1)}\\,\\%$")
    acc_note = ",\n    ".join(accs)

    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Exp.~6 (mid, epoch~50): metrics $\mathcal{{R}}$,
    $\mathcal{{C}}$, $\mathcal{{D}}$ by distance~$d$
    (mean over 5~seeds).
    Test accuracy: {acc_note}.}}
  \label{{tab:exp6-mid}}
  \small
  \begin{{tabular}}{{@{{}}llccc@{{}}}}
    \toprule
    Condition & $d$ & $\mathcal{{R}}(d)$ & $\mathcal{{C}}(d)$ & $\mathcal{{D}}(d)$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# tab:exp6-gngap-full
# ---------------------------------------------------------------------------


def tab_exp6_gngap_full(exp6: dict[str, Any]) -> str:
    """Generate appendix Table: GN-Gap across all checkpoints."""
    rows: list[str] = []
    for pi, phase in enumerate(["init", "mid", "final"]):
        if pi > 0:
            rows.append(r"    \midrule")
        for ci, (act, arch) in enumerate(_CONDITION_ORDER):
            label = _CONDITION_LABEL[(act, arch)]
            phase_cell = phase if ci == 0 else ""
            cells = [phase_cell, label]
            for d in _DISTANCES:
                entry = exp6[act][arch][phase]["gn_gap_distances"][d]
                cells.append(_gngap_cell_init(entry["gap_mean"], act, d))
            rows.append("    " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
  \centering
  \caption{{Exp.~6: GN-Gap$(d)$ across checkpoints
    (mean over 5~seeds).
    $^\dagger$~At $d\!=\!4$ the segment is purely linear
    ($H^T\!\equiv\!0$).}}
  \label{{tab:exp6-gngap-full}}
  \small
  \begin{{tabular}}{{@{{}}llccccc@{{}}}}
    \toprule
    Checkpoint & Condition & $d\!=\!0$ & $d\!=\!1$ & $d\!=\!2$ & $d\!=\!3$ & $d\!=\!4^\dagger$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""
