"""LaTeX number formatting utilities.

Every formatter returns a *raw* LaTeX fragment (no surrounding ``$``);
callers wrap in math mode where needed.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Rounding helpers
# ---------------------------------------------------------------------------


def round_str(val: float, decimals: int) -> str:
    """Round *val* and return a fixed-width decimal string.

    >>> round_str(2.5926, 2)
    '2.59'
    """
    return f"{val:.{decimals}f}"


def sign_str(val: float, decimals: int) -> str:
    """Format with explicit LaTeX sign for negative values.

    Positive or zero values render plain; negative values use ``$-$`` prefix
    so that the minus sign is typeset as a math dash.

    >>> sign_str(-0.023, 2)
    '$-$0.02'
    >>> sign_str(0.04, 2)
    '0.04'
    """
    if val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Scientific notation
# ---------------------------------------------------------------------------


def sci(val: float, sig: int = 2) -> str:
    r"""Format *val* in LaTeX scientific notation.

    Returns e.g. ``1.51{\cdot}10^{-2}`` (without surrounding ``$``).
    *sig* is the number of significant digits in the mantissa.

    >>> sci(0.0151, 2)
    '1.51{\\cdot}10^{-2}'
    """
    if val == 0:
        return "0"
    exp = math.floor(math.log10(abs(val)))
    mantissa = val / 10**exp
    # Format mantissa with appropriate decimal places
    dec = sig  # digits after the decimal point in mantissa
    m_str = f"{mantissa:.{dec}f}"
    # Strip trailing zeros only beyond sig digits
    return rf"{m_str}{{\cdot}}10^{{{exp}}}"


def sci_or_plain(
    val: float, threshold: float = 0.01, sig: int = 2, decimals: int = 2
) -> str:
    """Use scientific notation below *threshold*, plain otherwise."""
    if abs(val) < threshold:
        return sci(val, sig)
    return round_str(val, decimals)


# ---------------------------------------------------------------------------
# +/- formatting
# ---------------------------------------------------------------------------


def pm(mean: float, std: float, decimals: int) -> str:
    r"""Format ``mean \pm std`` with fixed decimal places.

    >>> pm(1.65, 0.23, 2)
    '1.65 \\pm 0.23'
    """
    return rf"{mean:.{decimals}f} \pm {std:.{decimals}f}"


def pm_sci(mean: float, std: float, sig: int = 2) -> str:
    r"""Format ``mean \pm std`` using scientific notation for both."""
    return rf"{sci(mean, sig)} \pm {sci(std, sig)}"


# ---------------------------------------------------------------------------
# TikZ coordinate helpers
# ---------------------------------------------------------------------------


def _auto_fmt(val: float) -> str:
    """Pick a suitable representation for a tikz coordinate value."""
    av = abs(val)
    if av == 0:
        return "0"
    if av >= 0.1:
        # Use 4 decimal places for values >= 0.1
        return f"{val:.4f}"
    # Scientific notation for small values
    return f"{val:.4e}"


def tikz_coord(x: int | float, y: float, y_err: float | None = None) -> str:
    """Format a single PGFPlots coordinate, optionally with error bars.

    >>> tikz_coord(0, 1.0364e-01, 8.5838e-03)
    '(0,1.0364e-01) +- (0,8.5838e-03)'
    """
    x_s = str(x) if isinstance(x, int) else f"{x:.3f}"
    y_s = _auto_fmt(y)
    if y_err is not None:
        return f"({x_s},{y_s}) +- (0,{_auto_fmt(y_err)})"
    return f"({x_s},{y_s})"


def tikz_coords(points: list[tuple[int | float, float, float | None]]) -> str:
    """Format a list of *(x, y, y_err)* tuples into a PGFPlots coordinates string."""
    parts = []
    for pt in points:
        x, y = pt[0], pt[1]
        y_err = pt[2] if len(pt) > 2 else None
        parts.append(tikz_coord(x, y, y_err))
    return "\n        ".join(parts)


# ---------------------------------------------------------------------------
# Threshold / special formatting
# ---------------------------------------------------------------------------


def less_than(order: int, prefix: str = "") -> str:
    r"""Format as ``{<}\,10^{order}`` or ``{<}\,2{\cdot}10^{order}``.

    >>> less_than(-7)
    '{<}\\,10^{-7}'
    >>> less_than(-6, prefix="2")
    '{<}\\,2{\\cdot}10^{-6}'
    """
    if prefix:
        return rf"{{<}}\,{prefix}{{\cdot}}10^{{{order}}}"
    return rf"{{<}}\,10^{{{order}}}"


def wrap_math(s: str) -> str:
    """Wrap a LaTeX fragment in inline math mode."""
    return f"${s}$"
