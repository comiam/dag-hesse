"""CLI entry point: generate all LaTeX tables and TikZ figures.

Usage
-----
    # Uses canonical results from results/ by default
    python -m paper_data.main

    # Or specify a custom results directory
    python -m paper_data.main \\
        --results-dir /path/to/results

    # Generate only specific items
    python -m paper_data.main \\
        --only tab:exp4 fig:decay-R
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import DEFAULT_RESULTS_DIR, DataPaths
from .figures.exp1 import fig_coupling_C, fig_decay_R, fig_decay_R_sn
from .figures.exp2 import fig_dfar_bottleneck
from .figures.exp3 import fig_gn_gap_activation
from .figures.exp4 import fig_cross_branch_R, fig_diamond_gap
from .figures.exp6 import fig_exp6_R
from .loader import load
from .tables.exp1 import tab_exp1_deep, tab_exp1b_supp, tab_rho_accuracy
from .tables.exp2 import tab_exp2, tab_exp2_depth8
from .tables.exp3 import tab_ablation, tab_exp3, tab_exp3_decomp, tab_exp3b
from .tables.exp4 import tab_exp4, tab_exp4_sweep
from .tables.exp5 import tab_exp5
from .tables.exp6 import (
    tab_exp6_gngap,
    tab_exp6_gngap_full,
    tab_exp6_main,
    tab_exp6_mid,
)

log = logging.getLogger(__name__)

# Registry: label -> (generator_fn, required_experiments)
# Generator receives pre-loaded dicts in the same order as experiments list.
GeneratorEntry = tuple[Callable[..., str], list[str]]

REGISTRY: dict[str, GeneratorEntry] = {
    # --- tables ---
    "tab:rho-accuracy": (tab_rho_accuracy, ["exp1"]),
    "tab:exp1b-supp": (tab_exp1b_supp, ["exp1b"]),
    "tab:exp1-deep": (tab_exp1_deep, ["exp1_appendix"]),
    "tab:exp2": (tab_exp2, ["exp2_unified"]),
    "tab:exp2-depth8": (tab_exp2_depth8, ["exp2_unified"]),
    "tab:exp3b": (tab_exp3b, ["exp3_ln"]),
    "tab:exp3": (tab_exp3, ["exp3"]),
    "tab:exp3-decomp": (tab_exp3_decomp, ["exp3"]),
    "tab:ablation": (tab_ablation, ["exp3", "exp3_appendix"]),
    "tab:exp4": (tab_exp4, ["exp4"]),
    "tab:exp4-sweep": (tab_exp4_sweep, ["exp4"]),
    "tab:exp5": (tab_exp5, ["exp5"]),
    "tab:exp6-gngap": (tab_exp6_gngap, ["exp6"]),
    "tab:exp6-main": (tab_exp6_main, ["exp6"]),
    "tab:exp6-mid": (tab_exp6_mid, ["exp6"]),
    "tab:exp6-gngap-full": (tab_exp6_gngap_full, ["exp6"]),
    # --- figures ---
    "fig:coupling-C": (fig_coupling_C, ["exp1"]),
    "fig:decay-R": (fig_decay_R, ["exp1"]),
    "fig:decay-R-sn-supp": (fig_decay_R_sn, ["exp1b"]),
    "fig:dfar-bottleneck": (fig_dfar_bottleneck, ["exp2_unified"]),
    "fig:gn-gap-activation": (fig_gn_gap_activation, ["exp3"]),
    "fig:diamond-gap": (fig_diamond_gap, ["exp4"]),
    "fig:cross-branch-R": (fig_cross_branch_R, ["exp4"]),
    "fig:exp6-R": (fig_exp6_R, ["exp6"]),
}


def _sanitise_filename(label: str) -> str:
    """Convert a LaTeX label to a safe filename."""
    return label.replace(":", "_").replace(" ", "_") + ".tex"


def generate_all(
    paths: DataPaths, output_dir: Path, only: list[str] | None = None
) -> None:
    """Run generation pipeline and write .tex files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(REGISTRY) if only is None else only

    for label in labels:
        if label not in REGISTRY:
            log.warning("Unknown label %r -- skipping", label)
            continue
        fn, experiments = REGISTRY[label]
        data: list[dict[str, Any]] = [load(paths, e) for e in experiments]
        log.info("Generating %s", label)
        tex = fn(*data)
        out_path = output_dir / _sanitise_filename(label)
        out_path.write_text(tex + "\n", encoding="utf-8")
        log.info("  -> %s", out_path)


def _build_overrides(raw: list[str]) -> dict[str, Path]:
    """Parse ``key=path`` override pairs from CLI."""
    overrides: dict[str, Path] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be key=path, got: {item!r}")
        key, val = item.split("=", 1)
        overrides[key.strip()] = Path(val.strip())
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and TikZ figures from results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing per-experiment subdirectories "
        f"(default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write .tex files to.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Generate only these labels (e.g. tab:exp4 fig:decay-R).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override experiment paths: exp_name=/path/to/results.json",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_labels",
        help="List available labels and exit.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if args.list_labels:
        for label in REGISTRY:
            _, exps = REGISTRY[label]
            print(f"  {label:30s}  ({', '.join(exps)})")
        return

    overrides = _build_overrides(args.override)
    paths = DataPaths(results_dir=args.results_dir, overrides=overrides)
    generate_all(paths, args.output_dir, args.only)
    log.info(
        "Done. %d items generated.",
        len(REGISTRY) if args.only is None else len(args.only),
    )


if __name__ == "__main__":
    main()
