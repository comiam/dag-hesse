"""Data path configuration and experiment registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Canonical results directory (relative to this file).
DEFAULT_RESULTS_DIR: Path = Path(__file__).resolve().parent.parent / "results"

# Default sub-paths inside the results directory (mirrors run_all.sh output layout).
EXPERIMENT_FILES: dict[str, str] = {
    "exp1": "exp1/results.json",
    "exp1b": "exp1b/results.json",
    "exp1_appendix": "exp1_appendix/results.json",
    "exp2_unified": "exp2_unified/results.json",
    "exp3": "exp3/results.json",
    "exp3_ln": "exp3_ln/results.json",
    "exp3_appendix": "exp3_appendix/results.json",
    "exp4": "exp4/results.json",
    "exp5": "exp5/results.json",
    "exp6": "exp6/results.json",
}


@dataclass
class DataPaths:
    """Resolve experiment data file paths.

    Parameters
    ----------
    results_dir : Path
        Root directory containing per-experiment subdirectories.
    overrides : dict[str, Path]
        Override specific experiment paths (e.g. when data is scattered).
    """

    results_dir: Path
    overrides: dict[str, Path] = field(default_factory=dict)

    def get(self, experiment: str) -> Path:
        """Return the resolved path for *experiment*."""
        if experiment in self.overrides:
            return self.overrides[experiment]
        if experiment not in EXPERIMENT_FILES:
            raise KeyError(f"Unknown experiment {experiment!r}")
        return self.results_dir / EXPERIMENT_FILES[experiment]
