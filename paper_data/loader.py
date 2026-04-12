"""Load and cache experiment results.json files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DataPaths

log = logging.getLogger(__name__)

# Module-level cache: path -> parsed dict.
_cache: dict[Path, dict[str, Any]] = {}


def load(paths: DataPaths, experiment: str) -> dict[str, Any]:
    """Load *experiment* results, caching for repeated access."""
    p = paths.get(experiment).resolve()
    if p in _cache:
        return _cache[p]
    log.info("Loading %s from %s", experiment, p)
    with open(p) as f:
        data = json.load(f)
    _cache[p] = data
    return data


def clear_cache() -> None:
    """Drop all cached data (useful in tests)."""
    _cache.clear()
