"""Bootstrap confidence intervals for scalar metrics.

Extracted from `scripts/run_task1_experiment.py` so binary-scoring code can
reuse the same machinery without importing the runner.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def bootstrap_mean_ci(
    values: Iterable[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Percentile-bootstrap mean and 95% CI.

    Filters None values; returns mean=None if no usable values, mean only
    (no CI) if only one usable value, full {mean, lo, hi, n} otherwise.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 2:
        return {
            "mean": float(arr.mean()) if arr.size == 1 else None,
            "lo": None,
            "hi": None,
            "n": int(arr.size),
        }
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1.0 - alpha)),
        "n": int(arr.size),
    }
