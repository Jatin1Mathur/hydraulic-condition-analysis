"""Vectorised per-cycle features: mean, std, min, max for each sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_STATS, SENSORS

_STAT_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


def feature_names() -> list[str]:
    return [f"{s}_{stat}" for s in SENSORS for stat in FEATURE_STATS]


def cycle_features(signals: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per cycle, 17 sensors x 4 stats = 68 columns.

    Each statistic is computed over axis 1 in a single NumPy call, so there
    is no Python loop over cycles. Sensors with different sampling rates are
    aligned simply by summarising each cycle to fixed-size statistics.
    """
    cols: dict[str, np.ndarray] = {}
    for sensor in SENSORS:
        arr = np.asarray(signals[sensor], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{sensor}: expected (cycles, samples), got shape {arr.shape}")
        for stat in FEATURE_STATS:
            cols[f"{sensor}_{stat}"] = _STAT_FUNCS[stat](arr, axis=1)
    return pd.DataFrame(cols, columns=feature_names())
