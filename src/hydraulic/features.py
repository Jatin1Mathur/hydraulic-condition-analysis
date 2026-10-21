"""Vectorised per-cycle features.

Per sensor (17 sensors x 12 = 204 features):
  - whole-cycle statistics: mean, std, min, max
  - window means: the cycle split into N_WINDOWS equal time windows
  - slope: least-squares linear trend over the cycle
  - absdiff: mean absolute sample-to-sample change (roughness)

Summary statistics alone describe a signal's level and spread but not *when*
things happen. Valve lag changes how fast pressure builds up, so window means
and slope carry the information the plain statistics lose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_STATS, N_WINDOWS, SENSORS, SHAPE_STATS

_STAT_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


def feature_names() -> list[str]:
    names = []
    for s in SENSORS:
        names += [f"{s}_{stat}" for stat in FEATURE_STATS]
        names += [f"{s}_w{i}_mean" for i in range(N_WINDOWS)]
        names += [f"{s}_{stat}" for stat in SHAPE_STATS]
    return names


def sensor_of(feature: str) -> str:
    return feature.split("_", 1)[0]


def _sensor_features(name: str, arr: np.ndarray) -> dict[str, np.ndarray]:
    n = arr.shape[1]
    if n < max(2, N_WINDOWS):
        raise ValueError(f"{name}: need at least {max(2, N_WINDOWS)} samples per cycle, got {n}")
    out = {f"{name}_{stat}": _STAT_FUNCS[stat](arr, axis=1) for stat in FEATURE_STATS}

    for i, idx in enumerate(np.array_split(np.arange(n), N_WINDOWS)):
        out[f"{name}_w{i}_mean"] = arr[:, idx].mean(axis=1)

    t = np.linspace(-1.0, 1.0, n)
    centred = arr - arr.mean(axis=1, keepdims=True)
    out[f"{name}_slope"] = centred @ t / (t @ t)
    out[f"{name}_absdiff"] = np.abs(np.diff(arr, axis=1)).mean(axis=1)
    return out


def cycle_features(signals: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per cycle, 204 columns, computed without a loop over cycles.

    Sensors with different sampling rates are aligned because every sensor
    is reduced to the same fixed set of numbers per cycle.
    """
    cols: dict[str, np.ndarray] = {}
    for sensor in SENSORS:
        arr = np.asarray(signals[sensor], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{sensor}: expected (cycles, samples), got shape {arr.shape}")
        cols.update(_sensor_features(sensor, arr))
    return pd.DataFrame(cols, columns=feature_names())
