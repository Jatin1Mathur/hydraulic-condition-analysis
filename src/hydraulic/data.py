"""Loading the raw sensor files and generating a synthetic stand-in.

Raw format (UCI): one whitespace-separated text file per sensor, e.g.
PS1.txt with shape (n_cycles, 6000), plus profile.txt with shape (n_cycles, 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_CYCLES, PROFILE_COLUMNS, SENSORS, TARGETS


@dataclass
class HydraulicDataset:
    signals: dict[str, np.ndarray]  # sensor -> (n_cycles, n_samples)
    profile: pd.DataFrame           # condition labels per cycle

    @property
    def n_cycles(self) -> int:
        return len(self.profile)

    def validate(self) -> None:
        """Every sensor must be 2-D, aligned on cycles, and finite."""
        missing = set(SENSORS) - set(self.signals)
        if missing:
            raise ValueError(f"missing sensors: {sorted(missing)}")
        for name, arr in self.signals.items():
            if arr.ndim != 2:
                raise ValueError(f"{name}: expected 2-D array, got {arr.ndim}-D")
            if arr.shape[0] != self.n_cycles:
                raise ValueError(
                    f"{name}: {arr.shape[0]} cycles, profile has {self.n_cycles}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}: contains NaN or inf")


def load_dataset(data_dir: str | Path) -> HydraulicDataset:
    """Load the UCI files from `data_dir`."""
    data_dir = Path(data_dir)
    signals = {}
    for name in SENSORS:
        path = data_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; see README for the download link")
        # pandas' C parser is ~20x faster than np.loadtxt on the 90 MB files
        signals[name] = pd.read_csv(
            path, sep=r"\s+", header=None, dtype=np.float32, engine="c"
        ).to_numpy()
    profile = pd.read_csv(
        data_dir / "profile.txt", sep=r"\s+", header=None, names=PROFILE_COLUMNS
    )
    ds = HydraulicDataset(signals, profile)
    ds.validate()
    return ds


def make_synthetic(
    n_cycles: int = 600, sample_scale: float = 0.05, seed: int = 0
) -> HydraulicDataset:
    """Synthetic data with the same structure and plausible physics.

    Fault states shift the signals the way they do on the real rig:
    a weak cooler raises temperatures, valve lag slows pressure build-up,
    pump leakage lowers flow and raises motor power, and a weak accumulator
    leaves more pressure ripple. `sample_scale`
    shrinks the samples per cycle so tests run fast.
    """
    rng = np.random.default_rng(seed)
    profile = pd.DataFrame({
        "cooler": rng.choice(list(TARGETS["cooler"]), n_cycles),
        "valve": rng.choice(list(TARGETS["valve"]), n_cycles),
        "pump": rng.choice(list(TARGETS["pump"]), n_cycles),
        "accumulator": rng.choice([130, 115, 100, 90], n_cycles),
        "stable": rng.integers(0, 2, n_cycles),
    })

    cooler_deg = (100 - profile["cooler"].to_numpy()) / 97      # 0 good .. 1 failed
    valve_deg = (100 - profile["valve"].to_numpy()) / 27
    pump_deg = profile["pump"].to_numpy() / 2
    acc_deg = (130 - profile["accumulator"].to_numpy()) / 40   # 0 good .. 1 failed

    signals: dict[str, np.ndarray] = {}
    for name, full_len in SENSORS.items():
        n = max(10, int(full_len * sample_scale))
        t = np.linspace(0, 1, n)
        noise = rng.normal(0, 1, (n_cycles, n))

        if name.startswith("PS"):
            base = 150 if name in ("PS1", "PS2", "PS3") else 10
            rise = 1 - np.exp(-t[None, :] * (12 - 8 * valve_deg[:, None]))
            # a weak accumulator damps pressure pulsation less -> more ripple
            ripple = (1 + 6 * acc_deg[:, None]) * np.sin(2 * np.pi * 8 * t[None, :])
            sig = (base * (0.6 + 0.4 * rise) * (1 - 0.05 * pump_deg[:, None])
                   + ripple + 1.5 * noise)
        elif name == "EPS1":
            sig = 2400 + 120 * pump_deg[:, None] + 25 * noise
        elif name.startswith("FS"):
            base = 9 if name == "FS2" else 6
            sig = base * (1 - 0.12 * pump_deg[:, None]) + 0.15 * noise
        elif name.startswith("TS"):
            sig = 40 + 15 * cooler_deg[:, None] + 4 * t[None, :] + 0.3 * noise
        elif name == "VS1":
            sig = 0.55 + 0.08 * pump_deg[:, None] + 0.03 * valve_deg[:, None] + 0.02 * noise
        elif name == "CE":
            sig = 46 - 25 * cooler_deg[:, None] + 1.0 * noise
        elif name == "CP":
            sig = 2.2 - 1.2 * cooler_deg[:, None] + 0.08 * noise
        else:  # SE
            sig = 60 - 8 * pump_deg[:, None] - 4 * valve_deg[:, None] + 1.5 * noise
        signals[name] = sig.astype(np.float32)

    ds = HydraulicDataset(signals, profile)
    ds.validate()
    return ds


def write_dataset(ds: HydraulicDataset, out_dir: str | Path) -> None:
    """Write a dataset in the UCI file layout."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in ds.signals.items():
        np.savetxt(out_dir / f"{name}.txt", arr, fmt="%.3f", delimiter="\t")
    ds.profile.to_csv(out_dir / "profile.txt", sep="\t", header=False, index=False)


__all__ = ["HydraulicDataset", "load_dataset", "make_synthetic", "write_dataset", "N_CYCLES"]
