import numpy as np
import pytest

from hydraulic.config import N_WINDOWS, SENSORS
from hydraulic.features import cycle_features, feature_names, sensor_of

N_PER_SENSOR = 4 + N_WINDOWS + 2


def test_shape_is_17_sensors_by_12_features(dataset, features):
    assert features.shape == (dataset.n_cycles, 17 * N_PER_SENSOR)
    assert list(features.columns) == feature_names()


def test_stats_match_hand_computed_values():
    ramp = np.arange(12, dtype=float)               # 0, 1, ..., 11
    flat = np.full(12, 10.0)
    signals = {s: np.vstack([ramp, flat]) for s in SENSORS}
    f = cycle_features(signals)

    assert f.loc[0, "PS1_mean"] == pytest.approx(5.5)
    assert f.loc[0, "PS1_std"] == pytest.approx(np.std(ramp))
    assert f.loc[0, "PS1_min"] == 0.0
    assert f.loc[0, "PS1_max"] == 11.0
    # 6 windows of 2 samples: (0,1) -> 0.5, ..., (10,11) -> 10.5
    assert [f.loc[0, f"PS1_w{i}_mean"] for i in range(6)] == pytest.approx(
        [0.5, 2.5, 4.5, 6.5, 8.5, 10.5]
    )
    # time axis runs -1..1 over 12 samples, so a 0..11 ramp has slope 11 / 2
    assert f.loc[0, "PS1_slope"] == pytest.approx(5.5)
    assert f.loc[0, "PS1_absdiff"] == pytest.approx(1.0)

    assert f.loc[1, "PS1_std"] == 0.0
    assert f.loc[1, "PS1_slope"] == pytest.approx(0.0)
    assert f.loc[1, "PS1_absdiff"] == 0.0


def test_window_means_capture_timing_that_plain_stats_miss():
    """Same values, different order: plain stats equal, window means differ."""
    early = np.r_[np.full(6, 100.0), np.zeros(6)]
    late = early[::-1]
    signals = {s: np.vstack([early, late]) for s in SENSORS}
    f = cycle_features(signals)
    for stat in ("mean", "std", "min", "max"):
        assert f.loc[0, f"PS1_{stat}"] == pytest.approx(f.loc[1, f"PS1_{stat}"])
    assert f.loc[0, "PS1_w0_mean"] != f.loc[1, "PS1_w0_mean"]
    assert f.loc[0, "PS1_slope"] < 0 < f.loc[1, "PS1_slope"]


def test_works_with_different_sampling_rates():
    signals = {s: np.ones((3, max(N_WINDOWS, n // 100))) for s, n in SENSORS.items()}
    assert cycle_features(signals).shape == (3, 17 * N_PER_SENSOR)


def test_rejects_1d_input():
    signals = {s: np.ones((2, 12)) for s in SENSORS}
    signals["TS1"] = np.ones(12)
    with pytest.raises(ValueError, match="TS1"):
        cycle_features(signals)


def test_rejects_too_few_samples():
    signals = {s: np.ones((2, 12)) for s in SENSORS}
    signals["VS1"] = np.ones((2, 3))
    with pytest.raises(ValueError, match="VS1"):
        cycle_features(signals)


def test_sensor_of():
    assert sensor_of("EPS1_w3_mean") == "EPS1"
    assert sensor_of("PS1_slope") == "PS1"
