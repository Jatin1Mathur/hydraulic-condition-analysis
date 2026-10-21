import numpy as np
import pytest

from hydraulic.config import SENSORS
from hydraulic.features import cycle_features, feature_names


def test_shape_is_17_sensors_by_4_stats(dataset, features):
    assert features.shape == (dataset.n_cycles, 68)
    assert list(features.columns) == feature_names()


def test_stats_match_hand_computed_values():
    signals = {s: np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]]) for s in SENSORS}
    f = cycle_features(signals)
    assert f.loc[0, "PS1_mean"] == pytest.approx(2.5)
    assert f.loc[0, "PS1_std"] == pytest.approx(np.std([1, 2, 3, 4]))
    assert f.loc[0, "PS1_min"] == 1.0
    assert f.loc[0, "PS1_max"] == 4.0
    assert f.loc[1, "PS1_std"] == 0.0


def test_works_with_different_sampling_rates():
    signals ={s: np.ones((3, max(2, n // 100))) for s, n in SENSORS.items()}
    assert cycle_features(signals).shape == (3, 68)


def test_rejects_1d_input():
    signals = {s: np.ones((2, 5)) for s in SENSORS}
    signals["TS1"] = np.ones(5)
    with pytest.raises(ValueError, match="TS1"):
        cycle_features(signals)
