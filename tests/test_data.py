import numpy as np
import pytest

from hydraulic.config import SENSORS
from hydraulic.data import HydraulicDataset, load_dataset, make_synthetic, write_dataset


def test_synthetic_has_all_sensors_aligned(dataset):
    assert set(dataset.signals) == set(SENSORS)
    assert all(a.shape[0] == dataset.n_cycles for a in dataset.signals.values())


def test_synthetic_is_reproducible():
    a = make_synthetic(n_cycles=20, sample_scale=0.01, seed=7)
    b = make_synthetic(n_cycles=20, sample_scale=0.01, seed=7)
    np.testing.assert_array_equal(a.signals["PS1"], b.signals["PS1"])


def test_roundtrip_uci_file_layout(tmp_path):
    ds = make_synthetic(n_cycles=15, sample_scale=0.01, seed=3)
    write_dataset(ds, tmp_path)
    loaded = load_dataset(tmp_path)
    assert loaded.n_cycles == 15
    np.testing.assert_allclose(loaded.signals["TS1"], ds.signals["TS1"], atol=1e-3)
    assert (loaded.profile["valve"] == ds.profile["valve"]).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_validate_rejects_misaligned_cycles(dataset):
    signals = dict(dataset.signals)
    signals["PS1"] = signals["PS1"][:-1]
    with pytest.raises(ValueError, match="cycles"):
        HydraulicDataset(signals, dataset.profile).validate()


def test_validate_rejects_nan(dataset):
    signals = dict(dataset.signals)
    bad = signals["FS1"].copy()
    bad[0, 0] = np.nan
    signals["FS1"] = bad
    with pytest.raises(ValueError, match="NaN"):
        HydraulicDataset(signals, dataset.profile).validate()


def test_validate_rejects_missing_sensor(dataset):
    signals = {k: v for k, v in dataset.signals.items() if k != "VS1"}
    with pytest.raises(ValueError, match="VS1"):
        HydraulicDataset(signals, dataset.profile).validate()
