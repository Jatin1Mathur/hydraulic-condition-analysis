import pytest

from hydraulic.data import make_synthetic
from hydraulic.features import cycle_features
from hydraulic.model import save_models, train_all


@pytest.fixture(scope="session")
def dataset():
    return make_synthetic(n_cycles=240, sample_scale=0.02, seed=1)


@pytest.fixture(scope="session")
def features(dataset):
    return cycle_features(dataset.signals)


@pytest.fixture(scope="session")
def trained(dataset, features):
    return train_all(features, dataset.profile)


@pytest.fixture(scope="session")
def model_path(trained, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "bundle.joblib"
    save_models(trained[0], path)
    return path
