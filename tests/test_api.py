import pytest
from fastapi.testclient import TestClient

from hydraulic import api
from hydraulic.model import ConditionPredictor


@pytest.fixture
def client(model_path):
    predictor = ConditionPredictor.load(model_path)
    api.app.dependency_overrides[api.get_predictor] = lambda: predictor
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def _cycle(dataset, i=0):
    return {"signals": {k: v[i].tolist() for k, v in dataset.signals.items()}}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_cycle(client, dataset):
    r = client.post("/predict/cycle", json=_cycle(dataset))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"cooler", "valve", "pump", "accumulator"}
    assert body["pump"]["label"] in (0, 1, 2)


def test_predict_features(client, features):
    r = client.post("/predict/features", json={"features": features.iloc[3].to_dict()})
    assert r.status_code == 200


def test_missing_sensor_is_422(client, dataset):
    payload = _cycle(dataset)
    del payload["signals"]["PS1"]
    r = client.post("/predict/cycle", json=payload)
    assert r.status_code == 422
    assert "PS1" in r.text


def test_unknown_sensor_is_422(client, dataset):
    payload = _cycle(dataset)
    payload["signals"]["XX9"] = [1.0, 2.0]
    assert client.post("/predict/cycle", json=payload).status_code == 422


def test_too_few_samples_is_422(client, dataset):
    payload = _cycle(dataset)
    payload["signals"]["TS1"] = [40.0]
    assert client.post("/predict/cycle", json=payload).status_code == 422


def test_non_numeric_sample_is_422(client, dataset):
    payload = _cycle(dataset)
    payload["signals"]["TS1"][0] = "hot"
    assert client.post("/predict/cycle", json=payload).status_code == 422


def test_missing_features_is_422(client, features):
    row = features.iloc[0].to_dict()
    row.pop("VS1_max")
    r = client.post("/predict/features", json={"features": row})
    assert r.status_code == 422
    assert "VS1_max" in r.text


def test_no_model_is_503(monkeypatch, dataset):
    api.app.dependency_overrides.clear()
    api.get_predictor.cache_clear()
    monkeypatch.setattr(api, "MODEL_PATH", "does/not/exist.joblib")
    r = TestClient(api.app).post("/predict/cycle", json=_cycle(dataset))
    assert r.status_code == 503
