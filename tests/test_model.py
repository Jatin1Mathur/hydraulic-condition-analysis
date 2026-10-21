import pytest

from hydraulic.config import TARGETS
from hydraulic.model import ConditionPredictor


def test_report_covers_all_targets_and_models(trained):
    _, report = trained
    assert set(report.results) == set(TARGETS)
    for r in report.results.values():
        assert set(r.scores) == {"logistic_regression", "knn", "random_forest"}
        assert r.best_model == max(r.scores, key=r.scores.get)


def test_models_beat_chance_on_synthetic_data(trained):
    _, report = trained
    for target, r in report.results.items():
        chance = 1 / len(TARGETS[target])
        assert r.scores[r.best_model] > chance + 0.2, target


def test_predictor_roundtrip(model_path, features):
    predictor = ConditionPredictor.load(model_path)
    out = predictor.predict(features.iloc[0].to_dict())
    assert set(out) == set(TARGETS)
    for target, res in out.items():
        assert res["label"] in TARGETS[target]
        assert 0 <= res["confidence"] <= 1


def test_predictor_rejects_missing_features(model_path, features):
    predictor = ConditionPredictor.load(model_path)
    row = features.iloc[0].to_dict()
    del row["PS1_mean"]
    with pytest.raises(KeyError, match="PS1_mean"):
        predictor.predict(row)
