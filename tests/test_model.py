import numpy as np
import pytest
from sklearn.model_selection import KFold

from hydraulic.config import TARGET_CV, TARGETS
from hydraulic.model import ConditionPredictor, make_cv, sensor_importance

CANDIDATES = {"logistic_regression", "random_forest", "extra_trees", "gradient_boosting"}


def test_report_covers_all_targets_and_models(trained):
    _, report = trained
    assert set(report.results) == set(TARGETS)
    for t, r in report.results.items():
        assert set(r.cv_scores) == CANDIDATES
        assert r.best_model == max(r.cv_scores, key=lambda m: r.cv_scores[m][0])
        assert r.cv_scheme == TARGET_CV[t]


def test_models_beat_chance_on_synthetic_data(trained):
    _, report = trained
    for target, r in report.results.items():
        chance = 1 / len(TARGETS[target])
        assert r.test_accuracy > chance + 0.2, target


def test_summary_has_test_metrics(trained):
    summary = trained[1].summary()
    assert list(summary.index) == list(TARGETS)
    assert {"best_model", "cv_accuracy", "test_accuracy", "test_f1_macro"} <= set(summary.columns)


def test_blocked_cv_keeps_time_order():
    cv = make_cv("blocked")
    assert isinstance(cv, KFold) and not cv.shuffle
    folds = [test for _, test in cv.split(np.arange(100))]
    assert all((np.diff(f) == 1).all() for f in folds)


def test_unknown_cv_scheme_raises():
    with pytest.raises(ValueError):
        make_cv("random")


def test_sensor_importance_sums_to_one(dataset, features):
    imp = sensor_importance(features, dataset.profile["cooler"].to_numpy())
    assert imp.sum() == pytest.approx(1.0)
    assert len(imp) == 17
    # the synthetic cooler fault only moves temperature and cooling signals
    assert imp.index[0] in {"TS1", "TS2", "TS3", "TS4", "CE", "CP"}


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
