"""Model selection, evaluation and prediction for the four components.

Evaluation protocol per target:
  1. Hold out a stratified 20 % test set. It is used exactly once, at the end.
  2. On the remaining 80 %, compare candidate models with 5-fold
     cross-validation (time-blocked or stratified, see config.TARGET_CV).
  3. Pick the model with the best mean CV accuracy.
  4. Fit it on the 80 % and report accuracy and macro F1 on the test set.
  5. Refit on all data for deployment.

Choosing the model on CV and reporting on an untouched test set keeps the
reported number honest: the test set played no part in the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TARGET_CV, TARGETS
from .features import feature_names, sensor_of

N_FOLDS = 5


def candidate_models(seed: int = 42) -> dict[str, ClassifierMixin]:
    return {
        # linear baseline; needs scaled features
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000)
        ),
        # tree ensembles; scale-invariant
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def make_cv(scheme: str, seed: int = 42):
    if scheme == "blocked":
        return KFold(N_FOLDS, shuffle=False)
    if scheme == "stratified":
        return StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    raise ValueError(f"unknown CV scheme: {scheme}")


@dataclass
class TargetResult:
    target: str
    cv_scheme: str
    cv_scores: dict[str, tuple[float, float]]  # model -> (mean, std) CV accuracy
    best_model: str
    test_accuracy: float
    test_f1_macro: float
    confusion: np.ndarray
    labels: list[int]


@dataclass
class TrainingReport:
    results: dict[str, TargetResult] = field(default_factory=dict)

    def cv_table(self) -> pd.DataFrame:
        """Mean CV accuracy of every candidate, one row per target."""
        return pd.DataFrame(
            {t: {m: s[0] for m, s in r.cv_scores.items()} for t, r in self.results.items()}
        ).T.round(4)

    def summary(self) -> pd.DataFrame:
        rows = []
        for t, r in self.results.items():
            mean, std = r.cv_scores[r.best_model]
            rows.append({
                "component": t,
                "cv_scheme": r.cv_scheme,
                "best_model": r.best_model,
                "cv_accuracy": f"{mean:.4f} ± {std:.4f}",
                "test_accuracy": round(r.test_accuracy, 4),
                "test_f1_macro": round(r.test_f1_macro, 4),
            })
        return pd.DataFrame(rows).set_index("component")


def train_all(
    X: pd.DataFrame,
    profile: pd.DataFrame,
    targets: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, ClassifierMixin], TrainingReport]:
    targets = targets or list(TARGETS)
    X = X.reset_index(drop=True)
    best: dict[str, ClassifierMixin] = {}
    report = TrainingReport()

    for target in targets:
        y = profile[target].to_numpy()
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
        )
        train_idx.sort()  # keep time order so blocked folds are contiguous in time
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx], y[test_idx]

        scheme = TARGET_CV.get(target, "stratified")
        cv_scores = {}
        for name, model in candidate_models(seed).items():
            s = cross_val_score(model, X_tr, y_tr, cv=make_cv(scheme, seed))
            cv_scores[name] = (float(s.mean()), float(s.std()))
        winner = max(cv_scores, key=lambda m: cv_scores[m][0])

        model = candidate_models(seed)[winner].fit(X_tr, y_tr)
        pred = model.predict(X_te)
        labels = sorted(TARGETS[target])
        report.results[target] = TargetResult(
            target=target,
            cv_scheme=scheme,
            cv_scores=cv_scores,
            best_model=winner,
            test_accuracy=float(accuracy_score(y_te, pred)),
            test_f1_macro=float(f1_score(y_te, pred, average="macro")),
            confusion=confusion_matrix(y_te, pred, labels=labels),
            labels=labels,
        )

        best[target] = candidate_models(seed)[winner].fit(X, y)
    return best, report


def sensor_importance(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> pd.Series:
    """Share of extra-trees impurity importance per sensor (sums to 1)."""
    et = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    imp = pd.Series(et.feature_importances_, index=X.columns)
    return imp.groupby(imp.index.map(sensor_of)).sum().sort_values(ascending=False)


def save_models(models: dict[str, ClassifierMixin], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": feature_names(), "models": models}, path)


class ConditionPredictor:
    """Prediction logic, independent of any web framework."""

    def __init__(self, models: dict[str, ClassifierMixin], features: list[str]):
        self.models = models
        self.features = features

    @classmethod
    def load(cls, path: str | Path) -> "ConditionPredictor":
        bundle = joblib.load(path)
        return cls(bundle["models"], bundle["features"])

    def predict(self, feature_row: dict[str, float]) -> dict[str, dict]:
        missing = [f for f in self.features if f not in feature_row]
        if missing:
            raise KeyError(f"missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        X = pd.DataFrame([[feature_row[f] for f in self.features]], columns=self.features)
        out = {}
        for target, model in self.models.items():
            label = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            out[target] = {
                "label": label,
                "description": TARGETS[target][label],
                "confidence": round(float(proba.max()), 4),
            }
        return out
